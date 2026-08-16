from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from context_manager import estimate_tokens
from model_router import (
    ModelCapability,
    ModelConfigurationError,
    ModelRoute,
    ModelRouter,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_LOG_PATH = BASE_DIR / "logs" / "model_calls.jsonl"
_LOG_LOCK = threading.Lock()
_SEMAPHORE_LOCK = threading.Lock()
_SEMAPHORES: dict[int, threading.BoundedSemaphore] = {}
_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(
        r"(?i)\b(api[_ -]?key|authorization|bearer|token)\b"
        r"(\s*[:=]?\s*)([A-Za-z0-9._-]{8,})"
    ),
    re.compile(
        r"(?i)\b(api[_ -]?key|access[_ -]?token|token)\b"
        r"(\s*[:=]\s*)([^\s,;\"']+)"
    ),
)


class ModelBudgetExceeded(RuntimeError):
    """Raised before a call that would exceed the configured turn budget."""


class ModelInvocationError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        error_type: str,
        retryable: bool,
        snapshot: dict[str, Any],
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.retryable = retryable
        self.snapshot = snapshot


def _env_int(env: Mapping[str, str], name: str, default: int, minimum: int) -> int:
    try:
        return max(minimum, int(env.get(name, str(default))))
    except (TypeError, ValueError):
        return default


def _env_float(
    env: Mapping[str, str],
    name: str,
    default: float,
    minimum: float,
) -> float:
    try:
        return max(minimum, float(env.get(name, str(default))))
    except (TypeError, ValueError):
        return default


def _env_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    value = env.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class HarnessPolicy:
    max_model_calls: int = 2
    max_input_tokens: int = 6000
    max_reserved_output_tokens: int = 2000
    max_estimated_cost_cny: float = 0.25
    max_concurrency: int = 2
    backoff_seconds: float = 0.5
    log_enabled: bool = True
    log_path: Path = DEFAULT_MODEL_LOG_PATH

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
    ) -> "HarnessPolicy":
        source = os.environ if env is None else env
        configured_path = source.get("AGENT_MODEL_LOG_PATH", "").strip()
        log_path = Path(configured_path) if configured_path else DEFAULT_MODEL_LOG_PATH
        if not log_path.is_absolute():
            log_path = BASE_DIR / log_path
        return cls(
            max_model_calls=_env_int(source, "AGENT_MAX_MODEL_CALLS", 2, 0),
            max_input_tokens=_env_int(source, "AGENT_MAX_MODEL_INPUT_TOKENS", 6000, 0),
            max_reserved_output_tokens=_env_int(
                source,
                "AGENT_MAX_MODEL_OUTPUT_TOKENS",
                2000,
                0,
            ),
            max_estimated_cost_cny=_env_float(
                source,
                "AGENT_MAX_ESTIMATED_COST_CNY",
                0.25,
                0,
            ),
            max_concurrency=_env_int(source, "AGENT_MAX_MODEL_CONCURRENCY", 2, 1),
            backoff_seconds=_env_float(
                source,
                "AGENT_MODEL_BACKOFF_SECONDS",
                0.5,
                0,
            ),
            log_enabled=_env_bool(source, "AGENT_MODEL_LOG_ENABLED", True),
            log_path=log_path,
        )

    def public_dict(self) -> dict[str, Any]:
        return {
            "max_model_calls": self.max_model_calls,
            "max_input_tokens": self.max_input_tokens,
            "max_reserved_output_tokens": self.max_reserved_output_tokens,
            "max_estimated_cost_cny": self.max_estimated_cost_cny,
            "max_concurrency": self.max_concurrency,
            "backoff_seconds": self.backoff_seconds,
            "log_enabled": self.log_enabled,
            "log_path": str(self.log_path),
        }


@dataclass
class ModelLedger:
    trace_id: str
    request_id: str = ""
    thread_id: str = ""
    calls: int = 0
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    estimated_input_tokens: int = 0
    reserved_output_tokens: int = 0
    estimated_cost_cny: float = 0
    last_error_type: str = ""
    events: list[dict[str, Any]] = field(default_factory=list)

    def public_dict(
        self,
        *,
        policy: HarnessPolicy,
        route: ModelRoute | None = None,
    ) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "request_id": self.request_id,
            "thread_id": self.thread_id,
            "route": route.public_dict() if route else {},
            "calls": self.calls,
            "attempts": self.attempts,
            "successes": self.successes,
            "failures": self.failures,
            "estimated_input_tokens": self.estimated_input_tokens,
            "reserved_output_tokens": self.reserved_output_tokens,
            "estimated_cost_cny": round(self.estimated_cost_cny, 6),
            "last_error_type": self.last_error_type,
            "policy": policy.public_dict(),
            "events": list(self.events),
        }


def classify_model_error(exc: Exception) -> tuple[str, bool]:
    if isinstance(exc, ModelBudgetExceeded):
        return "budget_exceeded", False
    if isinstance(exc, ModelConfigurationError):
        return "configuration", False

    name = type(exc).__name__.lower()
    message = str(exc).lower()
    raw_status_code = getattr(exc, "status_code", None)
    if raw_status_code is None:
        response = getattr(exc, "response", None)
        raw_status_code = getattr(response, "status_code", None)
    try:
        status_code = int(raw_status_code) if raw_status_code is not None else None
    except (TypeError, ValueError):
        status_code = None

    if status_code in {401, 403} or any(
        marker in message
        for marker in ("invalid api key", "authentication", "unauthorized", "forbidden")
    ):
        return "authentication", False
    if status_code == 429 or "rate limit" in message or "ratelimit" in name:
        return "rate_limit", True
    if status_code is not None and status_code >= 500:
        return "provider_5xx", True
    if isinstance(exc, TimeoutError) or "timeout" in name or "timed out" in message:
        return "timeout", True
    if isinstance(exc, ConnectionError) or any(
        marker in name for marker in ("connection", "connecterror")
    ):
        return "connection", True
    if "json" in name or "json" in message or "validation" in name:
        return "invalid_response", True
    return "unknown", False


def sanitize_error_message(exc: Exception | str) -> str:
    message = str(exc)
    for pattern in _SECRET_PATTERNS:
        if pattern.groups >= 3:
            message = pattern.sub(r"\1\2[REDACTED]", message)
        else:
            message = pattern.sub("[REDACTED]", message)
    return message[:1000]


def _shared_semaphore(max_concurrency: int) -> threading.BoundedSemaphore:
    with _SEMAPHORE_LOCK:
        semaphore = _SEMAPHORES.get(max_concurrency)
        if semaphore is None:
            semaphore = threading.BoundedSemaphore(max_concurrency)
            _SEMAPHORES[max_concurrency] = semaphore
        return semaphore


class AgentHarness:
    """Apply model routing, budgets, retries, concurrency and safe telemetry."""

    def __init__(
        self,
        *,
        router: ModelRouter | None = None,
        policy: HarnessPolicy | None = None,
        request_id: str = "",
        thread_id: str = "",
        trace_id: str = "",
    ) -> None:
        self.router = router or ModelRouter.from_env()
        self.policy = policy or HarnessPolicy.from_env()
        self.ledger = ModelLedger(
            trace_id=trace_id or uuid.uuid4().hex,
            request_id=request_id,
            thread_id=thread_id,
        )
        self._last_route: ModelRoute | None = None

    def _estimated_cost(
        self,
        route: ModelRoute,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        return (
            input_tokens * route.input_cost_per_million_cny
            + output_tokens * route.output_cost_per_million_cny
        ) / 1_000_000

    def _reserve(
        self,
        *,
        route: ModelRoute,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        estimated_cost = self._estimated_cost(route, input_tokens, output_tokens)
        next_calls = self.ledger.calls + 1
        next_input = self.ledger.estimated_input_tokens + input_tokens
        next_output = self.ledger.reserved_output_tokens + output_tokens
        next_cost = self.ledger.estimated_cost_cny + estimated_cost

        if next_calls > self.policy.max_model_calls:
            raise ModelBudgetExceeded(
                f"模型调用次数预算不足：{next_calls}/{self.policy.max_model_calls}。"
            )
        if next_input > self.policy.max_input_tokens:
            raise ModelBudgetExceeded(
                f"模型输入预算不足：{next_input}/{self.policy.max_input_tokens} tokens。"
            )
        if next_output > self.policy.max_reserved_output_tokens:
            raise ModelBudgetExceeded(
                "模型输出预留预算不足："
                f"{next_output}/{self.policy.max_reserved_output_tokens} tokens。"
            )
        if (
            self.policy.max_estimated_cost_cny > 0
            and next_cost > self.policy.max_estimated_cost_cny
        ):
            raise ModelBudgetExceeded(
                "模型估算费用预算不足："
                f"{next_cost:.6f}/{self.policy.max_estimated_cost_cny:.6f} 元。"
            )

        self.ledger.calls = next_calls
        self.ledger.estimated_input_tokens = next_input
        self.ledger.reserved_output_tokens = next_output
        self.ledger.estimated_cost_cny = next_cost
        return estimated_cost

    def _append_log(self, event: dict[str, Any]) -> None:
        if not self.policy.log_enabled:
            return
        self.policy.log_path.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_LOCK:
            with self.policy.log_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")

    def snapshot(self) -> dict[str, Any]:
        return self.ledger.public_dict(policy=self.policy, route=self._last_route)

    def invoke(
        self,
        *,
        capability: ModelCapability,
        input_text: str,
        operation: Callable[[Any], Any],
        model_override: Any | None = None,
        reserved_output_tokens: int | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        try:
            route = (
                self.router.get_route(capability)
                if model_override is None
                else ModelRoute(
                    capability=capability,
                    provider="injected",
                    model=type(model_override).__name__,
                    base_url="in-process",
                    api_key_env="",
                    api_key="injected",
                    timeout_seconds=30,
                    max_retries=0,
                    max_output_tokens=reserved_output_tokens or 800,
                    input_cost_per_million_cny=0,
                    output_cost_per_million_cny=0,
                )
            )
        except Exception as exc:
            error_type, retryable = classify_model_error(exc)
            self.ledger.failures += 1
            self.ledger.last_error_type = error_type
            snapshot = self.snapshot()
            raise ModelInvocationError(
                sanitize_error_message(exc),
                error_type=error_type,
                retryable=retryable,
                snapshot=snapshot,
            ) from exc
        self._last_route = route
        input_tokens = estimate_tokens(input_text)
        output_tokens = reserved_output_tokens or route.max_output_tokens

        try:
            estimated_cost = self._reserve(
                route=route,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        except Exception as exc:
            error_type, retryable = classify_model_error(exc)
            self.ledger.failures += 1
            self.ledger.last_error_type = error_type
            snapshot = self.snapshot()
            raise ModelInvocationError(
                sanitize_error_message(exc),
                error_type=error_type,
                retryable=retryable,
                snapshot=snapshot,
            ) from exc

        try:
            model = model_override or self.router.create_chat_model(capability)
        except Exception as exc:
            error_type, retryable = classify_model_error(exc)
            self.ledger.failures += 1
            self.ledger.last_error_type = error_type
            snapshot = self.snapshot()
            raise ModelInvocationError(
                sanitize_error_message(exc),
                error_type=error_type,
                retryable=retryable,
                snapshot=snapshot,
            ) from exc
        max_attempts = route.max_retries + 1
        semaphore = _shared_semaphore(self.policy.max_concurrency)
        acquired = semaphore.acquire(timeout=route.timeout_seconds)
        if not acquired:
            exc = TimeoutError("等待模型并发槽位超时。")
            error_type, retryable = classify_model_error(exc)
            self.ledger.failures += 1
            self.ledger.last_error_type = error_type
            snapshot = self.snapshot()
            raise ModelInvocationError(
                sanitize_error_message(exc),
                error_type=error_type,
                retryable=retryable,
                snapshot=snapshot,
            ) from exc

        started = time.perf_counter()
        try:
            for attempt in range(1, max_attempts + 1):
                self.ledger.attempts += 1
                attempt_started = time.perf_counter()
                try:
                    result = operation(model)
                    latency_ms = round((time.perf_counter() - attempt_started) * 1000, 2)
                    self.ledger.successes += 1
                    event = {
                        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "trace_id": self.ledger.trace_id,
                        "request_id": self.ledger.request_id,
                        "thread_id": self.ledger.thread_id,
                        "capability": capability,
                        "provider": route.provider,
                        "model": route.model,
                        "status": "success",
                        "attempt": attempt,
                        "latency_ms": latency_ms,
                        "estimated_input_tokens": input_tokens,
                        "reserved_output_tokens": output_tokens,
                        "estimated_cost_cny": round(estimated_cost, 6),
                    }
                    self.ledger.events.append(event)
                    self._append_log(event)
                    return result, self.snapshot()
                except Exception as exc:
                    error_type, retryable = classify_model_error(exc)
                    latency_ms = round((time.perf_counter() - attempt_started) * 1000, 2)
                    event = {
                        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "trace_id": self.ledger.trace_id,
                        "request_id": self.ledger.request_id,
                        "thread_id": self.ledger.thread_id,
                        "capability": capability,
                        "provider": route.provider,
                        "model": route.model,
                        "status": "error",
                        "attempt": attempt,
                        "latency_ms": latency_ms,
                        "error_type": error_type,
                        "retryable": retryable,
                    }
                    self.ledger.events.append(event)
                    self._append_log(event)
                    if retryable and attempt < max_attempts:
                        time.sleep(self.policy.backoff_seconds * (2 ** (attempt - 1)))
                        continue

                    self.ledger.failures += 1
                    self.ledger.last_error_type = error_type
                    snapshot = self.snapshot()
                    raise ModelInvocationError(
                        sanitize_error_message(exc),
                        error_type=error_type,
                        retryable=retryable,
                        snapshot=snapshot,
                    ) from exc
        finally:
            semaphore.release()
            total_latency_ms = round((time.perf_counter() - started) * 1000, 2)
            if self.ledger.events:
                self.ledger.events[-1]["total_latency_ms"] = total_latency_ms
