from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from dotenv import load_dotenv


ModelCapability = Literal["text", "vision"]
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / ".env")


class ModelConfigurationError(RuntimeError):
    """Raised when a requested model route is missing or incomplete."""


@dataclass(frozen=True)
class ProviderPreset:
    api_key_env: str
    base_url: str
    text_model: str
    vision_model: str


PROVIDER_PRESETS: dict[str, ProviderPreset] = {
    "deepseek": ProviderPreset(
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
        text_model="deepseek-v4-flash",
        vision_model="",
    ),
    "zhipu": ProviderPreset(
        api_key_env="ZHIPU_API_KEY",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        text_model="glm-4.7-flash",
        vision_model="glm-4.1v-thinking-flash",
    ),
    "qwen": ProviderPreset(
        api_key_env="DASHSCOPE_API_KEY",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        text_model="qwen-plus",
        vision_model="qwen3-vl-flash",
    ),
    "tencent": ProviderPreset(
        api_key_env="TENCENT_TOKENHUB_API_KEY",
        base_url="https://tokenhub.tencentmaas.com/v1",
        text_model="",
        vision_model="hy-vision-2.0-instruct",
    ),
    "openai": ProviderPreset(
        api_key_env="OPENAI_API_KEY",
        base_url="https://api.openai.com/v1",
        text_model="",
        vision_model="",
    ),
    "custom": ProviderPreset(
        api_key_env="CUSTOM_MODEL_API_KEY",
        base_url="",
        text_model="",
        vision_model="",
    ),
}


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


@dataclass(frozen=True)
class ModelRoute:
    capability: ModelCapability
    provider: str
    model: str
    base_url: str
    api_key_env: str
    api_key: str
    timeout_seconds: int
    max_retries: int
    max_output_tokens: int
    input_cost_per_million_cny: float
    output_cost_per_million_cny: float

    @property
    def configured(self) -> bool:
        return bool(
            self.provider != "disabled"
            and self.model
            and self.base_url
            and self.api_key
        )

    def public_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "api_key_configured": bool(self.api_key),
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "max_output_tokens": self.max_output_tokens,
            "input_cost_per_million_cny": self.input_cost_per_million_cny,
            "output_cost_per_million_cny": self.output_cost_per_million_cny,
            "configured": self.configured,
        }


class ModelRouter:
    """Resolve text and vision providers without exposing API keys to callers."""

    def __init__(self, routes: Mapping[ModelCapability, ModelRoute]) -> None:
        self._routes = dict(routes)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "ModelRouter":
        source = os.environ if env is None else env
        return cls(
            {
                "text": cls._route_from_env("text", source),
                "vision": cls._route_from_env("vision", source),
            }
        )

    @staticmethod
    def _route_from_env(
        capability: ModelCapability,
        env: Mapping[str, str],
    ) -> ModelRoute:
        prefix = f"AGENT_{capability.upper()}"
        default_provider = "deepseek" if capability == "text" else "disabled"
        provider = env.get(f"{prefix}_PROVIDER", default_provider).strip().lower()

        if provider == "disabled":
            return ModelRoute(
                capability=capability,
                provider="disabled",
                model="",
                base_url="",
                api_key_env="",
                api_key="",
                timeout_seconds=_env_int(env, f"{prefix}_TIMEOUT_SECONDS", 30, 1),
                max_retries=_env_int(env, f"{prefix}_MAX_RETRIES", 1, 0),
                max_output_tokens=_env_int(
                    env,
                    f"{prefix}_MAX_OUTPUT_TOKENS",
                    1200 if capability == "vision" else 800,
                    1,
                ),
                input_cost_per_million_cny=0,
                output_cost_per_million_cny=0,
            )

        preset = PROVIDER_PRESETS.get(provider)
        if preset is None:
            supported = ", ".join(sorted([*PROVIDER_PRESETS, "disabled"]))
            raise ModelConfigurationError(
                f"未知 {capability} Provider：{provider}。支持：{supported}。"
            )

        legacy_model = ""
        legacy_base_url = ""
        if provider == "deepseek":
            legacy_model = env.get("DEEPSEEK_MODEL", "")
            legacy_base_url = env.get("DEEPSEEK_BASE_URL", "")

        default_model = (
            preset.text_model if capability == "text" else preset.vision_model
        )
        api_key_env = env.get(
            f"{prefix}_API_KEY_ENV",
            preset.api_key_env,
        ).strip()
        model = env.get(
            f"{prefix}_MODEL",
            legacy_model or default_model,
        ).strip()
        base_url = env.get(
            f"{prefix}_BASE_URL",
            legacy_base_url or preset.base_url,
        ).strip().rstrip("/")

        return ModelRoute(
            capability=capability,
            provider=provider,
            model=model,
            base_url=base_url,
            api_key_env=api_key_env,
            api_key=env.get(api_key_env, "").strip() if api_key_env else "",
            timeout_seconds=_env_int(env, f"{prefix}_TIMEOUT_SECONDS", 30, 1),
            max_retries=_env_int(env, f"{prefix}_MAX_RETRIES", 1, 0),
            max_output_tokens=_env_int(
                env,
                f"{prefix}_MAX_OUTPUT_TOKENS",
                1200 if capability == "vision" else 800,
                1,
            ),
            input_cost_per_million_cny=_env_float(
                env,
                f"{prefix}_INPUT_COST_PER_1M_CNY",
                0,
                0,
            ),
            output_cost_per_million_cny=_env_float(
                env,
                f"{prefix}_OUTPUT_COST_PER_1M_CNY",
                0,
                0,
            ),
        )

    def get_route(self, capability: ModelCapability) -> ModelRoute:
        route = self._routes[capability]
        if route.provider == "disabled":
            raise ModelConfigurationError(f"{capability} 模型路由当前已关闭。")
        missing = []
        if not route.model:
            missing.append(f"AGENT_{capability.upper()}_MODEL")
        if not route.base_url:
            missing.append(f"AGENT_{capability.upper()}_BASE_URL")
        if not route.api_key:
            missing.append(route.api_key_env or f"{capability.upper()} API Key")
        if missing:
            raise ModelConfigurationError(
                f"{route.provider} {capability} 路由配置不完整，缺少："
                + "、".join(missing)
                + "。"
            )
        return route

    def create_chat_model(self, capability: ModelCapability) -> Any:
        route = self.get_route(capability)
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ModelConfigurationError(
                "langchain-openai 未安装，无法创建模型客户端。"
            ) from exc

        return ChatOpenAI(
            model=route.model,
            api_key=route.api_key,
            base_url=route.base_url,
            temperature=0,
            timeout=route.timeout_seconds,
            max_retries=0,
            max_tokens=route.max_output_tokens,
        )

    def describe(self) -> dict[str, dict[str, Any]]:
        return {
            capability: route.public_dict()
            for capability, route in self._routes.items()
        }
