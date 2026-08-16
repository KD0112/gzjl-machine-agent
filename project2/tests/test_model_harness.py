from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from langchain_core.runnables import RunnableLambda

from agent_harness import (
    AgentHarness,
    HarnessPolicy,
    ModelInvocationError,
    sanitize_error_message,
)
from langchain_adapter import parse_customer_question_with_langchain
from model_router import (
    ModelConfigurationError,
    ModelRoute,
    ModelRouter,
)


class FakeRouter:
    def __init__(self, route: ModelRoute, model: object | None = None) -> None:
        self.route = route
        self.model = model or object()

    def get_route(self, _capability: str) -> ModelRoute:
        return self.route

    def create_chat_model(self, _capability: str) -> object:
        return self.model


class FakeStructuredModel:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def with_structured_output(self, schema):
        return RunnableLambda(lambda _prompt: schema.model_validate(self.payload))


def make_route(*, retries: int = 0) -> ModelRoute:
    return ModelRoute(
        capability="text",
        provider="fake",
        model="fake-model",
        base_url="https://example.invalid/v1",
        api_key_env="FAKE_KEY",
        api_key="secret-key",
        timeout_seconds=1,
        max_retries=retries,
        max_output_tokens=300,
        input_cost_per_million_cny=1,
        output_cost_per_million_cny=2,
    )


class ModelRouterTests(unittest.TestCase):
    def test_deepseek_route_keeps_legacy_environment_compatibility(self) -> None:
        router = ModelRouter.from_env(
            {
                "DEEPSEEK_API_KEY": "secret-deepseek-key",
                "DEEPSEEK_MODEL": "deepseek-test",
                "DEEPSEEK_BASE_URL": "https://deepseek.example/v1/",
            }
        )
        route = router.get_route("text")
        public = route.public_dict()

        self.assertEqual(route.provider, "deepseek")
        self.assertEqual(route.model, "deepseek-test")
        self.assertEqual(route.base_url, "https://deepseek.example/v1")
        self.assertTrue(public["api_key_configured"])
        self.assertNotIn("secret-deepseek-key", json.dumps(public))

    def test_vision_is_disabled_by_default(self) -> None:
        router = ModelRouter.from_env({})
        self.assertFalse(router.describe()["vision"]["configured"])
        with self.assertRaises(ModelConfigurationError):
            router.get_route("vision")

    def test_zhipu_free_vision_preset_only_needs_key(self) -> None:
        router = ModelRouter.from_env(
            {
                "AGENT_VISION_PROVIDER": "zhipu",
                "ZHIPU_API_KEY": "secret-zhipu-key",
            }
        )
        route = router.get_route("vision")

        self.assertEqual(route.provider, "zhipu")
        self.assertEqual(route.model, "glm-4.1v-thinking-flash")
        self.assertEqual(route.base_url, "https://open.bigmodel.cn/api/paas/v4")
        self.assertTrue(route.configured)


class AgentHarnessTests(unittest.TestCase):
    def test_error_message_redacts_api_credentials(self) -> None:
        message = sanitize_error_message(
            "Authorization: Bearer sk-secretcredential123456"
        )
        self.assertNotIn("secretcredential", message)
        self.assertIn("[REDACTED]", message)

    def test_error_message_redacts_short_named_api_key(self) -> None:
        message = sanitize_error_message(
            "request failed api_key=abcdef and token:123456"
        )
        self.assertNotIn("abcdef", message)
        self.assertNotIn("123456", message)
        self.assertEqual(message.count("[REDACTED]"), 2)

    def test_budget_blocks_call_before_operation(self) -> None:
        operation_called = False

        def operation(_model):
            nonlocal operation_called
            operation_called = True
            return "unexpected"

        harness = AgentHarness(
            router=FakeRouter(make_route()),
            policy=HarnessPolicy(max_model_calls=0, log_enabled=False),
        )
        with self.assertRaises(ModelInvocationError) as captured:
            harness.invoke(
                capability="text",
                input_text="测试",
                operation=operation,
            )

        self.assertFalse(operation_called)
        self.assertEqual(captured.exception.error_type, "budget_exceeded")

    def test_retryable_timeout_recovers_once(self) -> None:
        calls = 0

        def operation(_model):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise TimeoutError("provider timed out")
            return "ok"

        harness = AgentHarness(
            router=FakeRouter(make_route(retries=1)),
            policy=HarnessPolicy(backoff_seconds=0, log_enabled=False),
        )
        result, snapshot = harness.invoke(
            capability="text",
            input_text="测试超时恢复",
            operation=operation,
        )

        self.assertEqual(result, "ok")
        self.assertEqual(calls, 2)
        self.assertEqual(snapshot["attempts"], 2)
        self.assertEqual(snapshot["successes"], 1)
        self.assertEqual([item["status"] for item in snapshot["events"]], ["error", "success"])

    def test_invalid_structured_response_recovers_once(self) -> None:
        calls = 0

        def operation(_model):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ValueError("model did not return a JSON object")
            return {"status": "ok"}

        harness = AgentHarness(
            router=FakeRouter(make_route(retries=1)),
            policy=HarnessPolicy(backoff_seconds=0, log_enabled=False),
        )
        result, snapshot = harness.invoke(
            capability="vision",
            input_text="测试结构化响应恢复",
            operation=operation,
        )

        self.assertEqual(result, {"status": "ok"})
        self.assertEqual(calls, 2)
        self.assertEqual(snapshot["attempts"], 2)
        self.assertEqual(snapshot["events"][0]["error_type"], "invalid_response")

    def test_json_decode_error_is_retryable(self) -> None:
        calls = 0

        def operation(_model):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise json.JSONDecodeError(
                    "Invalid control character",
                    '{"value":"bad\ntext"}',
                    13,
                )
            return {"status": "ok"}

        harness = AgentHarness(
            router=FakeRouter(make_route(retries=1)),
            policy=HarnessPolicy(backoff_seconds=0, log_enabled=False),
        )
        result, snapshot = harness.invoke(
            capability="vision",
            input_text="测试 JSONDecodeError 恢复",
            operation=operation,
        )

        self.assertEqual(result, {"status": "ok"})
        self.assertEqual(calls, 2)
        self.assertEqual(snapshot["events"][0]["error_type"], "invalid_response")

    def test_authentication_failure_is_not_retried(self) -> None:
        calls = 0

        def operation(_model):
            nonlocal calls
            calls += 1
            raise RuntimeError("invalid api key")

        harness = AgentHarness(
            router=FakeRouter(make_route(retries=3)),
            policy=HarnessPolicy(backoff_seconds=0, log_enabled=False),
        )
        with self.assertRaises(ModelInvocationError) as captured:
            harness.invoke(
                capability="text",
                input_text="测试鉴权失败",
                operation=operation,
            )

        self.assertEqual(calls, 1)
        self.assertEqual(captured.exception.error_type, "authentication")
        self.assertFalse(captured.exception.retryable)

    def test_safe_jsonl_log_excludes_prompt_and_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "model_calls.jsonl"
            harness = AgentHarness(
                router=FakeRouter(make_route()),
                policy=HarnessPolicy(
                    log_enabled=True,
                    log_path=log_path,
                    backoff_seconds=0,
                ),
                request_id="request-001",
                thread_id="thread-001",
            )
            result, _snapshot = harness.invoke(
                capability="text",
                input_text="客户手机号 13800000000",
                operation=lambda _model: "ok",
            )
            payload = log_path.read_text(encoding="utf-8")

        self.assertEqual(result, "ok")
        self.assertIn("request-001", payload)
        self.assertNotIn("13800000000", payload)
        self.assertNotIn("secret-key", payload)

    def test_langchain_parse_exposes_harness_snapshot(self) -> None:
        model = FakeStructuredModel(
            {
                "intents": ["compatibility"],
                "slots": {
                    "brand": "小松",
                    "machine_model": "PC200",
                    "part_name": "液压泵",
                },
                "confidence": 0.9,
                "reason": "客户询问适配。",
            }
        )
        result = parse_customer_question_with_langchain(
            "这个能不能装？",
            mode="llm",
            model=model,
            request_id="request-parse",
            thread_id="thread-parse",
        )
        runtime = result["debug"]["model_runtime"]

        self.assertEqual(result["parse_source"], "langchain")
        self.assertEqual(runtime["request_id"], "request-parse")
        self.assertEqual(runtime["thread_id"], "thread-parse")
        self.assertEqual(runtime["route"]["provider"], "injected")
        self.assertEqual(runtime["successes"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
