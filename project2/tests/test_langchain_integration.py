from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.runnables import RunnableLambda

import agent_graph
from langchain_adapter import parse_customer_question_with_langchain
from langchain_tools import get_langchain_tool_map, get_langchain_tools
from schemas import AgentParsePlan
from tool_dispatcher import TOOL_REGISTRY, execute_tool_with_args
from tools.knowledge_tool import query_knowledge


class FakeStructuredModel:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def with_structured_output(self, schema):
        def invoke(_prompt):
            self.calls += 1
            return schema.model_validate(self.payload)

        return RunnableLambda(invoke)


class FailingModel:
    def with_structured_output(self, _schema):
        return RunnableLambda(lambda _prompt: (_ for _ in ()).throw(RuntimeError("model down")))


class LangChainIntegrationTests(unittest.TestCase):
    def test_hybrid_keeps_high_confidence_rule_result(self) -> None:
        model = FailingModel()
        result = parse_customer_question_with_langchain(
            "小松PC200原厂液压泵要1件，多少钱？",
            mode="hybrid",
            model=model,
        )
        self.assertEqual(result["parse_source"], "rules")
        self.assertEqual(result["intents"], ["quote"])
        self.assertGreaterEqual(result["confidence"], 0.65)

    def test_low_confidence_question_uses_langchain_structured_output(self) -> None:
        model = FakeStructuredModel(
            {
                "intents": ["compatibility"],
                "slots": {
                    "brand": "小松",
                    "machine_model": "PC200",
                    "part_name": "液压泵",
                },
                "confidence": 0.88,
                "reason": "客户询问配件是否适配。",
            }
        )
        result = parse_customer_question_with_langchain(
            "帮我看看这个能不能装",
            mode="hybrid",
            model=model,
        )
        self.assertEqual(model.calls, 1)
        self.assertEqual(result["parse_source"], "hybrid_langchain")
        self.assertEqual(result["intents"], ["compatibility"])
        self.assertEqual(result["slots"]["machine_model"], "PC200")
        self.assertEqual(result["confidence"], 0.88)

    def test_langchain_failure_falls_back_to_rules(self) -> None:
        result = parse_customer_question_with_langchain(
            "这个怎么处理",
            mode="llm",
            model=FailingModel(),
        )
        self.assertEqual(result["parse_source"], "rules_fallback")
        self.assertIn("model down", result["debug"]["langchain_error"])

    def test_existing_functions_are_exposed_as_structured_tools(self) -> None:
        tools = get_langchain_tools()
        names = {tool.name for tool in tools}
        self.assertEqual(
            names,
            {
                "inventory_tool",
                "quote_tool",
                "logistics_tool",
                "ticket_tool",
                "knowledge_tool",
            },
        )
        self.assertTrue(all(tool.args_schema is not None for tool in tools))

    def test_dispatcher_executes_the_canonical_structured_tool(self) -> None:
        self.assertIs(TOOL_REGISTRY, get_langchain_tool_map())
        result = execute_tool_with_args(
            "inventory_tool",
            {
                "brand": "小松",
                "machine_model": "PC200",
                "part_name": "液压泵",
                "quality_level": "原厂",
            },
        )
        self.assertTrue(result["matched"])
        self.assertTrue(result["in_stock"])
        self.assertGreater(result["stock_count"], 0)

    def test_knowledge_tool_serializes_rag_sources(self) -> None:
        fake_doc = SimpleNamespace(
            page_content="PC200 液压泵适配需要核对铭牌和零件号。",
            metadata={
                "source": "docs/配件适配说明.md",
                "retrieval_rank": 1,
                "retrieval_distance": 0.32,
            },
        )
        fake_rag = SimpleNamespace(
            DEEPSEEK_API_KEY="test-key",
            answer_with_metadata=lambda question, k: {
                "answer": "建议先核对设备铭牌、旧件照片和零件号。",
                "answer_source": "llm",
                "docs": [fake_doc],
                "retrieval": {
                    "status": "ok",
                    "reason": "命中",
                    "top_distance": 0.32,
                    "max_distance": 1.0,
                },
            }
        )
        with patch("tools.knowledge_tool._load_rag_api", return_value=fake_rag):
            result = query_knowledge("PC200液压泵能不能适配？", top_k=3)

        self.assertTrue(result["matched"])
        self.assertFalse(result["needs_handoff"])
        self.assertEqual(result["sources"][0]["source_name"], "配件适配说明.md")

    def test_graph_adds_knowledge_tool_only_when_enabled(self) -> None:
        parse_state = {
            "question": "PC200液压泵能不能适配？",
            "thread_id": "knowledge-route",
            "request_id": "knowledge-route",
            "approval_mode": "auto",
            "handoff_mode": "off",
            "parser_mode": "rules",
            "knowledge_mode": True,
        }
        update = agent_graph.parse_node(parse_state)
        self.assertIn("knowledge_tool", update["tool_queue"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
