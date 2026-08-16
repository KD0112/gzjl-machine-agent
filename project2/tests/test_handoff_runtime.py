from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import agent_graph
from handoff_repository import HandoffRepository


AFTER_SALES_QUESTION = "订单号 A20260616001，买错了能不能退货？"
MISSING_INFO_QUESTION = "PC200液压泵多少钱？"


class HandoffRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.checkpointer = agent_graph.create_sqlite_checkpointer(root / "checkpoints.sqlite3")
        self.repository = HandoffRepository(root / "handoff.sqlite3")
        self.graph = agent_graph.build_graph(self.checkpointer, self.repository)

    def tearDown(self) -> None:
        self.checkpointer.conn.close()
        self.temp_dir.cleanup()

    def test_explicit_human_request_creates_case_and_resumes(self) -> None:
        started = agent_graph.start_graph_agent(
            "我要找人工客服确认PC200液压泵",
            thread_id="handoff-explicit",
            approval_mode="auto",
            handoff_mode="manual",
            graph=self.graph,
        )

        self.assertEqual(started["status"], "waiting_human")
        self.assertTrue(started["handoff_id"].startswith("HF-"))
        case = self.repository.get_case(started["handoff_id"])
        self.assertIsNotNone(case)
        self.assertEqual(case["status"], "queued")
        self.assertEqual(case["reason_code"], "explicit_human_request")
        self.assertIn("parse_result", case["context"])

        self.repository.claim_case(started["handoff_id"], "客服小王")
        completed = agent_graph.resume_handoff_agent(
            "handoff-explicit",
            "您好，我是客服小王，已经接手并帮您继续核对。",
            agent_name="客服小王",
            graph=self.graph,
        )

        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["assigned_agent"], "客服小王")
        self.assertIn("已经接手", completed["customer_reply"])
        self.assertEqual(
            self.repository.get_case(started["handoff_id"])["status"],
            "resolved",
        )
        self.assertIn(
            "human_response",
            [item["step"] for item in completed["execution_trace"]],
        )

    def test_after_sales_approval_then_handoff_are_separate_interrupts(self) -> None:
        waiting_approval = agent_graph.start_graph_agent(
            AFTER_SALES_QUESTION,
            thread_id="handoff-after-sales",
            approval_mode="manual",
            handoff_mode="manual",
            graph=self.graph,
        )
        self.assertEqual(waiting_approval["status"], "waiting_approval")
        self.assertEqual(
            waiting_approval["approval_request"]["kind"],
            "tool_approval",
        )

        waiting_human = agent_graph.resume_graph_agent(
            "handoff-after-sales",
            "approve",
            graph=self.graph,
        )
        self.assertEqual(waiting_human["status"], "waiting_human")
        self.assertIn("ticket_tool", waiting_human["called_tools"])
        self.assertEqual(
            waiting_human["handoff_reason"]["reason_code"],
            "after_sales_review",
        )
        self.assertEqual(waiting_human["interrupts"][0]["kind"], "human_response")

    def test_tool_error_routes_to_human_when_enabled(self) -> None:
        with patch.object(
            agent_graph,
            "execute_tool_with_args",
            side_effect=ValueError("inventory payload failed"),
        ):
            result = agent_graph.start_graph_agent(
                "小松PC200原厂液压泵有没有现货？",
                thread_id="handoff-tool-error",
                approval_mode="auto",
                handoff_mode="manual",
                graph=self.graph,
            )

        self.assertEqual(result["status"], "waiting_human")
        self.assertEqual(result["handoff_reason"]["reason_code"], "tool_failure")
        self.assertIn("inventory_tool", result["tool_errors"])

    def test_repeated_missing_information_routes_to_human(self) -> None:
        result = agent_graph.start_graph_agent(
            MISSING_INFO_QUESTION,
            thread_id="handoff-repeated-missing",
            approval_mode="auto",
            handoff_mode="manual",
            clarification_count=2,
            graph=self.graph,
        )
        self.assertEqual(result["status"], "waiting_human")
        self.assertEqual(
            result["handoff_reason"]["reason_code"],
            "repeated_missing_information",
        )

    def test_handoff_can_be_disabled_for_deterministic_baseline(self) -> None:
        result = agent_graph.start_graph_agent(
            "我要找人工客服确认PC200液压泵",
            thread_id="handoff-disabled",
            approval_mode="auto",
            handoff_mode="off",
            graph=self.graph,
        )
        self.assertEqual(result["status"], "completed")
        self.assertFalse(result["handoff_id"])
        self.assertEqual(self.repository.list_cases(), [])

    def test_wechat_human_reply_enters_outbox_once(self) -> None:
        started = agent_graph.start_graph_agent(
            "转人工客服",
            thread_id="handoff-wechat",
            approval_mode="auto",
            handoff_mode="manual",
            channel="wechat",
            customer_id="openid-demo-001",
            graph=self.graph,
        )
        agent_graph.resume_handoff_agent(
            "handoff-wechat",
            "您好，人工客服已接手。",
            agent_name="客服小李",
            graph=self.graph,
        )
        self.repository.resolve_case(
            started["handoff_id"],
            human_reply="您好，人工客服已接手。",
            agent_name="客服小李",
        )

        outbox = self.repository.list_outbox("pending")
        self.assertEqual(len(outbox), 1)
        self.assertEqual(outbox[0]["channel"], "wechat")
        self.assertEqual(outbox[0]["customer_id"], "openid-demo-001")


if __name__ == "__main__":
    unittest.main(verbosity=2)
