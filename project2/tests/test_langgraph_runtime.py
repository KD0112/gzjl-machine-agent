from __future__ import annotations

import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch


PROJECT2_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT2_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT2_ROOT))

import agent_graph  # noqa: E402


QUOTE_QUESTION = "PC200原厂液压泵要1件，价格多少？"
INVENTORY_QUESTION = "小松PC200原厂液压泵要1件，有没有现货？"


class LangGraphRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.checkpoint_path = Path(self.temp_dir.name) / "checkpoints.sqlite3"
        self.savers = []
        self.graph = self._new_graph()

    def tearDown(self) -> None:
        for saver in self.savers:
            saver.conn.close()
        self.temp_dir.cleanup()

    def _new_graph(self):
        saver = agent_graph.create_sqlite_checkpointer(self.checkpoint_path)
        self.savers.append(saver)
        return agent_graph.build_graph(saver)

    def _thread_id(self, prefix: str) -> str:
        return f"{prefix}-{uuid.uuid4().hex}"

    def test_checkpoint_persists_and_resumes_in_new_graph(self) -> None:
        thread_id = self._thread_id("resume")
        paused = agent_graph.start_graph_agent(
            QUOTE_QUESTION,
            thread_id=thread_id,
            approval_mode="manual",
            graph=self.graph,
        )

        self.assertEqual(paused["status"], "waiting_approval")
        self.assertEqual(paused["approval_request"]["tool_name"], "quote_tool")
        self.assertTrue(self.checkpoint_path.exists())

        restarted_graph = self._new_graph()
        completed = agent_graph.resume_graph_agent(
            thread_id,
            "approve",
            comment="跨 Graph 实例恢复",
            graph=restarted_graph,
        )
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["called_tools"], ["quote_tool"])
        self.assertEqual(completed["approval_decisions"][0]["decision"], "approve")

        snapshot = agent_graph.get_graph_state(thread_id, graph=restarted_graph)
        history = agent_graph.get_graph_history(thread_id, graph=restarted_graph)
        self.assertEqual(snapshot["next"], [])
        self.assertGreaterEqual(len(history), 4)

    def test_human_can_edit_arguments_before_approval(self) -> None:
        thread_id = self._thread_id("edit")
        paused = agent_graph.start_graph_agent(
            QUOTE_QUESTION,
            thread_id=thread_id,
            approval_mode="manual",
            graph=self.graph,
        )
        edited_arguments = dict(paused["approval_request"]["arguments"])
        edited_arguments["quantity"] = 2

        completed = agent_graph.resume_graph_agent(
            thread_id,
            "edit",
            edited_arguments=edited_arguments,
            comment="客户确认改为两件",
            graph=self.graph,
        )
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["tool_arguments"]["quote_tool"]["quantity"], 2)
        self.assertEqual(completed["tool_results"]["quote_tool"]["total_price_range"], "57000-65600")
        self.assertEqual(completed["approval_decisions"][0]["decision"], "edit")

    def test_human_can_reject_tool_call(self) -> None:
        thread_id = self._thread_id("reject")
        agent_graph.start_graph_agent(
            QUOTE_QUESTION,
            thread_id=thread_id,
            approval_mode="manual",
            graph=self.graph,
        )
        completed = agent_graph.resume_graph_agent(
            thread_id,
            "reject",
            comment="客户暂不需要报价",
            graph=self.graph,
        )

        self.assertEqual(completed["status"], "completed")
        self.assertNotIn("quote_tool", completed["called_tools"])
        self.assertIn("quote_tool", completed["skipped_tools"])
        self.assertIn("没有获得人工批准", completed["customer_reply"])
        trace_steps = [item["step"] for item in completed["execution_trace"]]
        self.assertIn("human_approval", trace_steps)
        self.assertIn("skip_tool", trace_steps)

    def test_retry_policy_recovers_transient_tool_error(self) -> None:
        thread_id = self._thread_id("retry")
        original_execute = agent_graph.execute_tool_with_args
        attempts = {"count": 0}

        def flaky_execute(tool_name, arguments):
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise ConnectionError("temporary inventory connection failure")
            return original_execute(tool_name, arguments)

        with patch.object(agent_graph, "execute_tool_with_args", side_effect=flaky_execute):
            completed = agent_graph.start_graph_agent(
                INVENTORY_QUESTION,
                thread_id=thread_id,
                approval_mode="auto",
                graph=self.graph,
            )

        self.assertEqual(completed["status"], "completed")
        self.assertEqual(attempts["count"], 3)
        call_step = next(
            item for item in completed["execution_trace"] if item["step"] == "call_tool"
        )
        self.assertEqual(call_step["data"]["attempt"], 3)

    def test_non_retryable_error_routes_to_customer_fallback(self) -> None:
        thread_id = self._thread_id("error")
        with patch.object(
            agent_graph,
            "execute_tool_with_args",
            side_effect=ValueError("invalid inventory payload"),
        ):
            completed = agent_graph.start_graph_agent(
                INVENTORY_QUESTION,
                thread_id=thread_id,
                approval_mode="auto",
                graph=self.graph,
            )

        self.assertEqual(completed["status"], "completed_with_errors")
        self.assertEqual(completed["called_tools"], [])
        self.assertEqual(completed["tool_errors"]["inventory_tool"]["attempts"], 1)
        self.assertIn("暂时没有处理成功", completed["customer_reply"])
        self.assertIn(
            "tool_error",
            [item["step"] for item in completed["execution_trace"]],
        )

    def test_idempotency_record_reuses_existing_result(self) -> None:
        completed = agent_graph.start_graph_agent(
            INVENTORY_QUESTION,
            thread_id=self._thread_id("idempotent"),
            approval_mode="auto",
            graph=self.graph,
        )
        replay_state = dict(completed)
        replay_state["current_tool"] = "inventory_tool"
        replay_state["pending_tool_arguments"] = completed["tool_arguments"]["inventory_tool"]

        with patch.object(
            agent_graph,
            "execute_tool_with_args",
            side_effect=AssertionError("tool should not execute twice"),
        ):
            update = agent_graph.execute_tool_node(replay_state)

        self.assertEqual(update["execution_trace"][-1]["step"], "idempotent_reuse")


if __name__ == "__main__":
    unittest.main(verbosity=2)
