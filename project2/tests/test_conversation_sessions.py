from __future__ import annotations

import sys
import tempfile
import unittest
import uuid
from pathlib import Path


PROJECT2_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT2_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT2_ROOT))

import agent_graph  # noqa: E402
from conversation_repository import (  # noqa: E402
    ConversationAccessError,
    ConversationRepository,
)
from handoff_repository import HandoffRepository  # noqa: E402
from memory_repository import MemoryRepository  # noqa: E402


FIRST_TURN = "小松PC200原厂液压泵要1件，有没有现货？"
FOLLOW_UP = "这个多少钱？"


class ConversationSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.repository = ConversationRepository(self.root / "conversations.sqlite3")
        self.checkpoint_path = self.root / "checkpoints.sqlite3"
        self.memory_repository = MemoryRepository(self.root / "memory.sqlite3")
        self.handoff_repository = HandoffRepository(self.root / "handoff.sqlite3")
        self.savers = []

    def tearDown(self) -> None:
        for saver in self.savers:
            saver.conn.close()
        self.temp_dir.cleanup()

    def _new_graph(self):
        saver = agent_graph.create_sqlite_checkpointer(self.checkpoint_path)
        self.savers.append(saver)
        return agent_graph.build_graph(
            saver,
            self.handoff_repository,
            self.memory_repository,
        )

    @staticmethod
    def _thread(prefix: str) -> str:
        return f"{prefix}-{uuid.uuid4().hex}"

    @staticmethod
    def _result(thread_id: str, *, turn_count: int = 1) -> dict:
        return {
            "thread_id": thread_id,
            "request_id": uuid.uuid4().hex,
            "status": "completed",
            "turn_count": turn_count,
            "parse_result": {"raw_question": FIRST_TURN},
        }

    def test_record_result_creates_auto_title_and_preview(self) -> None:
        thread_id = self._thread("record")
        item = self.repository.record_result(
            self._result(thread_id),
            customer_id="customer-a",
            execution_mode="LangGraph",
        )

        self.assertEqual(item["thread_id"], thread_id)
        self.assertEqual(item["customer_id"], "customer-a")
        self.assertEqual(item["title"], FIRST_TURN)
        self.assertEqual(item["last_message_preview"], FIRST_TURN)
        self.assertEqual(item["turn_count"], 1)
        self.assertFalse(item["archived"])

    def test_threads_are_recent_first_and_customer_scoped(self) -> None:
        first_thread = self._thread("first")
        second_thread = self._thread("second")
        self.repository.record_result(
            self._result(first_thread),
            customer_id="customer-a",
        )
        self.repository.record_result(
            self._result(second_thread),
            customer_id="customer-a",
            question="卡特320D液压泵多少钱？",
        )
        self.repository.record_result(
            self._result(self._thread("other")),
            customer_id="customer-b",
        )

        items = self.repository.list_threads("customer-a")
        self.assertEqual(
            [item["thread_id"] for item in items],
            [second_thread, first_thread],
        )
        self.assertEqual(self.repository.list_threads("customer-b")[0]["customer_id"], "customer-b")

    def test_rename_archive_and_restore(self) -> None:
        thread_id = self._thread("lifecycle")
        self.repository.create_thread(
            thread_id=thread_id,
            customer_id="customer-a",
        )
        renamed = self.repository.rename_thread(
            thread_id,
            customer_id="customer-a",
            title="PC200 主泵报价",
        )
        self.assertEqual(renamed["title"], "PC200 主泵报价")
        self.assertTrue(renamed["title_is_custom"])

        archived = self.repository.archive_thread(
            thread_id,
            customer_id="customer-a",
        )
        self.assertTrue(archived["archived"])
        self.assertEqual(self.repository.list_threads("customer-a"), [])
        self.assertEqual(
            self.repository.list_threads(
                "customer-a",
                include_archived=True,
            )[0]["thread_id"],
            thread_id,
        )

        restored = self.repository.restore_thread(
            thread_id,
            customer_id="customer-a",
        )
        self.assertFalse(restored["archived"])
        self.assertEqual(self.repository.list_threads("customer-a")[0]["thread_id"], thread_id)

    def test_customer_access_is_enforced_for_all_mutations(self) -> None:
        thread_id = self._thread("owned")
        self.repository.create_thread(
            thread_id=thread_id,
            customer_id="customer-a",
        )

        with self.assertRaises(ConversationAccessError):
            self.repository.get_thread(thread_id, customer_id="customer-b")
        with self.assertRaises(ConversationAccessError):
            self.repository.rename_thread(
                thread_id,
                customer_id="customer-b",
                title="越权标题",
            )
        with self.assertRaises(ConversationAccessError):
            self.repository.archive_thread(
                thread_id,
                customer_id="customer-b",
            )
        with self.assertRaises(ConversationAccessError):
            self.repository.record_result(
                self._result(thread_id, turn_count=2),
                customer_id="customer-b",
            )

    def test_old_checkpoint_loads_and_continues_after_graph_restart(self) -> None:
        thread_id = self._thread("restart")
        graph = self._new_graph()
        first = agent_graph.start_graph_agent(
            FIRST_TURN,
            thread_id=thread_id,
            customer_id="customer-a",
            approval_mode="auto",
            graph=graph,
        )
        self.repository.record_result(first, customer_id="customer-a")

        restarted_graph = self._new_graph()
        loaded = agent_graph.load_graph_thread(
            thread_id,
            customer_id="customer-a",
            graph=restarted_graph,
        )
        self.assertEqual(loaded["turn_count"], 1)
        self.assertEqual(loaded["thread_customer_id"], "customer-a")
        self.assertEqual(loaded["parse_result"]["slots"]["machine_model"], "PC200")

        continued = agent_graph.start_graph_agent(
            FOLLOW_UP,
            thread_id=thread_id,
            customer_id="customer-a",
            approval_mode="auto",
            graph=restarted_graph,
        )
        self.repository.record_result(continued, customer_id="customer-a")
        self.assertEqual(continued["turn_count"], 2)
        self.assertEqual(continued["called_tools"], ["quote_tool"])
        self.assertEqual(
            self.repository.get_thread(
                thread_id,
                customer_id="customer-a",
            )["turn_count"],
            2,
        )

    def test_checkpoint_load_rejects_missing_or_other_customer(self) -> None:
        graph = self._new_graph()
        thread_id = self._thread("protected")
        agent_graph.start_graph_agent(
            FIRST_TURN,
            thread_id=thread_id,
            customer_id="customer-a",
            approval_mode="auto",
            graph=graph,
        )

        with self.assertRaisesRegex(PermissionError, "其他客户"):
            agent_graph.load_graph_thread(
                thread_id,
                customer_id="customer-b",
                graph=graph,
            )
        with self.assertRaisesRegex(KeyError, "checkpoint 不存在"):
            agent_graph.load_graph_thread(
                self._thread("missing"),
                customer_id="customer-a",
                graph=graph,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
