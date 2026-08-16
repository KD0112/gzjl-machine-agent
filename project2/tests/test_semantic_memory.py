from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from memory_repository import MemoryRepository
from semantic_memory import (
    MemoryConsolidator,
    MemoryWorker,
    SemanticMemoryStore,
    SummaryResult,
)


class FixedSummarizer:
    def summarize(self, messages):
        return SummaryResult(
            episode_summary="第 1 轮客户确认 PC200 原厂液压泵，后续需要确认交付时间。",
            stable_facts=(
                {"fact_type": "machine_model", "fact_value": "PC200", "confidence": 0.95},
            ),
            open_questions=("交付时间",),
        )


class SemanticMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "memory.sqlite3"
        self.repository = MemoryRepository(self.path)
        self.store = SemanticMemoryStore(self.path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_three_layers_and_100_turn_recall(self) -> None:
        for turn in range(1, 101):
            self.store.append_message(
                customer_id="customer-a",
                thread_id="thread-a",
                turn_id=turn,
                role="user",
                content=(
                    "第 1 轮确认 PC200 原厂液压泵，后续需要确认交付时间。"
                    if turn == 1
                    else f"第 {turn} 轮普通跟进问题"
                ),
                request_id=f"u-{turn}",
            )
            self.store.append_message(
                customer_id="customer-a",
                thread_id="thread-a",
                turn_id=turn,
                role="assistant",
                content=f"已处理第 {turn} 轮",
                request_id=f"a-{turn}",
            )
        job_id, duplicate = self.store.enqueue(
            customer_id="customer-a", thread_id="thread-a", upto_turn=100
        )
        self.assertFalse(duplicate)
        self.assertEqual(
            self.store.enqueue(
                customer_id="customer-a", thread_id="thread-a", upto_turn=100
            ),
            (job_id, True),
        )
        result = MemoryWorker(self.store).run_once(
            MemoryConsolidator(self.store, summarizer=FixedSummarizer())
        )
        self.assertEqual(result["status"], "consolidated")
        recalled = self.store.search(
            customer_id="customer-a", query="以前讨论的 PC200 原厂液压泵交付", limit=3
        )
        self.assertTrue(recalled)
        self.assertIn("PC200", recalled[0]["semantic_summary"])
        self.assertLess(len(self.store.messages_for_thread("customer-a", "thread-a")), 250)

    def test_candidates_are_not_auto_promoted_and_conflicts_are_recorded(self) -> None:
        self.repository.upsert_fact(
            customer_id="customer-a",
            fact_type="machine_model",
            fact_value="PC300",
            source="confirmed_customer_input",
            confidence=1.0,
        )
        self.store.append_message(
            customer_id="customer-a",
            thread_id="thread-a",
            turn_id=1,
            role="user",
            content="现在改成 PC200",
            request_id="u-1",
        )
        job_id, _ = self.store.enqueue(
            customer_id="customer-a", thread_id="thread-a", upto_turn=1
        )
        result = MemoryWorker(self.store).run_once(
            MemoryConsolidator(self.store, summarizer=FixedSummarizer())
        )
        self.assertEqual(result["job_id"], job_id)
        self.assertEqual(self.repository.get_fact("customer-a", "machine_model")["fact_value"], "PC300")
        self.assertTrue(self.store.candidates("customer-a"))
        self.assertTrue(self.store.conflicts("customer-a"))

    def test_customer_isolation(self) -> None:
        summary = SummaryResult(episode_summary="客户 A 的 PC200 交付讨论")
        self.store.add_episode(
            customer_id="customer-a",
            thread_id="thread-a",
            turn_start=1,
            turn_end=1,
            summary=summary,
        )
        self.assertEqual(self.store.search(customer_id="customer-b", query="PC200"), [])


if __name__ == "__main__":
    unittest.main()
