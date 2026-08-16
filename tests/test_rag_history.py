from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

import rag_chat
from rag_components import SemanticAnswerCache, create_semantic_cache_vector_store
from rag_history import CONVERSATION_STATE_FIELDS, RagHistoryRepository


class FakeRetriever:
    def __init__(self, docs: list[Document] | None = None) -> None:
        self.docs = docs or []
        self.queries: list[str] = []

    def invoke(self, query: str, config=None) -> list[Document]:
        self.queries.append(query)
        return [
            Document(page_content=doc.page_content, metadata=dict(doc.metadata))
            for doc in self.docs
        ]


class FakeAnswerChain:
    def __init__(self, answer: str = "这是基于当前证据的测试回答。") -> None:
        self.answer = answer
        self.inputs: list[dict[str, object]] = []

    def invoke(self, inputs: dict[str, object], config=None) -> str:
        self.inputs.append(inputs)
        return self.answer

    def stream(self, inputs: dict[str, object], config=None):
        self.inputs.append(inputs)
        midpoint = max(1, len(self.answer) // 2)
        yield self.answer[:midpoint]
        yield self.answer[midpoint:]


class FailingStreamingChain:
    def stream(self, _inputs: dict[str, object], config=None):
        yield "未完成的半截回答"
        raise RuntimeError("fake stream failure")


class CacheEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        if any(term in text for term in ("托轮", "支重轮", "托链轮")):
            return [1.0, 0.0, 0.0]
        return [0.0, 1.0, 0.0]


class FailingRewriter:
    def invoke(self, _payload):
        raise RuntimeError("fake rewrite failure")


def _evidence_doc(distance: float = 0.2) -> Document:
    return Document(
        page_content="小松 PC200-8 液压泵可按原厂品质询价，适配前需核对零件号。",
        metadata={
            "source": "upload/doc-1/ver-1/液压泵说明.md",
            "original_name": "液压泵说明.md",
            "document_id": "doc-1",
            "version_id": "ver-1",
            "chunk_id": "chunk-sha1",
            "section": "PC200-8 液压泵",
            "retrieval_rank": 1,
            "retrieval_distance": distance,
        },
    )


def _static_faq_doc(distance: float = 0.1) -> Document:
    return Document(
        page_content="托轮承托上方链条，支重轮位于履带架下方并承载整机重量。",
        metadata={
            "source": "docs/底盘件FAQ.md",
            "chunk_id": "static-chunk-sha1",
            "section": "托轮与支重轮",
            "retrieval_rank": 1,
            "retrieval_distance": distance,
        },
    )


class RagHistoryRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "rag_state.sqlite3"
        self.repository = RagHistoryRepository(self.db_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_creates_reads_and_switches_conversations(self) -> None:
        first = self.repository.create_conversation("user-a", "询价")
        second = self.repository.create_conversation("user-a", "售后")

        conversations = self.repository.list_conversations("user-a")

        self.assertEqual(len(conversations), 2)
        self.assertEqual(
            self.repository.get_conversation("user-a", first["conversation_id"])["title"],
            "询价",
        )
        self.assertEqual(
            self.repository.get_conversation("user-a", second["conversation_id"])["title"],
            "售后",
        )

    def test_sqlite_history_survives_repository_restart(self) -> None:
        conversation = self.repository.create_conversation("user-a")
        self.repository.add_message(
            "user-a",
            conversation["conversation_id"],
            "user",
            "我要 PC200 液压泵",
        )

        restarted = RagHistoryRepository(self.db_path)
        messages = restarted.list_messages("user-a", conversation["conversation_id"])

        self.assertEqual(messages[0]["content"], "我要 PC200 液压泵")

    def test_user_cannot_read_another_users_conversation(self) -> None:
        conversation = self.repository.create_conversation("user-a")
        self.repository.add_message(
            "user-a",
            conversation["conversation_id"],
            "user",
            "仅属于用户 A",
        )

        self.assertIsNone(
            self.repository.get_conversation("user-b", conversation["conversation_id"])
        )
        with self.assertRaises(PermissionError):
            self.repository.list_messages("user-b", conversation["conversation_id"])

    def test_new_conversation_does_not_inherit_state(self) -> None:
        first = self.repository.create_conversation("user-a")
        self.repository.update_conversation_state(
            "user-a",
            first["conversation_id"],
            {"brand": "小松", "machine_model": "PC200-8"},
        )
        second = self.repository.create_conversation("user-a")

        state = self.repository.get_conversation_state(
            "user-a",
            second["conversation_id"],
        )

        self.assertEqual(state, {field: "" for field in CONVERSATION_STATE_FIELDS})

    def test_recent_history_has_a_hard_limit(self) -> None:
        conversation = self.repository.create_conversation("user-a")
        for index in range(15):
            self.repository.add_message(
                "user-a",
                conversation["conversation_id"],
                "user",
                f"消息 {index}",
            )

        recent = self.repository.get_recent_history(
            "user-a",
            conversation["conversation_id"],
            max_messages=50,
        )

        self.assertEqual(len(recent), 12)
        self.assertEqual(recent[0]["content"], "消息 3")
        self.assertEqual(recent[-1]["content"], "消息 14")

    def test_required_tables_are_created_in_shared_database(self) -> None:
        connection = sqlite3.connect(self.db_path)
        try:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = ?",
                    ("table",),
                )
            }
        finally:
            connection.close()

        self.assertTrue(
            {
                "rag_conversations",
                "rag_messages",
                "rag_message_citations",
                "rag_conversation_state",
            }.issubset(tables)
        )


class RagConversationPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = Path(self.temp_dir.name) / "rag_state.sqlite3"
        self.repository = RagHistoryRepository(self.db_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_short_follow_up_becomes_complete_standalone_query(self) -> None:
        history = [
            {"role": "user", "content": "我要小松PC200-8液压泵。"},
            {"role": "assistant", "content": "请问需要原厂还是副厂？"},
        ]

        query = rag_chat.generate_standalone_query("原厂的呢？", history)

        for expected in ("小松", "PC200-8", "液压泵", "原厂"):
            self.assertIn(expected, query)

    def test_rewriter_failure_falls_back_to_original_question(self) -> None:
        question = "原厂的呢？"
        query = rag_chat.generate_standalone_query(
            question,
            [{"role": "user", "content": "我要小松PC200-8液压泵。"}],
            query_rewriter=FailingRewriter(),
        )

        self.assertEqual(query, question)

    def test_retrieval_uses_standalone_query_not_short_follow_up(self) -> None:
        conversation = self.repository.create_conversation("user-a")
        retriever = FakeRetriever([_evidence_doc()])
        chain = FakeAnswerChain("请确认原厂件零件号。")
        rag_chat.answer_with_metadata(
            "我要小松PC200-8液压泵。",
            user_id="user-a",
            conversation_id=conversation["conversation_id"],
            history_repository=self.repository,
            retriever=retriever,
            answer_chain=chain,
            cache_enabled=False,
        )

        result = rag_chat.answer_with_metadata(
            "原厂的呢？",
            user_id="user-a",
            conversation_id=conversation["conversation_id"],
            history_repository=self.repository,
            retriever=retriever,
            answer_chain=chain,
            cache_enabled=False,
        )

        self.assertNotEqual(result["standalone_query"], "原厂的呢？")
        self.assertEqual(retriever.queries[-1], result["standalone_query"])
        for expected in ("小松", "PC200-8", "液压泵", "原厂"):
            self.assertIn(expected, retriever.queries[-1])

    def test_prompt_contains_only_limited_recent_history(self) -> None:
        history = [
            {"role": "user", "content": f"历史消息 {index}"}
            for index in range(20)
        ]
        prompt = rag_chat.build_prompt(
            "现在呢？",
            "当前证据",
            rag_chat.classify_question("现在呢？"),
            {"status": "ok", "reason": "命中"},
            chat_history=history,
        )

        self.assertNotIn("历史消息 0", prompt)
        self.assertNotIn("历史消息 7", prompt)
        self.assertIn("历史消息 8", prompt)
        self.assertIn("历史消息 19", prompt)
        self.assertIn("本轮检索证据，回答事实的最高优先级", prompt)

    def test_assistant_answer_and_state_are_persisted(self) -> None:
        conversation = self.repository.create_conversation("user-a")
        result = rag_chat.answer_with_metadata(
            "我要小松PC200-8液压泵。",
            user_id="user-a",
            conversation_id=conversation["conversation_id"],
            history_repository=self.repository,
            retriever=FakeRetriever([_evidence_doc()]),
            answer_chain=FakeAnswerChain(),
            cache_enabled=False,
        )

        messages = self.repository.list_messages(
            "user-a",
            conversation["conversation_id"],
        )
        state = self.repository.get_conversation_state(
            "user-a",
            conversation["conversation_id"],
        )

        self.assertEqual(messages[-1]["role"], "assistant")
        self.assertEqual(messages[-1]["answer_status"], "completed")
        self.assertEqual(messages[-1]["message_id"], result["assistant_message_id"])
        self.assertEqual(state["brand"], "小松")
        self.assertEqual(state["machine_model"], "PC200-8")
        self.assertEqual(state["part_name"], "液压泵")

    def test_citations_come_from_retrieval_metadata(self) -> None:
        result = rag_chat.answer_with_metadata(
            "PC200-8 液压泵怎么确认？",
            retriever=FakeRetriever([_evidence_doc()]),
            answer_chain=FakeAnswerChain(),
            cache_enabled=False,
        )

        citation = result["citations"][0]
        self.assertEqual(citation["document_name"], "液压泵说明.md")
        self.assertEqual(citation["document_id"], "doc-1")
        self.assertEqual(citation["version_id"], "ver-1")
        self.assertEqual(citation["chunk_id"], "chunk-sha1")
        self.assertEqual(citation["retrieval_rank"], 1)
        self.assertEqual(citation["retrieval_distance"], 0.2)

    def test_low_confidence_answer_has_no_citations(self) -> None:
        conversation = self.repository.create_conversation("user-a")
        result = rag_chat.answer_with_metadata(
            "一个库外问题",
            user_id="user-a",
            conversation_id=conversation["conversation_id"],
            history_repository=self.repository,
            retriever=FakeRetriever([_evidence_doc(distance=1.5)]),
            answer_chain=FakeAnswerChain(),
            cache_enabled=False,
        )

        citations = self.repository.get_message_citations(
            "user-a",
            conversation["conversation_id"],
            result["assistant_message_id"],
        )
        self.assertEqual(result["answer_source"], "fallback")
        self.assertEqual(result["citations"], [])
        self.assertEqual(citations, [])

    def test_citation_is_linked_to_assistant_message(self) -> None:
        conversation = self.repository.create_conversation("user-a")
        result = rag_chat.answer_with_metadata(
            "PC200-8 液压泵",
            user_id="user-a",
            conversation_id=conversation["conversation_id"],
            history_repository=self.repository,
            retriever=FakeRetriever([_evidence_doc()]),
            answer_chain=FakeAnswerChain(),
            cache_enabled=False,
        )

        citations = self.repository.get_message_citations(
            "user-a",
            conversation["conversation_id"],
            result["assistant_message_id"],
        )

        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0]["message_id"], result["assistant_message_id"])

    def test_empty_conversation_id_creates_a_fresh_conversation(self) -> None:
        result = rag_chat.answer_with_metadata(
            "PC200-8 液压泵",
            user_id="user-a",
            conversation_id="",
            history_repository=self.repository,
            retriever=FakeRetriever([_evidence_doc()]),
            answer_chain=FakeAnswerChain(),
            cache_enabled=False,
        )

        self.assertIsNotNone(
            self.repository.get_conversation("user-a", result["conversation_id"])
        )

    def test_old_answer_with_metadata_call_remains_compatible(self) -> None:
        retriever = FakeRetriever([_evidence_doc()])
        chain = FakeAnswerChain()
        with (
            patch("rag_chat.get_retriever", return_value=retriever),
            patch("rag_chat.get_answer_chain", return_value=chain),
            patch("rag_chat.RAG_SEMANTIC_CACHE_ENABLED", False),
        ):
            result = rag_chat.answer_with_metadata("PC200-8 液压泵", k=3)

        self.assertEqual(result["answer_source"], "llm")
        self.assertEqual(result["conversation_id"], None)
        self.assertEqual(len(result["docs"]), 1)

    def _semantic_cache(self) -> SemanticAnswerCache:
        vector_store = create_semantic_cache_vector_store(
            CacheEmbeddings(),
            persist_directory=Path(self.temp_dir.name) / "semantic_cache",
            collection_name="rag_pipeline_cache",
        )
        return SemanticAnswerCache(
            vector_store,
            threshold=0.1,
            ttl_seconds=3600,
        )

    def test_stream_returns_real_deltas_and_persists_one_complete_answer(self) -> None:
        conversation = self.repository.create_conversation("user-a")
        chain = FakeAnswerChain("托轮承托链条，支重轮承载整机重量。")

        events = list(
            rag_chat.stream_answer_with_metadata(
                "托轮和支重轮有什么区别？",
                user_id="user-a",
                conversation_id=conversation["conversation_id"],
                history_repository=self.repository,
                retriever=FakeRetriever([_static_faq_doc()]),
                answer_chain=chain,
                cache_enabled=False,
            )
        )
        deltas = [event["text"] for event in events if event["type"] == "delta"]
        final = next(event["result"] for event in events if event["type"] == "final")
        messages = self.repository.list_messages(
            "user-a",
            conversation["conversation_id"],
        )
        assistant_messages = [
            message for message in messages if message["role"] == "assistant"
        ]

        self.assertGreaterEqual(len(deltas), 2)
        self.assertEqual(final["answer"], "".join(deltas))
        self.assertTrue(final["is_streaming"])
        self.assertEqual(len(assistant_messages), 1)
        self.assertEqual(assistant_messages[0]["answer_status"], "completed")
        self.assertEqual(assistant_messages[0]["content"], final["answer"])

    def test_stream_failure_saves_failed_status_without_partial_answer_or_citation(
        self,
    ) -> None:
        conversation = self.repository.create_conversation("user-a")

        events = list(
            rag_chat.stream_answer_with_metadata(
                "托轮和支重轮有什么区别？",
                user_id="user-a",
                conversation_id=conversation["conversation_id"],
                history_repository=self.repository,
                retriever=FakeRetriever([_static_faq_doc()]),
                answer_chain=FailingStreamingChain(),
                cache_enabled=False,
            )
        )
        messages = self.repository.list_messages(
            "user-a",
            conversation["conversation_id"],
        )
        assistant = next(
            message for message in messages if message["role"] == "assistant"
        )
        citations = self.repository.get_message_citations(
            "user-a",
            conversation["conversation_id"],
            assistant["message_id"],
        )

        self.assertTrue(any(event["type"] == "error" for event in events))
        self.assertFalse(any(event["type"] == "final" for event in events))
        self.assertEqual(assistant["answer_status"], "failed")
        self.assertNotIn("未完成的半截回答", assistant["content"])
        self.assertEqual(citations, [])

    def test_same_static_faq_second_call_hits_semantic_cache(self) -> None:
        semantic_cache = self._semantic_cache()
        chain = FakeAnswerChain("托轮承托链条，支重轮承载整机重量。")
        retriever = FakeRetriever([_static_faq_doc()])

        first = rag_chat.answer_with_metadata(
            "托轮和支重轮有什么区别？",
            retriever=retriever,
            answer_chain=chain,
            semantic_cache=semantic_cache,
            knowledge_fingerprint="fingerprint-v1",
            prompt_version="prompt-v1",
            generation_model_name="fake-model",
        )
        second = rag_chat.answer_with_metadata(
            "托轮和支重轮有什么区别？",
            retriever=retriever,
            answer_chain=chain,
            semantic_cache=semantic_cache,
            knowledge_fingerprint="fingerprint-v1",
            prompt_version="prompt-v1",
            generation_model_name="fake-model",
        )

        self.assertFalse(first["cache_hit"])
        self.assertEqual(first["cache"]["write_status"], "stored")
        self.assertTrue(second["cache_hit"])
        self.assertEqual(second["answer_source"], "semantic_cache")
        self.assertEqual(second["citations"][0]["chunk_id"], "static-chunk-sha1")
        self.assertEqual(len(chain.inputs), 1)

    def test_static_synonym_can_hit_existing_semantic_cache(self) -> None:
        semantic_cache = self._semantic_cache()
        chain = FakeAnswerChain("托轮承托链条，支重轮承载整机重量。")
        retriever = FakeRetriever([_static_faq_doc()])
        rag_chat.answer_with_metadata(
            "托轮和支重轮有什么区别？",
            retriever=retriever,
            answer_chain=chain,
            semantic_cache=semantic_cache,
            knowledge_fingerprint="fingerprint-v1",
            prompt_version="prompt-v1",
            generation_model_name="fake-model",
        )

        result = rag_chat.answer_with_metadata(
            "支重轮与托链轮的区别是什么？",
            retriever=retriever,
            answer_chain=chain,
            semantic_cache=semantic_cache,
            knowledge_fingerprint="fingerprint-v1",
            prompt_version="prompt-v1",
            generation_model_name="fake-model",
        )

        self.assertTrue(result["cache_hit"])
        self.assertEqual(len(chain.inputs), 1)

    def test_price_inventory_and_history_queries_bypass_cache(self) -> None:
        price = rag_chat.evaluate_cache_eligibility(
            "PC200液压泵多少钱？",
            rag_chat.classify_question("PC200液压泵多少钱？"),
        )
        inventory = rag_chat.evaluate_cache_eligibility(
            "PC200液压泵有没有现货？",
            rag_chat.classify_question("PC200液压泵有没有现货？"),
        )
        history = rag_chat.evaluate_cache_eligibility(
            "托轮和支重轮有什么区别？",
            rag_chat.classify_question("托轮和支重轮有什么区别？"),
            recent_history=[{"role": "user", "content": "上一轮问题"}],
        )

        self.assertFalse(price["eligible"])
        self.assertFalse(inventory["eligible"])
        self.assertFalse(history["eligible"])
        self.assertEqual(history["reason"], "history_present")

    def test_low_confidence_refusal_is_not_written_to_cache(self) -> None:
        semantic_cache = self._semantic_cache()
        result = rag_chat.answer_with_metadata(
            "托轮和支重轮有什么区别？",
            retriever=FakeRetriever([_static_faq_doc(distance=1.5)]),
            answer_chain=FakeAnswerChain(),
            semantic_cache=semantic_cache,
            knowledge_fingerprint="fingerprint-v1",
            prompt_version="prompt-v1",
            generation_model_name="fake-model",
        )

        self.assertEqual(result["answer_source"], "fallback")
        self.assertEqual(semantic_cache.vector_store.get()["ids"], [])

    def test_answer_without_citation_is_not_written_to_cache(self) -> None:
        semantic_cache = self._semantic_cache()
        with patch("rag_chat.build_citations", return_value=[]):
            result = rag_chat.answer_with_metadata(
                "托轮和支重轮有什么区别？",
                retriever=FakeRetriever([_static_faq_doc()]),
                answer_chain=FakeAnswerChain(),
                semantic_cache=semantic_cache,
                knowledge_fingerprint="fingerprint-v1",
                prompt_version="prompt-v1",
                generation_model_name="fake-model",
            )

        self.assertEqual(result["cache"]["rejection_reason"], "missing_citations")
        self.assertEqual(semantic_cache.vector_store.get()["ids"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
