from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda

from rag_chat import RAG_PROMPT, create_answer_chain
from rag_components import (
    SemanticAnswerCache,
    build_index_fingerprint,
    create_retriever,
    create_semantic_cache_vector_store,
    fingerprint_changes,
)


class FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, _text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]


class FakeSemanticEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        if any(term in text for term in ("托轮", "支重轮", "托链轮")):
            return [1.0, 0.0, 0.0, 0.0]
        if "液压泵" in text:
            return [0.0, 1.0, 0.0, 0.0]
        return [0.0, 0.0, 1.0, 0.0]


class FakeVectorStore:
    def similarity_search_with_score(self, query: str, k: int):
        return [
            (
                Document(
                    page_content=f"{query} 的知识片段",
                    metadata={"source": "docs/test.md"},
                ),
                0.25,
            )
        ][:k]


class RagComponentTests(unittest.TestCase):
    def test_scored_retriever_preserves_rank_distance_and_provider(self) -> None:
        retriever = create_retriever(FakeVectorStore(), k=3)
        docs = retriever.invoke("PC200 液压泵")

        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata["retrieval_rank"], 1)
        self.assertEqual(docs[0].metadata["retrieval_distance"], 0.25)
        self.assertEqual(docs[0].metadata["retrieval_provider"], "chroma")

    def test_index_fingerprint_records_vector_compatibility(self) -> None:
        fingerprint = build_index_fingerprint(
            FakeEmbeddings(),
            chunk_size=500,
            chunk_overlap=80,
        )

        self.assertEqual(fingerprint["embedding_dimension"], 4)
        self.assertEqual(fingerprint["chunk_size"], 500)
        self.assertEqual(fingerprint["chunk_overlap"], 80)
        self.assertIn("embedding_model", fingerprint)
        self.assertIn("vector_db_provider", fingerprint)

    def test_fingerprint_changes_detects_embedding_or_chunk_change(self) -> None:
        previous = {
            "embedding_model": "old-model",
            "chunk_size": 500,
        }
        current = {
            "embedding_model": "new-model",
            "chunk_size": 600,
        }
        changes = fingerprint_changes(previous, current)

        self.assertEqual(changes["embedding_model"]["previous"], "old-model")
        self.assertEqual(changes["embedding_model"]["current"], "new-model")
        self.assertIn("chunk_size", changes)

    def test_rag_generation_is_a_langchain_runnable(self) -> None:
        chain = create_answer_chain(RunnableLambda(lambda _prompt: "可验证回答"))
        result = chain.invoke(
            {
                "question_label": "配件匹配",
                "question_type": "part_match",
                "classification_reason": "命中适配关键词",
                "answer_style": "先核对件号",
                "missing_text": "旧件号",
                "retrieval_status": "ok",
                "retrieval_reason": "命中",
                "retrieval_guardrail": "只能基于证据回答",
                "follow_up_text": "- 请提供铭牌",
                "context": "PC200 不同年份可能使用不同液压泵。",
                "question": "PC200 液压泵通用吗？",
            }
        )

        self.assertEqual(result, "可验证回答")
        self.assertEqual(len(RAG_PROMPT.messages), 2)


class SemanticAnswerCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.vector_store = create_semantic_cache_vector_store(
            FakeSemanticEmbeddings(),
            persist_directory=Path(self.temp_dir.name) / "cache",
            collection_name="rag_cache_tests",
        )
        self.cache = SemanticAnswerCache(
            self.vector_store,
            threshold=0.1,
            ttl_seconds=60,
        )
        self.now = datetime(2026, 7, 29, tzinfo=timezone.utc)
        self.citations = [
            {
                "document_name": "底盘件FAQ.md",
                "source": "docs/底盘件FAQ.md",
                "chunk_id": "chunk-1",
                "retrieval_rank": 1,
                "retrieval_distance": 0.2,
            }
        ]
        self.cache.put(
            original_query="托轮和支重轮有什么区别？",
            standalone_query="托轮和支重轮有什么区别？",
            answer="托轮承托上方链条，支重轮承载整机重量。",
            citations=self.citations,
            category="general",
            knowledge_base_fingerprint="fingerprint-v1",
            prompt_version="prompt-v1",
            model_name="fake-model",
            now=self.now,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _lookup(self, query: str, **overrides):
        arguments = {
            "knowledge_base_fingerprint": "fingerprint-v1",
            "prompt_version": "prompt-v1",
            "model_name": "fake-model",
            "now": self.now,
        }
        arguments.update(overrides)
        return self.cache.lookup(query, **arguments)

    def test_identical_static_faq_hits(self) -> None:
        result = self._lookup("托轮和支重轮有什么区别？")

        self.assertTrue(result["hit"])
        self.assertEqual(result["distance"], 0.0)

    def test_explicit_static_synonym_hits(self) -> None:
        result = self._lookup("支重轮与托链轮的区别是什么？")

        self.assertTrue(result["hit"])

    def test_unrelated_question_does_not_hit(self) -> None:
        result = self._lookup("液压泵异响怎么排查？")

        self.assertFalse(result["hit"])
        self.assertEqual(result["reason"], "semantic_distance_too_high")

    def test_knowledge_fingerprint_change_invalidates_entry(self) -> None:
        result = self._lookup(
            "托轮和支重轮有什么区别？",
            knowledge_base_fingerprint="fingerprint-v2",
        )

        self.assertFalse(result["hit"])
        self.assertEqual(result["reason"], "knowledge_base_fingerprint_changed")

    def test_prompt_version_change_invalidates_entry(self) -> None:
        result = self._lookup(
            "托轮和支重轮有什么区别？",
            prompt_version="prompt-v2",
        )

        self.assertFalse(result["hit"])
        self.assertEqual(result["reason"], "prompt_version_changed")

    def test_ttl_expiry_invalidates_entry(self) -> None:
        result = self._lookup(
            "托轮和支重轮有什么区别？",
            now=self.now + timedelta(seconds=61),
        )

        self.assertFalse(result["hit"])
        self.assertEqual(result["reason"], "cache_expired")

    def test_cache_hit_returns_original_structured_citations(self) -> None:
        result = self._lookup("托轮和支重轮有什么区别？")

        self.assertEqual(result["citations"], self.citations)
        self.assertEqual(
            result["answer"],
            "托轮承托上方链条，支重轮承载整机重量。",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
