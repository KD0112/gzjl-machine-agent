from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from settings import (
    CHROMA_DB_DIR,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL,
    EMBEDDING_NORMALIZE,
    EMBEDDING_PROVIDER,
    RAG_SEMANTIC_CACHE_COLLECTION,
    RAG_SEMANTIC_CACHE_DIR,
    RAG_SEMANTIC_CACHE_THRESHOLD,
    RAG_SEMANTIC_CACHE_TTL_SECONDS,
    VECTOR_DB_COLLECTION,
    VECTOR_DB_DISTANCE_METRIC,
    VECTOR_DB_PROVIDER,
)


SUPPORTED_EMBEDDING_PROVIDERS = {"huggingface"}
SUPPORTED_VECTOR_DB_PROVIDERS = {"chroma"}


@lru_cache(maxsize=1)
def create_embeddings() -> Embeddings:
    """Create the configured embedding adapter from one shared factory."""
    if EMBEDDING_PROVIDER not in SUPPORTED_EMBEDDING_PROVIDERS:
        raise ValueError(
            f"Unsupported embedding provider: {EMBEDDING_PROVIDER}. "
            f"Supported providers: {sorted(SUPPORTED_EMBEDDING_PROVIDERS)}"
        )
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": EMBEDDING_NORMALIZE},
    )


def create_vector_store(
    embedding: Embeddings | None = None,
    *,
    persist_directory: Path | str = CHROMA_DB_DIR,
    collection_name: str = VECTOR_DB_COLLECTION,
) -> VectorStore:
    """Create the configured vector-store adapter behind one stable interface."""
    if VECTOR_DB_PROVIDER not in SUPPORTED_VECTOR_DB_PROVIDERS:
        raise ValueError(
            f"Unsupported vector DB provider: {VECTOR_DB_PROVIDER}. "
            f"Supported providers: {sorted(SUPPORTED_VECTOR_DB_PROVIDERS)}"
        )

    from langchain_chroma import Chroma

    collection_metadata = None
    if VECTOR_DB_DISTANCE_METRIC != "l2":
        collection_metadata = {"hnsw:space": VECTOR_DB_DISTANCE_METRIC}
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding or create_embeddings(),
        persist_directory=str(persist_directory),
        collection_metadata=collection_metadata,
    )


def knowledge_base_fingerprint(manifest_path: Path | str | None = None) -> str:
    path = Path(manifest_path or CHROMA_DB_DIR / "index_manifest.json")
    if not path.exists():
        return ""
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    payload = {
        "chunk_ids": manifest.get("chunk_ids", []),
        "index_fingerprint": manifest.get("index_fingerprint", {}),
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def create_semantic_cache_vector_store(
    embedding: Embeddings | None = None,
    *,
    persist_directory: Path | str = RAG_SEMANTIC_CACHE_DIR,
    collection_name: str = RAG_SEMANTIC_CACHE_COLLECTION,
) -> VectorStore:
    return create_vector_store(
        embedding,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )


class SemanticAnswerCache:
    def __init__(
        self,
        vector_store: VectorStore,
        *,
        threshold: float = RAG_SEMANTIC_CACHE_THRESHOLD,
        ttl_seconds: int = RAG_SEMANTIC_CACHE_TTL_SECONDS,
    ) -> None:
        self.vector_store = vector_store
        self.threshold = max(0.0, float(threshold))
        self.ttl_seconds = max(1, int(ttl_seconds))

    @staticmethod
    def _utc_now(now: datetime | None = None) -> datetime:
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        return current.astimezone(timezone.utc)

    def lookup(
        self,
        query: str,
        *,
        knowledge_base_fingerprint: str,
        prompt_version: str,
        model_name: str,
        citation_validator: Callable[[list[dict[str, Any]]], bool] | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        matches = self.vector_store.similarity_search_with_score(query, k=1)
        if not matches:
            return {"hit": False, "reason": "cache_empty"}

        document, raw_distance = matches[0]
        distance = float(raw_distance)
        metadata = dict(document.metadata or {})
        diagnostics = {
            "hit": False,
            "distance": round(distance, 4),
            "entry_created_at": metadata.get("created_at", ""),
            "expires_at": metadata.get("expires_at", ""),
            "knowledge_base_fingerprint": metadata.get(
                "knowledge_base_fingerprint",
                "",
            ),
        }
        if distance > self.threshold:
            return {**diagnostics, "reason": "semantic_distance_too_high"}
        if metadata.get("knowledge_base_fingerprint") != knowledge_base_fingerprint:
            return {**diagnostics, "reason": "knowledge_base_fingerprint_changed"}
        if metadata.get("prompt_version") != prompt_version:
            return {**diagnostics, "reason": "prompt_version_changed"}
        if metadata.get("model_name") != model_name:
            return {**diagnostics, "reason": "model_name_changed"}

        expires_at_raw = str(metadata.get("expires_at") or "")
        try:
            expires_at = datetime.fromisoformat(expires_at_raw)
        except ValueError:
            return {**diagnostics, "reason": "invalid_expiry"}
        if self._utc_now(now) >= self._utc_now(expires_at):
            return {**diagnostics, "reason": "cache_expired"}

        try:
            citations = json.loads(str(metadata.get("citations_json") or "[]"))
        except json.JSONDecodeError:
            return {**diagnostics, "reason": "invalid_citations"}
        if not citations:
            return {**diagnostics, "reason": "missing_citations"}
        if citation_validator is not None and not citation_validator(citations):
            return {**diagnostics, "reason": "citation_version_inactive"}

        return {
            **diagnostics,
            "hit": True,
            "reason": "cache_hit",
            "answer": metadata.get("answer", ""),
            "citations": citations,
            "original_query": metadata.get("original_query", ""),
            "standalone_query": metadata.get("standalone_query", ""),
            "category": metadata.get("category", ""),
            "prompt_version": metadata.get("prompt_version", ""),
            "model_name": metadata.get("model_name", ""),
        }

    def put(
        self,
        *,
        original_query: str,
        standalone_query: str,
        answer: str,
        citations: list[dict[str, Any]],
        category: str,
        knowledge_base_fingerprint: str,
        prompt_version: str,
        model_name: str,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        created_at = self._utc_now(now)
        expires_at = created_at + timedelta(seconds=self.ttl_seconds)
        identity = "|".join(
            (
                standalone_query,
                knowledge_base_fingerprint,
                prompt_version,
                model_name,
            )
        )
        cache_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        metadata = {
            "original_query": original_query,
            "standalone_query": standalone_query,
            "answer": answer,
            "citations_json": json.dumps(citations, ensure_ascii=False),
            "category": category,
            "knowledge_base_fingerprint": knowledge_base_fingerprint,
            "prompt_version": prompt_version,
            "model_name": model_name,
            "created_at": created_at.isoformat(timespec="seconds"),
            "expires_at": expires_at.isoformat(timespec="seconds"),
        }
        self.vector_store.add_documents(
            [Document(page_content=standalone_query, metadata=metadata)],
            ids=[cache_id],
        )
        return {
            "cache_id": cache_id,
            "created_at": metadata["created_at"],
            "expires_at": metadata["expires_at"],
        }

    def clear(self, knowledge_base_fingerprint: str | None = None) -> int:
        if knowledge_base_fingerprint:
            payload = self.vector_store.get(
                where={"knowledge_base_fingerprint": knowledge_base_fingerprint}
            )
        else:
            payload = self.vector_store.get()
        ids = list(payload.get("ids") or [])
        if ids:
            self.vector_store.delete(ids=ids)
        return len(ids)


class ScoredVectorStoreRetriever(BaseRetriever):
    """Retriever that preserves rank and raw vector-store distance in metadata."""

    vector_store: Any
    k: int = 3
    provider: str = VECTOR_DB_PROVIDER

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Any,
    ) -> list[Document]:
        docs_with_scores = self.vector_store.similarity_search_with_score(query, k=self.k)
        docs: list[Document] = []
        for rank, (doc, raw_score) in enumerate(docs_with_scores, start=1):
            metadata = dict(doc.metadata or {})
            metadata.update(
                {
                    "retrieval_rank": rank,
                    "retrieval_distance": round(float(raw_score), 4),
                    "retrieval_provider": self.provider,
                }
            )
            doc.metadata = metadata
            docs.append(doc)
        return docs


def create_retriever(
    vector_store: VectorStore,
    *,
    k: int = 3,
) -> ScoredVectorStoreRetriever:
    if k <= 0:
        raise ValueError("k must be greater than 0")
    return ScoredVectorStoreRetriever(vector_store=vector_store, k=k)


def build_index_fingerprint(
    embeddings: Embeddings,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> dict[str, Any]:
    """Describe every setting that affects persisted vector compatibility."""
    embedding_dimension = len(embeddings.embed_query("索引维度探针"))
    return {
        "embedding_provider": EMBEDDING_PROVIDER,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": embedding_dimension,
        "embedding_device": EMBEDDING_DEVICE,
        "embedding_normalize": EMBEDDING_NORMALIZE,
        "vector_db_provider": VECTOR_DB_PROVIDER,
        "vector_db_collection": VECTOR_DB_COLLECTION,
        "distance_metric": VECTOR_DB_DISTANCE_METRIC,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }


def fingerprint_changes(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if not previous:
        return {"index_fingerprint": {"previous": None, "current": current}}

    keys = sorted(set(previous) | set(current))
    return {
        key: {"previous": previous.get(key), "current": current.get(key)}
        for key in keys
        if previous.get(key) != current.get(key)
    }
