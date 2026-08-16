from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Protocol

from memory_repository import MemoryRepository


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _terms(text: str) -> list[str]:
    normalized = str(text or "").casefold()
    latin = re.findall(r"[a-z0-9][a-z0-9_.:/-]+", normalized)
    cjk_sequences = re.findall(r"[\u3400-\u9fff]+", normalized)
    cjk: list[str] = []
    for sequence in cjk_sequences:
        cjk.append(sequence)
        cjk.extend(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return latin + cjk


class EmbeddingProvider(Protocol):
    model: str
    dimensions: int

    def embed(self, texts: Iterable[str]) -> list[list[float]]: ...


class HashingEmbeddingProvider:
    """Offline deterministic baseline; replace with BGE/OpenAI-compatible embeddings in production."""

    model = "feature-hashing-memory-v1"

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions < 32:
            raise ValueError("embedding dimensions must be at least 32")
        self.dimensions = dimensions

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for term in set(_terms(text)):
            digest = hashlib.sha256(term.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0 if digest[4] % 2 == 0 else -1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions do not match")
    return sum(a * b for a, b in zip(left, right, strict=True))


@dataclass(frozen=True, slots=True)
class SummaryResult:
    episode_summary: str
    stable_facts: tuple[dict[str, Any], ...] = ()
    events: tuple[dict[str, Any], ...] = ()
    decisions: tuple[dict[str, Any], ...] = ()
    open_questions: tuple[str, ...] = ()
    conflicts: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode_summary": self.episode_summary,
            "stable_facts": list(self.stable_facts),
            "events": list(self.events),
            "decisions": list(self.decisions),
            "open_questions": list(self.open_questions),
            "conflicts": list(self.conflicts),
        }


class SemanticSummarizer(Protocol):
    def summarize(self, messages: list[dict[str, Any]]) -> SummaryResult: ...


class RuleBasedSemanticSummarizer:
    """Safe offline fallback used by tests and local demos.

    A live model can be injected through ``summarizer``. The output contract is
    intentionally identical so model output never bypasses memory governance.
    """

    _fact_patterns = {
        "brand": re.compile(r"(?:品牌|我是|使用)([\u3400-\u9fffA-Za-z0-9-]{2,20})"),
        "machine_model": re.compile(r"\b([A-Za-z]{1,8}\d{2,5})\b"),
        "city": re.compile(r"(?:在|位于|送到|城市是)([\u3400-\u9fff]{2,12})"),
    }

    def summarize(self, messages: list[dict[str, Any]]) -> SummaryResult:
        user_text = "\n".join(
            str(item.get("content", ""))
            for item in messages
            if item.get("role") == "user"
        ).strip()
        compact = " ".join(
            f"{item.get('role', 'unknown')}: {str(item.get('content', '')).strip()}"
            for item in messages
            if str(item.get("content", "")).strip()
        )
        facts: list[dict[str, Any]] = []
        for key, pattern in self._fact_patterns.items():
            match = pattern.search(user_text)
            if match:
                facts.append({"fact_type": key, "fact_value": match.group(1), "confidence": 0.7})
        return SummaryResult(
            episode_summary=compact[:1600],
            stable_facts=tuple(facts),
            open_questions=tuple(
                str(item.get("content", ""))[:240]
                for item in messages[-2:]
                if item.get("role") == "user" and "?" in str(item.get("content", ""))
            ),
        )


class CallableSemanticSummarizer:
    """Adapter for an LLM JSON function returning the documented summary shape."""

    def __init__(self, callback: Callable[[list[dict[str, Any]]], dict[str, Any]]) -> None:
        self.callback = callback

    def summarize(self, messages: list[dict[str, Any]]) -> SummaryResult:
        raw = self.callback(messages)
        return SummaryResult(
            episode_summary=str(raw.get("episode_summary", "")).strip()[:4000],
            stable_facts=tuple(raw.get("stable_facts") or ()),
            events=tuple(raw.get("events") or ()),
            decisions=tuple(raw.get("decisions") or ()),
            open_questions=tuple(str(item) for item in raw.get("open_questions") or ()),
            conflicts=tuple(raw.get("conflicts") or ()),
        )


class SemanticMemoryStore:
    """Raw messages, episodic memory, candidates, conflicts and durable jobs."""

    def __init__(
        self,
        path: Path,
        *,
        embedder: EmbeddingProvider | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or HashingEmbeddingProvider()
        self._setup()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(str(self.path), timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _setup(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    message_id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    turn_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    archived_at TEXT NOT NULL DEFAULT ''
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_conversation_message_dedupe
                ON conversation_messages(thread_id, request_id, role);
                CREATE INDEX IF NOT EXISTS idx_conversation_messages_scope
                ON conversation_messages(customer_id, thread_id, turn_id, created_at);

                CREATE TABLE IF NOT EXISTS episodic_memory_chunks (
                    episode_id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    turn_start INTEGER NOT NULL,
                    turn_end INTEGER NOT NULL,
                    semantic_summary TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    importance REAL NOT NULL,
                    source_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active'
                );
                CREATE INDEX IF NOT EXISTS idx_episode_scope
                ON episodic_memory_chunks(customer_id, status, expires_at, created_at);

                CREATE TABLE IF NOT EXISTS memory_candidates (
                    candidate_id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    fact_type TEXT NOT NULL,
                    fact_value TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source_episode_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memory_candidates_scope
                ON memory_candidates(customer_id, status, created_at);

                CREATE TABLE IF NOT EXISTS memory_conflicts (
                    conflict_id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL,
                    fact_type TEXT NOT NULL,
                    existing_value TEXT NOT NULL,
                    candidate_value TEXT NOT NULL,
                    source_episode_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open',
                    created_at TEXT NOT NULL,
                    resolved_at TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_memory_conflicts_scope
                ON memory_conflicts(customer_id, status, created_at);

                CREATE TABLE IF NOT EXISTS memory_jobs (
                    job_id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    upto_turn INTEGER NOT NULL,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL DEFAULT 'pending',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    lease_owner TEXT NOT NULL DEFAULT '',
                    lease_expires_at TEXT NOT NULL DEFAULT '',
                    last_error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memory_jobs_claim
                ON memory_jobs(status, lease_expires_at, created_at);
                """
            )

    def append_message(
        self,
        *,
        customer_id: str,
        thread_id: str,
        turn_id: int,
        role: str,
        content: str,
        request_id: str = "",
    ) -> str:
        normalized = str(content or "").strip()
        if not customer_id or not thread_id or not normalized:
            raise ValueError("customer_id, thread_id and content are required")
        message_id = hashlib.sha256(
            f"{thread_id}:{request_id}:{role}:{normalized}".encode("utf-8")
        ).hexdigest()[:32]
        now = _now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO conversation_messages(
                    message_id, customer_id, thread_id, turn_id, role, content,
                    content_hash, request_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    customer_id,
                    thread_id,
                    max(1, int(turn_id)),
                    role,
                    normalized,
                    hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
                    request_id or message_id,
                    now,
                ),
            )
        return message_id

    def messages_for_thread(self, customer_id: str, thread_id: str, upto_turn: int | None = None) -> list[dict[str, Any]]:
        query = """
            SELECT * FROM conversation_messages
            WHERE customer_id=? AND thread_id=? AND archived_at=''
        """
        params: list[Any] = [customer_id, thread_id]
        if upto_turn is not None:
            query += " AND turn_id<=?"
            params.append(int(upto_turn))
        query += " ORDER BY turn_id, created_at, message_id"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def enqueue(self, *, customer_id: str, thread_id: str, upto_turn: int, reason: str = "periodic") -> tuple[str, bool]:
        key = f"memory-consolidation:{customer_id}:{thread_id}:{int(upto_turn)}"
        now = _now()
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT job_id FROM memory_jobs WHERE idempotency_key=?", (key,)
            ).fetchone()
            if existing:
                return str(existing["job_id"]), True
            job_id = str(uuid.uuid4())
            connection.execute(
                """
                INSERT INTO memory_jobs(
                    job_id, customer_id, thread_id, upto_turn, idempotency_key,
                    status, created_at, updated_at, last_error
                ) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?)
                """,
                (job_id, customer_id, thread_id, int(upto_turn), key, now, now, reason),
            )
        return job_id, False

    def claim(self, *, worker_id: str, lease_seconds: int = 120) -> dict[str, Any] | None:
        now = datetime.now(timezone.utc)
        now_text = now.isoformat(timespec="seconds")
        lease_text = (now + timedelta(seconds=lease_seconds)).isoformat(timespec="seconds")
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM memory_jobs
                WHERE (status='pending' OR (status='running' AND lease_expires_at<=?))
                ORDER BY created_at, job_id LIMIT 1
                """,
                (now_text,),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """
                UPDATE memory_jobs
                SET status='running', attempts=attempts+1, lease_owner=?,
                    lease_expires_at=?, updated_at=?
                WHERE job_id=?
                """,
                (worker_id, lease_text, now_text, row["job_id"]),
            )
            claimed = connection.execute(
                "SELECT * FROM memory_jobs WHERE job_id=?", (row["job_id"],)
            ).fetchone()
        return dict(claimed) if claimed else None

    def finish(self, job_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE memory_jobs SET status='completed', lease_owner='', lease_expires_at='', updated_at=? WHERE job_id=?",
                (_now(), job_id),
            )

    def fail(self, job_id: str, error: Exception) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE memory_jobs SET status='failed', lease_owner='', lease_expires_at='', last_error=?, updated_at=? WHERE job_id=?",
                (str(error)[:1000], _now(), job_id),
            )

    def add_episode(
        self,
        *,
        customer_id: str,
        thread_id: str,
        turn_start: int,
        turn_end: int,
        summary: SummaryResult,
        importance: float = 0.5,
        ttl_days: int = 180,
    ) -> str:
        text = summary.episode_summary.strip()
        if not text:
            raise ValueError("episode summary cannot be empty")
        source_hash = hashlib.sha256(
            f"{customer_id}:{thread_id}:{turn_start}:{turn_end}:{text}".encode("utf-8")
        ).hexdigest()
        episode_id = f"EPI-{source_hash[:20].upper()}"
        vector = self.embedder.embed([text])[0]
        now = datetime.now(timezone.utc)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO episodic_memory_chunks(
                    episode_id, customer_id, thread_id, turn_start, turn_end,
                    semantic_summary, summary_json, embedding_json, embedding_model,
                    importance, source_hash, created_at, expires_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')
                """,
                (
                    episode_id,
                    customer_id,
                    thread_id,
                    turn_start,
                    turn_end,
                    text,
                    json.dumps(summary.as_dict(), ensure_ascii=False, sort_keys=True),
                    json.dumps(vector, separators=(",", ":")),
                    self.embedder.model,
                    min(1.0, max(0.0, float(importance))),
                    source_hash,
                    now.isoformat(timespec="seconds"),
                    (now + timedelta(days=max(1, ttl_days))).isoformat(timespec="seconds"),
                ),
            )
        return episode_id

    def search(self, *, customer_id: str, query: str, limit: int = 5) -> list[dict[str, Any]]:
        self.expire_episodes()
        query_vector = self.embedder.embed([query])[0]
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM episodic_memory_chunks WHERE customer_id=? AND status='active'",
                (customer_id,),
            ).fetchall()
        scored = []
        for row in rows:
            similarity = max(0.0, _cosine(query_vector, json.loads(row["embedding_json"])))
            scored.append(
                (
                    similarity * 0.8 + float(row["importance"]) * 0.2,
                    {**dict(row), "similarity": round(similarity, 6)},
                )
            )
        scored.sort(key=lambda item: (-item[0], item[1]["episode_id"]))
        return [item[1] for item in scored[: max(1, int(limit))] if item[0] > 0]

    def propose_facts(self, *, customer_id: str, thread_id: str, episode_id: str, facts: Iterable[dict[str, Any]]) -> int:
        count = 0
        now = _now()
        repository = MemoryRepository(self.path)
        with self._connect() as connection:
            for fact in facts:
                fact_type = str(fact.get("fact_type", "")).strip()
                value = str(fact.get("fact_value", "")).strip()
                confidence = min(1.0, max(0.0, float(fact.get("confidence", 0.0))))
                if not fact_type or not value or confidence < 0.65:
                    continue
                allowed, _ = repository.policy.validate(fact_type, value)
                if not allowed:
                    continue
                existing = repository.get_fact(customer_id, fact_type)
                if existing and str(existing.get("fact_value")) != value:
                    connection.execute(
                        """
                        INSERT INTO memory_conflicts(
                            conflict_id, customer_id, fact_type, existing_value,
                            candidate_value, source_episode_id, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            str(uuid.uuid4()),
                            customer_id,
                            fact_type,
                            str(existing["fact_value"]),
                            value,
                            episode_id,
                            now,
                        ),
                    )
                candidate_id = hashlib.sha256(
                    f"{customer_id}:{fact_type}:{value}:{episode_id}".encode("utf-8")
                ).hexdigest()[:32]
                connection.execute(
                    """
                    INSERT OR IGNORE INTO memory_candidates(
                        candidate_id, customer_id, thread_id, fact_type,
                        fact_value, confidence, source_episode_id, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (candidate_id, customer_id, thread_id, fact_type, value, confidence, episode_id, now),
                )
                count += 1
        return count

    def expire_episodes(self) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE episodic_memory_chunks SET status='expired' WHERE status='active' AND expires_at<=?",
                (_now(),),
            )
        return cursor.rowcount

    def candidates(self, customer_id: str, *, status: str = "pending") -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM memory_candidates WHERE customer_id=? AND status=? ORDER BY created_at",
                (customer_id, status),
            ).fetchall()
        return [dict(row) for row in rows]

    def conflicts(self, customer_id: str, *, status: str = "open") -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM memory_conflicts WHERE customer_id=? AND status=? ORDER BY created_at",
                (customer_id, status),
            ).fetchall()
        return [dict(row) for row in rows]


class MemoryConsolidator:
    def __init__(
        self,
        store: SemanticMemoryStore,
        *,
        summarizer: SemanticSummarizer | None = None,
    ) -> None:
        self.store = store
        self.summarizer = summarizer or RuleBasedSemanticSummarizer()

    def consolidate(self, job: dict[str, Any]) -> dict[str, Any]:
        messages = self.store.messages_for_thread(
            str(job["customer_id"]), str(job["thread_id"]), int(job["upto_turn"])
        )
        if not messages:
            return {"job_id": job["job_id"], "status": "empty"}
        summary = self.summarizer.summarize(messages)
        turn_start = min(int(item["turn_id"]) for item in messages)
        turn_end = max(int(item["turn_id"]) for item in messages)
        episode_id = self.store.add_episode(
            customer_id=str(job["customer_id"]),
            thread_id=str(job["thread_id"]),
            turn_start=turn_start,
            turn_end=turn_end,
            summary=summary,
            importance=0.8 if summary.decisions or summary.conflicts else 0.5,
        )
        candidate_count = self.store.propose_facts(
            customer_id=str(job["customer_id"]),
            thread_id=str(job["thread_id"]),
            episode_id=episode_id,
            facts=summary.stable_facts,
        )
        return {
            "job_id": job["job_id"],
            "status": "consolidated",
            "episode_id": episode_id,
            "message_count": len(messages),
            "candidate_count": candidate_count,
            "conflict_count": len(self.store.conflicts(str(job["customer_id"]))),
        }


class MemoryWorker:
    def __init__(self, store: SemanticMemoryStore, *, worker_id: str = "memory-worker") -> None:
        self.store = store
        self.worker_id = worker_id

    def run_once(self, consolidator: MemoryConsolidator | None = None) -> dict[str, Any] | None:
        job = self.store.claim(worker_id=self.worker_id)
        if job is None:
            return None
        active = consolidator or MemoryConsolidator(self.store)
        try:
            result = active.consolidate(job)
        except Exception as exc:
            self.store.fail(str(job["job_id"]), exc)
            raise
        self.store.finish(str(job["job_id"]))
        return result
