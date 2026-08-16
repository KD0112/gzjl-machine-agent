from __future__ import annotations

import hashlib
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MEMORY_DB_PATH = BASE_DIR / "logs" / "agent_memory.sqlite3"
ALLOWED_PROFILE_FACTS = {
    "brand",
    "machine_model",
    "quality_level",
    "city",
}
DEFAULT_TTL_DAYS = {
    "brand": 180,
    "machine_model": 180,
    "quality_level": 90,
    "city": 90,
}
SENSITIVE_FACT_TYPES = {
    "password",
    "api_key",
    "access_token",
    "id_card",
    "bank_card",
    "payment_password",
}
SENSITIVE_VALUE_PATTERNS = (
    re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"),
    re.compile(r"(?<!\d)\d{16,19}(?!\d)"),
    re.compile(r"\b(?:sk|api|token|key)[-_][A-Za-z0-9_-]{12,}\b", re.IGNORECASE),
    re.compile(r"(密码|口令|password)\s*[:：=]\s*\S+", re.IGNORECASE),
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: datetime | None = None) -> str:
    return (value or _utc_now()).isoformat(timespec="seconds")


def _default_path() -> Path:
    configured = os.getenv("AGENT_MEMORY_DB", "").strip()
    if not configured:
        return DEFAULT_MEMORY_DB_PATH
    path = Path(configured)
    return path if path.is_absolute() else BASE_DIR / path


def _memory_id(customer_id: str, fact_type: str) -> str:
    payload = f"{customer_id}:{fact_type}".encode("utf-8")
    return f"MEM-{hashlib.sha256(payload).hexdigest()[:16].upper()}"


def contains_sensitive_value(value: Any) -> bool:
    text = str(value)
    return any(pattern.search(text) for pattern in SENSITIVE_VALUE_PATTERNS)


class MemoryPolicy:
    """Deterministic allowlist policy for durable customer-profile memory."""

    def validate(self, fact_type: str, fact_value: Any) -> tuple[bool, str]:
        normalized_type = str(fact_type).strip()
        if normalized_type in SENSITIVE_FACT_TYPES:
            return False, "sensitive_fact_type"
        if normalized_type not in ALLOWED_PROFILE_FACTS:
            return False, "fact_type_not_allowlisted"
        if fact_value is None or not str(fact_value).strip():
            return False, "empty_fact_value"
        if contains_sensitive_value(fact_value):
            return False, "sensitive_value_detected"
        return True, ""


DEFAULT_MEMORY_POLICY = MemoryPolicy()


class MemoryRepository:
    """SQLite-backed cross-thread memory with audit, expiry and soft deletion."""

    def __init__(
        self,
        path: Path | None = None,
        *,
        policy: MemoryPolicy | None = None,
    ) -> None:
        self.path = path or _default_path()
        self.policy = policy or DEFAULT_MEMORY_POLICY
        self.path.parent.mkdir(parents=True, exist_ok=True)
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
                CREATE TABLE IF NOT EXISTS customer_memories (
                    memory_id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL,
                    fact_type TEXT NOT NULL,
                    fact_value TEXT NOT NULL,
                    source TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    status TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    UNIQUE(customer_id, fact_type)
                );

                CREATE INDEX IF NOT EXISTS idx_memory_customer_status
                ON customer_memories(customer_id, status, expires_at);

                CREATE TABLE IF NOT EXISTS customer_memory_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT NOT NULL,
                    customer_id TEXT NOT NULL,
                    fact_type TEXT NOT NULL,
                    action TEXT NOT NULL,
                    old_value TEXT NOT NULL,
                    new_value TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _decode(row: sqlite3.Row | None) -> dict[str, Any] | None:
        return dict(row) if row is not None else None

    def _record_event(
        self,
        connection: sqlite3.Connection,
        *,
        memory_id: str,
        customer_id: str,
        fact_type: str,
        action: str,
        old_value: str,
        new_value: str,
        source: str,
        created_at: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO customer_memory_events (
                memory_id, customer_id, fact_type, action, old_value,
                new_value, source, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory_id,
                customer_id,
                fact_type,
                action,
                old_value,
                new_value,
                source,
                created_at,
            ),
        )

    def upsert_fact(
        self,
        *,
        customer_id: str,
        fact_type: str,
        fact_value: Any,
        source: str,
        confidence: float,
        ttl_days: int | None = None,
        expires_at: str | None = None,
    ) -> dict[str, Any]:
        normalized_customer = str(customer_id).strip()
        normalized_type = str(fact_type).strip()
        normalized_value = str(fact_value).strip()
        if not normalized_customer:
            raise ValueError("customer_id 不能为空。")
        allowed, reason = self.policy.validate(normalized_type, normalized_value)
        if not allowed:
            raise ValueError(f"长期记忆被策略拒绝：{reason}")

        now = _utc_now()
        now_text = _timestamp(now)
        effective_ttl = ttl_days if ttl_days is not None else DEFAULT_TTL_DAYS[normalized_type]
        expiry = expires_at or _timestamp(now + timedelta(days=effective_ttl))
        memory_id = _memory_id(normalized_customer, normalized_type)
        bounded_confidence = min(1.0, max(0.0, float(confidence)))

        with self._connect() as connection:
            existing = connection.execute(
                """
                SELECT * FROM customer_memories
                WHERE customer_id = ? AND fact_type = ?
                """,
                (normalized_customer, normalized_type),
            ).fetchone()
            old_value = str(existing["fact_value"]) if existing else ""
            revision = int(existing["revision"]) + 1 if existing else 1
            created_at = str(existing["created_at"]) if existing else now_text
            action = "create" if existing is None else (
                "refresh" if old_value == normalized_value else "correct"
            )
            connection.execute(
                """
                INSERT INTO customer_memories (
                    memory_id, customer_id, fact_type, fact_value, source,
                    confidence, status, revision, created_at, updated_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?)
                ON CONFLICT(customer_id, fact_type) DO UPDATE SET
                    fact_value = excluded.fact_value,
                    source = excluded.source,
                    confidence = excluded.confidence,
                    status = 'active',
                    revision = excluded.revision,
                    updated_at = excluded.updated_at,
                    expires_at = excluded.expires_at
                """,
                (
                    memory_id,
                    normalized_customer,
                    normalized_type,
                    normalized_value,
                    str(source).strip() or "unknown",
                    bounded_confidence,
                    revision,
                    created_at,
                    now_text,
                    expiry,
                ),
            )
            self._record_event(
                connection,
                memory_id=memory_id,
                customer_id=normalized_customer,
                fact_type=normalized_type,
                action=action,
                old_value=old_value,
                new_value=normalized_value,
                source=str(source).strip() or "unknown",
                created_at=now_text,
            )
        item = self.get_fact(normalized_customer, normalized_type)
        if item is None:
            raise RuntimeError("长期记忆写入失败。")
        return item

    def expire_due(self) -> int:
        now = _timestamp()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM customer_memories
                WHERE status = 'active' AND expires_at <= ?
                """,
                (now,),
            ).fetchall()
            for row in rows:
                connection.execute(
                    """
                    UPDATE customer_memories
                    SET status = 'expired', updated_at = ?
                    WHERE memory_id = ?
                    """,
                    (now, row["memory_id"]),
                )
                self._record_event(
                    connection,
                    memory_id=str(row["memory_id"]),
                    customer_id=str(row["customer_id"]),
                    fact_type=str(row["fact_type"]),
                    action="expire",
                    old_value=str(row["fact_value"]),
                    new_value="",
                    source="retention_policy",
                    created_at=now,
                )
        return len(rows)

    def get_fact(self, customer_id: str, fact_type: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM customer_memories
                WHERE customer_id = ? AND fact_type = ?
                """,
                (str(customer_id).strip(), str(fact_type).strip()),
            ).fetchone()
        return self._decode(row)

    def list_active(self, customer_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
        normalized_customer = str(customer_id).strip()
        if not normalized_customer:
            return []
        self.expire_due()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM customer_memories
                WHERE customer_id = ? AND status = 'active'
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (normalized_customer, max(1, int(limit))),
            ).fetchall()
        return [dict(row) for row in rows]

    def correct_fact(
        self,
        *,
        customer_id: str,
        fact_type: str,
        fact_value: Any,
        source: str = "customer_correction",
        confidence: float = 1.0,
    ) -> dict[str, Any]:
        return self.upsert_fact(
            customer_id=customer_id,
            fact_type=fact_type,
            fact_value=fact_value,
            source=source,
            confidence=confidence,
        )

    def delete_fact(
        self,
        *,
        customer_id: str,
        fact_type: str,
        source: str = "customer_forget_request",
    ) -> bool:
        now = _timestamp()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM customer_memories
                WHERE customer_id = ? AND fact_type = ?
                """,
                (str(customer_id).strip(), str(fact_type).strip()),
            ).fetchone()
            if row is None:
                return False
            connection.execute(
                """
                UPDATE customer_memories
                SET status = 'deleted', updated_at = ?
                WHERE memory_id = ?
                """,
                (now, row["memory_id"]),
            )
            self._record_event(
                connection,
                memory_id=str(row["memory_id"]),
                customer_id=str(row["customer_id"]),
                fact_type=str(row["fact_type"]),
                action="delete",
                old_value=str(row["fact_value"]),
                new_value="",
                source=source,
                created_at=now,
            )
        return True

    def list_events(self, customer_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM customer_memory_events
                WHERE customer_id = ?
                ORDER BY event_id DESC
                LIMIT ?
                """,
                (str(customer_id).strip(), max(1, int(limit))),
            ).fetchall()
        return [dict(row) for row in rows]

    def remember_profile_slots(
        self,
        *,
        customer_id: str,
        slots: dict[str, Any],
        source: str,
        confidence: float,
    ) -> list[dict[str, Any]]:
        if not str(customer_id).strip() or confidence < 0.65:
            return []
        writes: list[dict[str, Any]] = []
        for fact_type in sorted(ALLOWED_PROFILE_FACTS):
            value = slots.get(fact_type)
            if value is None:
                continue
            allowed, reason = self.policy.validate(fact_type, value)
            if not allowed:
                writes.append(
                    {
                        "fact_type": fact_type,
                        "status": "rejected",
                        "reason": reason,
                    }
                )
                continue
            writes.append(
                self.upsert_fact(
                    customer_id=customer_id,
                    fact_type=fact_type,
                    fact_value=value,
                    source=source,
                    confidence=confidence,
                )
            )
        return writes


DEFAULT_MEMORY_REPOSITORY = MemoryRepository()
