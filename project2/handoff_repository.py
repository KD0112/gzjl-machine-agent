from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / ".env")
DEFAULT_HANDOFF_DB_PATH = BASE_DIR / "logs" / "handoff_cases.sqlite3"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _default_path() -> Path:
    configured = os.getenv("HANDOFF_DB_PATH", "").strip()
    if not configured:
        return DEFAULT_HANDOFF_DB_PATH
    path = Path(configured)
    return path if path.is_absolute() else BASE_DIR / path


def _handoff_id(thread_id: str) -> str:
    digest = hashlib.sha256(thread_id.encode("utf-8")).hexdigest()[:12].upper()
    return f"HF-{digest}"


class HandoffRepository:
    """SQLite-backed human-service queue, separate from LangGraph checkpoints."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or _default_path()
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
                CREATE TABLE IF NOT EXISTS handoff_cases (
                    handoff_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL UNIQUE,
                    request_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reason_code TEXT NOT NULL,
                    reason_text TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    customer_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    context_json TEXT NOT NULL,
                    assigned_to TEXT NOT NULL DEFAULT '',
                    human_reply TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    claimed_at TEXT NOT NULL DEFAULT '',
                    resolved_at TEXT NOT NULL DEFAULT ''
                );

                CREATE INDEX IF NOT EXISTS idx_handoff_status_created
                ON handoff_cases(status, created_at);

                CREATE TABLE IF NOT EXISTS outbox_messages (
                    outbox_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dedupe_key TEXT NOT NULL UNIQUE,
                    handoff_id TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    customer_id TEXT NOT NULL,
                    message TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    delivered_at TEXT NOT NULL DEFAULT '',
                    last_error TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY(handoff_id) REFERENCES handoff_cases(handoff_id)
                );

                CREATE INDEX IF NOT EXISTS idx_outbox_status_created
                ON outbox_messages(status, created_at);
                """
            )

    @staticmethod
    def _decode_case(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        item = dict(row)
        try:
            item["context"] = json.loads(item.pop("context_json"))
        except json.JSONDecodeError:
            item["context"] = {}
            item.pop("context_json", None)
        return item

    def create_case(
        self,
        *,
        thread_id: str,
        request_id: str,
        reason_code: str,
        reason_text: str,
        priority: str,
        question: str,
        context: dict[str, Any],
        channel: str = "web",
        customer_id: str = "",
    ) -> dict[str, Any]:
        handoff_id = _handoff_id(thread_id)
        now = _utc_now()
        context_json = json.dumps(context, ensure_ascii=False, sort_keys=True)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO handoff_cases (
                    handoff_id, thread_id, request_id, status, reason_code,
                    reason_text, priority, channel, customer_id, question,
                    context_json, created_at, updated_at
                ) VALUES (?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    reason_code = excluded.reason_code,
                    reason_text = excluded.reason_text,
                    priority = excluded.priority,
                    context_json = excluded.context_json,
                    updated_at = excluded.updated_at
                """,
                (
                    handoff_id,
                    thread_id,
                    request_id,
                    reason_code,
                    reason_text,
                    priority,
                    channel,
                    customer_id,
                    question,
                    context_json,
                    now,
                    now,
                ),
            )
        case = self.get_case(handoff_id)
        if case is None:
            raise RuntimeError("人工服务单创建失败")
        return case

    def get_case(self, handoff_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM handoff_cases WHERE handoff_id = ?",
                (handoff_id,),
            ).fetchone()
        return self._decode_case(row)

    def get_case_by_thread(self, thread_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM handoff_cases WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
        return self._decode_case(row)

    def list_cases(self, status: str = "", limit: int = 100) -> list[dict[str, Any]]:
        query = "SELECT * FROM handoff_cases"
        params: list[Any] = []
        if status and status != "all":
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY CASE priority WHEN '高' THEN 0 WHEN '中' THEN 1 ELSE 2 END, created_at ASC LIMIT ?"
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._decode_case(row) or {} for row in rows]

    def claim_case(self, handoff_id: str, agent_name: str) -> dict[str, Any]:
        assigned_to = agent_name.strip()
        if not assigned_to:
            raise ValueError("客服名称不能为空")
        now = _utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE handoff_cases
                SET status = 'claimed', assigned_to = ?, claimed_at = CASE
                    WHEN claimed_at = '' THEN ? ELSE claimed_at END,
                    updated_at = ?
                WHERE handoff_id = ? AND status IN ('queued', 'claimed')
                """,
                (assigned_to, now, now, handoff_id),
            )
        case = self.get_case(handoff_id)
        if case is None:
            raise KeyError(f"人工服务单不存在：{handoff_id}")
        return case

    def resolve_case(
        self,
        handoff_id: str,
        *,
        human_reply: str,
        agent_name: str,
    ) -> dict[str, Any]:
        message = human_reply.strip()
        assigned_to = agent_name.strip()
        if not message or not assigned_to:
            raise ValueError("人工回复和客服名称不能为空")

        now = _utc_now()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM handoff_cases WHERE handoff_id = ?",
                (handoff_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"人工服务单不存在：{handoff_id}")

            connection.execute(
                """
                UPDATE handoff_cases
                SET status = 'resolved', assigned_to = ?, human_reply = ?,
                    resolved_at = CASE WHEN resolved_at = '' THEN ? ELSE resolved_at END,
                    updated_at = ?
                WHERE handoff_id = ?
                """,
                (assigned_to, message, now, now, handoff_id),
            )

            channel = str(row["channel"])
            customer_id = str(row["customer_id"])
            if channel != "web" and customer_id:
                dedupe_key = f"{handoff_id}:human_reply"
                connection.execute(
                    """
                    INSERT OR IGNORE INTO outbox_messages (
                        dedupe_key, handoff_id, channel, customer_id, message,
                        status, created_at
                    ) VALUES (?, ?, ?, ?, ?, 'pending', ?)
                    """,
                    (dedupe_key, handoff_id, channel, customer_id, message, now),
                )

        case = self.get_case(handoff_id)
        if case is None:
            raise RuntimeError("人工服务单更新失败")
        return case

    def list_outbox(self, status: str = "pending", limit: int = 100) -> list[dict[str, Any]]:
        query = "SELECT * FROM outbox_messages"
        params: list[Any] = []
        if status and status != "all":
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY created_at ASC LIMIT ?"
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def mark_outbox_delivered(self, outbox_id: int) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE outbox_messages
                SET status = 'delivered', delivered_at = ?, last_error = ''
                WHERE outbox_id = ?
                """,
                (_utc_now(), outbox_id),
            )

    def close(self) -> None:
        """Compatibility hook; connections are already short-lived per method."""


DEFAULT_HANDOFF_REPOSITORY = HandoffRepository()
