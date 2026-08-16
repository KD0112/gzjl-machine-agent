from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / ".env")
DEFAULT_CONVERSATION_DB_PATH = BASE_DIR / "logs" / "conversation_threads.sqlite3"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _default_path() -> Path:
    configured = os.getenv("CONVERSATION_DB_PATH", "").strip()
    if not configured:
        return DEFAULT_CONVERSATION_DB_PATH
    path = Path(configured)
    return path if path.is_absolute() else BASE_DIR / path


def _clean_text(value: Any, *, max_length: int) -> str:
    return " ".join(str(value or "").split())[:max_length]


def _auto_title(question: str) -> str:
    return _clean_text(question, max_length=48) or "新会话"


class ConversationAccessError(PermissionError):
    """Raised when a customer attempts to access another customer's thread."""


class ConversationRepository:
    """Product-level thread catalog kept separate from LangGraph checkpoints."""

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
                CREATE TABLE IF NOT EXISTS conversation_threads (
                    thread_id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    title_is_custom INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'new',
                    channel TEXT NOT NULL DEFAULT 'web',
                    execution_mode TEXT NOT NULL DEFAULT 'LangGraph',
                    turn_count INTEGER NOT NULL DEFAULT 0,
                    last_message_preview TEXT NOT NULL DEFAULT '',
                    last_request_id TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    archived_at TEXT NOT NULL DEFAULT ''
                );

                CREATE INDEX IF NOT EXISTS idx_conversation_customer_updated
                ON conversation_threads(customer_id, archived_at, updated_at DESC);

                CREATE TABLE IF NOT EXISTS conversation_catalog_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _decode(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        item = dict(row)
        item["title_is_custom"] = bool(item.get("title_is_custom"))
        item["archived"] = bool(item.get("archived_at"))
        return item

    @staticmethod
    def _require_identity(thread_id: str, customer_id: str) -> tuple[str, str]:
        clean_thread_id = _clean_text(thread_id, max_length=128)
        clean_customer_id = _clean_text(customer_id, max_length=128)
        if not clean_thread_id:
            raise ValueError("thread_id 不能为空")
        if not clean_customer_id:
            raise ValueError("customer_id 不能为空")
        return clean_thread_id, clean_customer_id

    def _get_owned_row(
        self,
        connection: sqlite3.Connection,
        thread_id: str,
        customer_id: str,
    ) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM conversation_threads WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"会话不存在：{thread_id}")
        if str(row["customer_id"]) != customer_id:
            raise ConversationAccessError("无权访问其他客户的会话")
        return row

    def create_thread(
        self,
        *,
        thread_id: str,
        customer_id: str,
        title: str = "新会话",
        channel: str = "web",
        execution_mode: str = "LangGraph",
    ) -> dict[str, Any]:
        clean_thread_id, clean_customer_id = self._require_identity(
            thread_id,
            customer_id,
        )
        clean_title = _clean_text(title, max_length=80) or "新会话"
        now = _utc_now()
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT * FROM conversation_threads WHERE thread_id = ?",
                (clean_thread_id,),
            ).fetchone()
            if existing is not None:
                if str(existing["customer_id"]) != clean_customer_id:
                    raise ConversationAccessError("同一 thread_id 不能绑定其他客户")
                return self._decode(existing) or {}
            connection.execute(
                """
                INSERT INTO conversation_threads (
                    thread_id, customer_id, title, status, channel,
                    execution_mode, created_at, updated_at
                ) VALUES (?, ?, ?, 'new', ?, ?, ?, ?)
                """,
                (
                    clean_thread_id,
                    clean_customer_id,
                    clean_title,
                    _clean_text(channel, max_length=32) or "web",
                    _clean_text(execution_mode, max_length=64) or "LangGraph",
                    now,
                    now,
                ),
            )
        item = self.get_thread(clean_thread_id, customer_id=clean_customer_id)
        if item is None:
            raise RuntimeError("会话目录创建失败")
        return item

    def record_result(
        self,
        result: dict[str, Any],
        *,
        customer_id: str,
        question: str = "",
        channel: str = "web",
        execution_mode: str = "LangGraph",
    ) -> dict[str, Any]:
        thread_id, clean_customer_id = self._require_identity(
            str(result.get("thread_id", "")),
            customer_id,
        )
        parsed_question = (
            question
            or (result.get("parse_result") or {}).get("raw_question", "")
            or result.get("question", "")
        )
        clean_question = _clean_text(parsed_question, max_length=240)
        status = _clean_text(result.get("status", "completed"), max_length=64)
        request_id = _clean_text(result.get("request_id", ""), max_length=128)
        turn_count = max(0, int(result.get("turn_count", 0) or 0))
        now = _utc_now()

        with self._connect() as connection:
            existing = connection.execute(
                "SELECT * FROM conversation_threads WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO conversation_threads (
                        thread_id, customer_id, title, status, channel,
                        execution_mode, turn_count, last_message_preview,
                        last_request_id, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        thread_id,
                        clean_customer_id,
                        _auto_title(clean_question),
                        status or "completed",
                        _clean_text(channel, max_length=32) or "web",
                        _clean_text(execution_mode, max_length=64) or "LangGraph",
                        turn_count,
                        clean_question,
                        request_id,
                        now,
                        now,
                    ),
                )
            else:
                if str(existing["customer_id"]) != clean_customer_id:
                    raise ConversationAccessError("同一 thread_id 不能绑定其他客户")
                title = str(existing["title"])
                if not bool(existing["title_is_custom"]) and int(existing["turn_count"]) == 0:
                    title = _auto_title(clean_question)
                connection.execute(
                    """
                    UPDATE conversation_threads
                    SET title = ?, status = ?, channel = ?, execution_mode = ?,
                        turn_count = ?, last_message_preview = ?,
                        last_request_id = ?, updated_at = ?
                    WHERE thread_id = ?
                    """,
                    (
                        title,
                        status or str(existing["status"]),
                        _clean_text(channel, max_length=32) or str(existing["channel"]),
                        _clean_text(execution_mode, max_length=64)
                        or str(existing["execution_mode"]),
                        turn_count,
                        clean_question or str(existing["last_message_preview"]),
                        request_id or str(existing["last_request_id"]),
                        now,
                        thread_id,
                    ),
                )

        item = self.get_thread(thread_id, customer_id=clean_customer_id)
        if item is None:
            raise RuntimeError("会话目录更新失败")
        return item

    def get_thread(
        self,
        thread_id: str,
        *,
        customer_id: str = "",
    ) -> dict[str, Any] | None:
        clean_thread_id = _clean_text(thread_id, max_length=128)
        if not clean_thread_id:
            return None
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM conversation_threads WHERE thread_id = ?",
                (clean_thread_id,),
            ).fetchone()
        item = self._decode(row)
        if item is None:
            return None
        clean_customer_id = _clean_text(customer_id, max_length=128)
        if clean_customer_id and item["customer_id"] != clean_customer_id:
            raise ConversationAccessError("无权访问其他客户的会话")
        return item

    def list_threads(
        self,
        customer_id: str,
        *,
        include_archived: bool = False,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        clean_customer_id = _clean_text(customer_id, max_length=128)
        if not clean_customer_id:
            return []
        query = "SELECT * FROM conversation_threads WHERE customer_id = ?"
        params: list[Any] = [clean_customer_id]
        if not include_archived:
            query += " AND archived_at = ''"
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(max(1, min(int(limit), 500)))
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._decode(row) or {} for row in rows]

    def rename_thread(
        self,
        thread_id: str,
        *,
        customer_id: str,
        title: str,
    ) -> dict[str, Any]:
        clean_thread_id, clean_customer_id = self._require_identity(
            thread_id,
            customer_id,
        )
        clean_title = _clean_text(title, max_length=80)
        if not clean_title:
            raise ValueError("会话标题不能为空")
        with self._connect() as connection:
            self._get_owned_row(connection, clean_thread_id, clean_customer_id)
            connection.execute(
                """
                UPDATE conversation_threads
                SET title = ?, title_is_custom = 1, updated_at = ?
                WHERE thread_id = ?
                """,
                (clean_title, _utc_now(), clean_thread_id),
            )
        return self.get_thread(
            clean_thread_id,
            customer_id=clean_customer_id,
        ) or {}

    def archive_thread(self, thread_id: str, *, customer_id: str) -> dict[str, Any]:
        clean_thread_id, clean_customer_id = self._require_identity(
            thread_id,
            customer_id,
        )
        now = _utc_now()
        with self._connect() as connection:
            self._get_owned_row(connection, clean_thread_id, clean_customer_id)
            connection.execute(
                """
                UPDATE conversation_threads
                SET archived_at = CASE WHEN archived_at = '' THEN ? ELSE archived_at END,
                    updated_at = ?
                WHERE thread_id = ?
                """,
                (now, now, clean_thread_id),
            )
        return self.get_thread(
            clean_thread_id,
            customer_id=clean_customer_id,
        ) or {}

    def restore_thread(self, thread_id: str, *, customer_id: str) -> dict[str, Any]:
        clean_thread_id, clean_customer_id = self._require_identity(
            thread_id,
            customer_id,
        )
        with self._connect() as connection:
            self._get_owned_row(connection, clean_thread_id, clean_customer_id)
            connection.execute(
                """
                UPDATE conversation_threads
                SET archived_at = '', updated_at = ?
                WHERE thread_id = ?
                """,
                (_utc_now(), clean_thread_id),
            )
        return self.get_thread(
            clean_thread_id,
            customer_id=clean_customer_id,
        ) or {}

    def get_metadata(self, key: str) -> str:
        clean_key = _clean_text(key, max_length=128)
        if not clean_key:
            return ""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM conversation_catalog_metadata WHERE key = ?",
                (clean_key,),
            ).fetchone()
        return str(row["value"]) if row is not None else ""

    def set_metadata(self, key: str, value: str) -> None:
        clean_key = _clean_text(key, max_length=128)
        if not clean_key:
            raise ValueError("metadata key 不能为空")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversation_catalog_metadata(key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (clean_key, str(value), _utc_now()),
            )

    def close(self) -> None:
        """Compatibility hook; connections are already short-lived per method."""


DEFAULT_CONVERSATION_REPOSITORY = ConversationRepository()
