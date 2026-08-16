from __future__ import annotations

import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from settings import RAG_STATE_DB_PATH


CONVERSATION_STATE_FIELDS = (
    "brand",
    "machine_model",
    "part_name",
    "part_number",
    "quality_level",
    "destination",
    "fault_description",
)
MESSAGE_ROLES = {"user", "assistant", "system", "tool"}


class _ClosingConnection(sqlite3.Connection):
    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _clean_text(value: Any, *, max_length: int) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", str(value or ""))
    return text.strip()[:max_length]


def _row_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


class RagHistoryRepository:
    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path or RAG_STATE_DB_PATH)
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path, factory=_ClosingConnection)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS rag_conversations (
                    conversation_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS rag_messages (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    rewritten_query TEXT NOT NULL DEFAULT '',
                    answer_status TEXT NOT NULL DEFAULT '',
                    retrieval_status TEXT NOT NULL DEFAULT '',
                    model_name TEXT NOT NULL DEFAULT '',
                    cache_hit INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY(conversation_id)
                        REFERENCES rag_conversations(conversation_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS rag_message_citations (
                    citation_id TEXT PRIMARY KEY,
                    message_id TEXT NOT NULL,
                    document_id TEXT,
                    version_id TEXT,
                    chunk_id TEXT,
                    document_name TEXT NOT NULL,
                    page_or_sheet TEXT,
                    section TEXT,
                    retrieval_rank INTEGER,
                    retrieval_distance REAL,
                    FOREIGN KEY(message_id)
                        REFERENCES rag_messages(message_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS rag_conversation_state (
                    conversation_id TEXT PRIMARY KEY,
                    brand TEXT NOT NULL DEFAULT '',
                    machine_model TEXT NOT NULL DEFAULT '',
                    part_name TEXT NOT NULL DEFAULT '',
                    part_number TEXT NOT NULL DEFAULT '',
                    quality_level TEXT NOT NULL DEFAULT '',
                    destination TEXT NOT NULL DEFAULT '',
                    fault_description TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(conversation_id)
                        REFERENCES rag_conversations(conversation_id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS ix_rag_conversations_user_updated
                ON rag_conversations(user_id, status, updated_at);

                CREATE INDEX IF NOT EXISTS ix_rag_messages_conversation_created
                ON rag_messages(conversation_id, created_at);

                CREATE INDEX IF NOT EXISTS ix_rag_citations_message
                ON rag_message_citations(message_id, retrieval_rank);
                """
            )
            message_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(rag_messages)"
                ).fetchall()
            }
            if "cache_hit" not in message_columns:
                connection.execute(
                    """
                    ALTER TABLE rag_messages
                    ADD COLUMN cache_hit INTEGER NOT NULL DEFAULT 0
                    """
                )

    @staticmethod
    def _require_user_id(user_id: str) -> str:
        clean_user_id = _clean_text(user_id, max_length=100)
        if not clean_user_id:
            raise ValueError("user_id 不能为空。")
        return clean_user_id

    @staticmethod
    def _owned_conversation(
        connection: sqlite3.Connection,
        user_id: str,
        conversation_id: str,
    ) -> sqlite3.Row:
        row = connection.execute(
            """
            SELECT *
            FROM rag_conversations
            WHERE user_id = ? AND conversation_id = ?
            """,
            (user_id, conversation_id),
        ).fetchone()
        if row is None:
            raise PermissionError("会话不存在或无权访问。")
        return row

    def create_conversation(
        self,
        user_id: str,
        title: str = "新会话",
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        clean_user_id = self._require_user_id(user_id)
        clean_title = _clean_text(title, max_length=100) or "新会话"
        new_id = _clean_text(conversation_id, max_length=100) or uuid.uuid4().hex
        now = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO rag_conversations (
                    conversation_id, user_id, title, status, created_at, updated_at
                ) VALUES (?, ?, ?, 'active', ?, ?)
                """,
                (new_id, clean_user_id, clean_title, now, now),
            )
            connection.execute(
                """
                INSERT INTO rag_conversation_state (
                    conversation_id, updated_at
                ) VALUES (?, ?)
                """,
                (new_id, now),
            )
        return self.get_conversation(clean_user_id, new_id) or {}

    def list_conversations(
        self,
        user_id: str,
        *,
        include_archived: bool = False,
    ) -> list[dict[str, Any]]:
        clean_user_id = self._require_user_id(user_id)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM rag_conversations
                WHERE user_id = ?
                  AND (? = 1 OR status = 'active')
                ORDER BY updated_at DESC, created_at DESC
                """,
                (clean_user_id, int(include_archived)),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_conversation(
        self,
        user_id: str,
        conversation_id: str,
    ) -> dict[str, Any] | None:
        clean_user_id = self._require_user_id(user_id)
        clean_conversation_id = _clean_text(conversation_id, max_length=100)
        if not clean_conversation_id:
            return None
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM rag_conversations
                WHERE user_id = ? AND conversation_id = ?
                """,
                (clean_user_id, clean_conversation_id),
            ).fetchone()
        return _row_dict(row)

    def rename_conversation(
        self,
        user_id: str,
        conversation_id: str,
        title: str,
    ) -> bool:
        clean_user_id = self._require_user_id(user_id)
        clean_title = _clean_text(title, max_length=100)
        if not clean_title:
            raise ValueError("会话标题不能为空。")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._owned_conversation(connection, clean_user_id, conversation_id)
            cursor = connection.execute(
                """
                UPDATE rag_conversations
                SET title = ?, updated_at = ?
                WHERE user_id = ? AND conversation_id = ?
                """,
                (clean_title, _utc_now(), clean_user_id, conversation_id),
            )
        return cursor.rowcount == 1

    def archive_conversation(self, user_id: str, conversation_id: str) -> bool:
        clean_user_id = self._require_user_id(user_id)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._owned_conversation(connection, clean_user_id, conversation_id)
            cursor = connection.execute(
                """
                UPDATE rag_conversations
                SET status = 'archived', updated_at = ?
                WHERE user_id = ? AND conversation_id = ?
                """,
                (_utc_now(), clean_user_id, conversation_id),
            )
        return cursor.rowcount == 1

    def add_message(
        self,
        user_id: str,
        conversation_id: str,
        role: str,
        content: str,
        *,
        rewritten_query: str = "",
        answer_status: str = "",
        retrieval_status: str = "",
        model_name: str = "",
        cache_hit: bool = False,
    ) -> dict[str, Any]:
        clean_user_id = self._require_user_id(user_id)
        clean_role = _clean_text(role, max_length=20).lower()
        if clean_role not in MESSAGE_ROLES:
            raise ValueError(f"不支持的消息角色：{clean_role}")
        clean_content = _clean_text(content, max_length=30000)
        if not clean_content:
            raise ValueError("消息内容不能为空。")
        message_id = uuid.uuid4().hex
        now = _utc_now()
        suggested_title = clean_content.replace("\n", " ")[:30] or "新会话"
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._owned_conversation(connection, clean_user_id, conversation_id)
            connection.execute(
                """
                INSERT INTO rag_messages (
                    message_id, conversation_id, role, content, created_at,
                    rewritten_query, answer_status, retrieval_status, model_name,
                    cache_hit
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    conversation_id,
                    clean_role,
                    clean_content,
                    now,
                    _clean_text(rewritten_query, max_length=2000),
                    _clean_text(answer_status, max_length=50),
                    _clean_text(retrieval_status, max_length=50),
                    _clean_text(model_name, max_length=100),
                    int(bool(cache_hit)),
                ),
            )
            connection.execute(
                """
                UPDATE rag_conversations
                SET title = CASE
                        WHEN title = '新会话' AND ? = 'user' THEN ?
                        ELSE title
                    END,
                    updated_at = ?
                WHERE user_id = ? AND conversation_id = ?
                """,
                (clean_role, suggested_title, now, clean_user_id, conversation_id),
            )
        return self.get_message(clean_user_id, conversation_id, message_id)

    def get_message(
        self,
        user_id: str,
        conversation_id: str,
        message_id: str,
    ) -> dict[str, Any]:
        clean_user_id = self._require_user_id(user_id)
        with self._connect() as connection:
            self._owned_conversation(connection, clean_user_id, conversation_id)
            row = connection.execute(
                """
                SELECT m.*
                FROM rag_messages AS m
                JOIN rag_conversations AS c
                  ON c.conversation_id = m.conversation_id
                WHERE c.user_id = ?
                  AND m.conversation_id = ?
                  AND m.message_id = ?
                """,
                (clean_user_id, conversation_id, message_id),
            ).fetchone()
        if row is None:
            raise PermissionError("消息不存在或无权访问。")
        return dict(row)

    def update_message(
        self,
        user_id: str,
        conversation_id: str,
        message_id: str,
        *,
        rewritten_query: str | None = None,
        answer_status: str | None = None,
        retrieval_status: str | None = None,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        current = self.get_message(user_id, conversation_id, message_id)
        values = {
            "rewritten_query": current["rewritten_query"]
            if rewritten_query is None
            else _clean_text(rewritten_query, max_length=2000),
            "answer_status": current["answer_status"]
            if answer_status is None
            else _clean_text(answer_status, max_length=50),
            "retrieval_status": current["retrieval_status"]
            if retrieval_status is None
            else _clean_text(retrieval_status, max_length=50),
            "model_name": current["model_name"]
            if model_name is None
            else _clean_text(model_name, max_length=100),
        }
        clean_user_id = self._require_user_id(user_id)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._owned_conversation(connection, clean_user_id, conversation_id)
            connection.execute(
                """
                UPDATE rag_messages
                SET rewritten_query = ?,
                    answer_status = ?,
                    retrieval_status = ?,
                    model_name = ?
                WHERE message_id = ? AND conversation_id = ?
                """,
                (
                    values["rewritten_query"],
                    values["answer_status"],
                    values["retrieval_status"],
                    values["model_name"],
                    message_id,
                    conversation_id,
                ),
            )
        return self.get_message(clean_user_id, conversation_id, message_id)

    def list_messages(
        self,
        user_id: str,
        conversation_id: str,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        clean_user_id = self._require_user_id(user_id)
        with self._connect() as connection:
            self._owned_conversation(connection, clean_user_id, conversation_id)
            if limit is None:
                rows = connection.execute(
                    """
                    SELECT rowid AS sequence_no, *
                    FROM rag_messages
                    WHERE conversation_id = ?
                    ORDER BY created_at, rowid
                    """,
                    (conversation_id,),
                ).fetchall()
            else:
                safe_limit = max(1, min(int(limit), 100))
                rows = connection.execute(
                    """
                    SELECT *
                    FROM (
                        SELECT rowid AS sequence_no, *
                        FROM rag_messages
                        WHERE conversation_id = ?
                        ORDER BY created_at DESC, rowid DESC
                        LIMIT ?
                    )
                    ORDER BY created_at, sequence_no
                    """,
                    (conversation_id, safe_limit),
                ).fetchall()
        return [dict(row) for row in rows]

    def get_recent_history(
        self,
        user_id: str,
        conversation_id: str,
        *,
        max_messages: int = 12,
    ) -> list[dict[str, Any]]:
        return self.list_messages(
            user_id,
            conversation_id,
            limit=max(1, min(int(max_messages), 12)),
        )

    def add_citations(
        self,
        user_id: str,
        conversation_id: str,
        message_id: str,
        citations: Iterable[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        clean_user_id = self._require_user_id(user_id)
        citation_rows = list(citations)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._owned_conversation(connection, clean_user_id, conversation_id)
            message = connection.execute(
                """
                SELECT message_id
                FROM rag_messages
                WHERE conversation_id = ? AND message_id = ?
                """,
                (conversation_id, message_id),
            ).fetchone()
            if message is None:
                raise PermissionError("消息不存在或无权写入引用。")
            connection.execute(
                """
                DELETE FROM rag_message_citations
                WHERE message_id = ?
                """,
                (message_id,),
            )
            for citation in citation_rows:
                connection.execute(
                    """
                    INSERT INTO rag_message_citations (
                        citation_id, message_id, document_id, version_id,
                        chunk_id, document_name, page_or_sheet, section,
                        retrieval_rank, retrieval_distance
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        uuid.uuid4().hex,
                        message_id,
                        _clean_text(citation.get("document_id"), max_length=100) or None,
                        _clean_text(citation.get("version_id"), max_length=100) or None,
                        _clean_text(citation.get("chunk_id"), max_length=100) or None,
                        _clean_text(
                            citation.get("document_name")
                            or citation.get("source")
                            or "unknown",
                            max_length=300,
                        ),
                        _clean_text(citation.get("page_or_sheet"), max_length=200) or None,
                        _clean_text(citation.get("section"), max_length=300) or None,
                        citation.get("retrieval_rank"),
                        citation.get("retrieval_distance"),
                    ),
                )
        return self.get_message_citations(clean_user_id, conversation_id, message_id)

    def get_message_citations(
        self,
        user_id: str,
        conversation_id: str,
        message_id: str,
    ) -> list[dict[str, Any]]:
        clean_user_id = self._require_user_id(user_id)
        with self._connect() as connection:
            self._owned_conversation(connection, clean_user_id, conversation_id)
            rows = connection.execute(
                """
                SELECT ci.*
                FROM rag_message_citations AS ci
                JOIN rag_messages AS m ON m.message_id = ci.message_id
                WHERE m.conversation_id = ? AND ci.message_id = ?
                ORDER BY ci.retrieval_rank, ci.citation_id
                """,
                (conversation_id, message_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def citations_are_active(self, citations: Iterable[dict[str, Any]]) -> bool:
        version_ids = {
            _clean_text(citation.get("version_id"), max_length=100)
            for citation in citations
            if citation.get("version_id")
        }
        if not version_ids:
            return True
        with self._connect() as connection:
            table_exists = connection.execute(
                """
                SELECT 1
                FROM sqlite_master
                WHERE type = ? AND name = ?
                """,
                ("table", "rag_document_versions"),
            ).fetchone()
            if table_exists is None:
                return False
            for version_id in version_ids:
                active = connection.execute(
                    """
                    SELECT 1
                    FROM rag_document_versions AS v
                    JOIN rag_documents AS d
                      ON d.document_id = v.document_id
                    WHERE v.version_id = ?
                      AND v.is_active = 1
                      AND v.parse_status = 'parsed'
                      AND d.is_active = 1
                    """,
                    (version_id,),
                ).fetchone()
                if active is None:
                    return False
        return True

    def get_conversation_state(
        self,
        user_id: str,
        conversation_id: str,
    ) -> dict[str, str]:
        clean_user_id = self._require_user_id(user_id)
        with self._connect() as connection:
            self._owned_conversation(connection, clean_user_id, conversation_id)
            row = connection.execute(
                """
                SELECT brand, machine_model, part_name, part_number,
                       quality_level, destination, fault_description
                FROM rag_conversation_state
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()
        if row is None:
            return {field: "" for field in CONVERSATION_STATE_FIELDS}
        return {field: str(row[field] or "") for field in CONVERSATION_STATE_FIELDS}

    def update_conversation_state(
        self,
        user_id: str,
        conversation_id: str,
        updates: dict[str, Any],
    ) -> dict[str, str]:
        unknown_fields = set(updates) - set(CONVERSATION_STATE_FIELDS)
        if unknown_fields:
            raise ValueError(f"不支持的会话状态字段：{sorted(unknown_fields)}")
        current = self.get_conversation_state(user_id, conversation_id)
        merged = {
            field: _clean_text(updates.get(field, current[field]), max_length=1000)
            for field in CONVERSATION_STATE_FIELDS
        }
        clean_user_id = self._require_user_id(user_id)
        now = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._owned_conversation(connection, clean_user_id, conversation_id)
            connection.execute(
                """
                UPDATE rag_conversation_state
                SET brand = ?, machine_model = ?, part_name = ?,
                    part_number = ?, quality_level = ?, destination = ?,
                    fault_description = ?, updated_at = ?
                WHERE conversation_id = ?
                """,
                (
                    merged["brand"],
                    merged["machine_model"],
                    merged["part_name"],
                    merged["part_number"],
                    merged["quality_level"],
                    merged["destination"],
                    merged["fault_description"],
                    now,
                    conversation_id,
                ),
            )
            connection.execute(
                """
                UPDATE rag_conversations
                SET updated_at = ?
                WHERE user_id = ? AND conversation_id = ?
                """,
                (now, clean_user_id, conversation_id),
            )
        return self.get_conversation_state(clean_user_id, conversation_id)
