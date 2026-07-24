"""Session adapters — SingleTurn (no-op) and Local (SQLite)."""
from __future__ import annotations

from contextlib import asynccontextmanager
from uuid import uuid4

import aiosqlite
import ujson
from pydantic import TypeAdapter
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse

from agent_framework.types import SessionManager

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at TEXT DEFAULT (datetime('now')),
    metadata TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS messages (
    entry_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    turn_index INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_messages_session_turn
    ON messages(session_id, turn_index);
"""

_MessageAdapter: TypeAdapter[ModelMessage] = TypeAdapter(ModelMessage)
"""SDK-backed serializer for a single message (DB stores one message per row)."""


class SingleTurnSessionManager(SessionManager):
    async def create_session(self, *, metadata: dict | None = None) -> str:
        return str(uuid4())

    async def load_history(self, session_id: str) -> list:
        return []

    async def save_messages(self, session_id: str, messages: list) -> None:
        """No-op: single-turn sessions do not persist history."""
        pass

    async def apply_compaction(self, session_id: str, summary: str, boundary_entry_id: str) -> None:
        raise NotImplementedError("SingleTurnSessionManager does not support compaction")


class LocalSessionManager(SessionManager):
    def __init__(self, *, db_path: str):
        self._db_path = db_path
        self._schema_initialized = False

    @asynccontextmanager
    async def _connect(self):
        db = await aiosqlite.connect(self._db_path)
        db.row_factory = aiosqlite.Row
        if not self._schema_initialized:
            await db.executescript(SCHEMA)
            await db.commit()
            self._schema_initialized = True
        try:
            yield db
        finally:
            await db.close()

    async def create_session(self, *, metadata: dict | None = None) -> str:
        sid = str(uuid4())
        async with self._connect() as db:
            await db.execute(
                "INSERT INTO sessions (session_id, metadata) VALUES (?, ?)",
                (sid, ujson.dumps(metadata or {})),
            )
            await db.commit()
        return sid

    async def load_history(self, session_id: str) -> list:
        async with self._connect() as db:
            cursor = await db.execute(
                "SELECT content FROM messages WHERE session_id = ? AND role != 'compaction' ORDER BY turn_index",
                (session_id,),
            )
            rows = await cursor.fetchall()
            return [_MessageAdapter.validate_json(row["content"].encode()) for row in rows]

    async def save_messages(self, session_id: str, messages: list) -> None:
        async with self._connect() as db:
            cursor = await db.execute(
                "SELECT COALESCE(MAX(turn_index), -1) + 1 AS next_turn FROM messages WHERE session_id = ?",
                (session_id,),
            )
            row = await cursor.fetchone()
            turn = row["next_turn"]
            for msg in messages:
                role = _infer_role(msg)
                content = _MessageAdapter.dump_json(msg).decode()
                await db.execute(
                    "INSERT INTO messages (entry_id, session_id, turn_index, role, content) VALUES (?, ?, ?, ?, ?)",
                    (str(uuid4()), session_id, turn, role, content),
                )
                turn += 1
            await db.commit()

    async def apply_compaction(self, session_id: str, summary: str, boundary_entry_id: str) -> None:
        async with self._connect() as db:
            await db.execute(
                "INSERT INTO messages (entry_id, session_id, turn_index, role, content) VALUES (?, ?, -1, ?, ?)",
                (str(uuid4()), session_id, "compaction", ujson.dumps({
                    "type": "compaction", "summary": summary, "boundary_entry_id": boundary_entry_id,
                })),
            )
            await db.commit()

    async def cleanup_stale_sessions(self, timeout_seconds: int | None = None) -> int:
        """Delete sessions older than ``timeout_seconds`` (defaults to 1 day)."""
        timeout = timeout_seconds if timeout_seconds is not None else 86400
        async with self._connect() as db:
            cursor = await db.execute(
                "DELETE FROM sessions WHERE created_at < datetime('now', '-{} seconds')".format(timeout),
            )
            await db.commit()
            return cursor.rowcount


def _infer_role(msg) -> str:
    if isinstance(msg, ModelRequest):
        parts = msg.parts
        if parts and hasattr(parts[0], "tool_name"):
            return "tool"
        return "user"
    if isinstance(msg, ModelResponse):
        return "assistant"
    return "unknown"
