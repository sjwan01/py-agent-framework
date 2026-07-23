"""Session adapters — SingleTurn (no-op) and Local (SQLite)."""
from __future__ import annotations

from uuid import uuid4

import aiosqlite
import ujson

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


class SingleTurnSessionManager(SessionManager):
    async def create_session(self, *, metadata: dict | None = None) -> str:
        return str(uuid4())

    async def load_history(self, session_id: str) -> list:
        return []

    async def save_messages(self, session_id: str, messages: list) -> None:
        pass

    async def apply_compaction(self, session_id: str, summary: str, boundary_entry_id: str) -> None:
        raise NotImplementedError("SingleTurnSessionManager does not support compaction")


class LocalSessionManager(SessionManager):
    def __init__(self, *, db_path: str, serializer):
        self._db_path = db_path
        self._serializer = serializer
        self._db: aiosqlite.Connection | None = None

    async def _get_db(self) -> aiosqlite.Connection:
        if self._db is None:
            self._db = await aiosqlite.connect(self._db_path)
            self._db.row_factory = aiosqlite.Row
            await self._db.executescript(SCHEMA)
            await self._db.commit()
        return self._db

    async def create_session(self, *, metadata: dict | None = None) -> str:
        db = await self._get_db()
        sid = str(uuid4())
        await db.execute(
            "INSERT INTO sessions (session_id, metadata) VALUES (?, ?)",
            (sid, ujson.dumps(metadata or {})),
        )
        await db.commit()
        return sid

    async def load_history(self, session_id: str) -> list:
        db = await self._get_db()
        cursor = await db.execute(
            "SELECT content FROM messages WHERE session_id = ? ORDER BY turn_index",
            (session_id,),
        )
        rows = await cursor.fetchall()
        return [self._serializer.deserialize(row["content"]) for row in rows]

    async def save_messages(self, session_id: str, messages: list) -> None:
        db = await self._get_db()
        cursor = await db.execute(
            "SELECT COALESCE(MAX(turn_index), -1) + 1 AS next_turn FROM messages WHERE session_id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        turn = row["next_turn"]
        for msg in messages:
            role = _infer_role(msg)
            content = self._serializer.serialize(msg)
            await db.execute(
                "INSERT INTO messages (entry_id, session_id, turn_index, role, content) VALUES (?, ?, ?, ?, ?)",
                (str(uuid4()), session_id, turn, role, content),
            )
            turn += 1
        await db.commit()

    async def apply_compaction(self, session_id: str, summary: str, boundary_entry_id: str) -> None:
        db = await self._get_db()
        await db.execute(
            "INSERT INTO messages (entry_id, session_id, turn_index, role, content) VALUES (?, ?, -1, ?, ?)",
            (str(uuid4()), session_id, "compaction", ujson.dumps({
                "type": "compaction", "summary": summary, "boundary": boundary_entry_id,
            })),
        )
        await db.commit()


def _infer_role(msg) -> str:
    from pydantic_ai.messages import ModelRequest, ModelResponse
    if isinstance(msg, ModelRequest):
        parts = msg.parts
        if parts and hasattr(parts[0], "tool_name"):
            return "tool"
        return "user"
    if isinstance(msg, ModelResponse):
        return "assistant"
    return "unknown"
