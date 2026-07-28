"""SQLite session implementation."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from uuid import uuid4

import aiosqlite
from pydantic_ai.messages import ModelRequest, UserPromptPart

from py_agent.session._shared import _infer_role, _is_turn_start, _MessageAdapter
from py_agent.types import SessionManager

# SQLite schema
# sessions: one row per session, with creation time and custom metadata.
# messages: one row per message, indexed by (session_id, message_seq).
#   - entry_id: unique message identifier (UUID).
#   - message_seq: monotonic sequence number starting at 0.
#   - role: message role (user / assistant / tool).
#   - content: JSON string serialized by _MessageAdapter.
# compactions: one row per compaction, with boundary_seq marking the summarized range.
SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at TEXT DEFAULT (datetime('now')),
    metadata TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS messages (
    entry_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    message_seq INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_messages_session_seq
    ON messages(session_id, message_seq);

CREATE TABLE IF NOT EXISTS compactions (
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    boundary_seq INTEGER NOT NULL,
    summary TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);
"""


class LocalSessionManager(SessionManager):
    """SQLite-backed multi-turn session.

    Args:
        db_path: Path to the SQLite database file.

    Creates tables on first use. Supports compaction using ``boundary_seq``.
    """

    def __init__(self, *, db_path: str):
        # path to the SQLite database file
        self._db_path = db_path
        # avoid re-running schema creation on every connection
        self._schema_initialized = False

    @asynccontextmanager
    async def _connect(self):
        """Acquire a SQLite connection, creating tables on first connect."""
        db = await aiosqlite.connect(self._db_path)
        # enable column access on rows, e.g. row["content"]
        db.row_factory = aiosqlite.Row
        if not self._schema_initialized:
            await db.executescript(SCHEMA)
            await db.commit()
            self._schema_initialized = True
        try:
            yield db
        finally:
            await db.close()

    # SessionManager interface implementation

    async def create_session(self, *, metadata: dict | None = None) -> str:
        """Insert a row into ``sessions`` and return the new ``session_id``."""
        sid = str(uuid4())
        async with self._connect() as db:
            await db.execute(
                "INSERT INTO sessions (session_id, metadata) VALUES (?, ?)",
                (sid, json.dumps(metadata or {})),
            )
            await db.commit()
        return sid

    async def ensure_session(self, session_id: str, *, metadata: dict | None = None) -> str:
        """Ensure the session row exists, creating it if necessary."""
        async with self._connect() as db:
            await db.execute(
                "INSERT OR IGNORE INTO sessions (session_id, metadata) VALUES (?, ?)",
                (session_id, json.dumps(metadata or {})),
            )
            await db.commit()
        return session_id

    async def load_history(self, session_id: str, *, protect_turns: int = 0) -> list:
        """Load messages for the session.

        Checks compaction first: if no compaction record exists, loads all
        messages. If a record exists and at least ``protect_turns`` user turns
        have passed the boundary, loads only messages with ``seq > boundary``
        and prepends a summary.
        """
        async with self._connect() as db:
            # 1. fetch the latest compaction
            cursor = await db.execute(
                "SELECT boundary_seq, summary FROM compactions WHERE session_id = ? ORDER BY boundary_seq DESC LIMIT 1",
                (session_id,),
            )
            comp_row = await cursor.fetchone()

            if comp_row is None:
                # no compaction → load everything
                return await self._load_all_messages(db, session_id)

            boundary_seq = comp_row["boundary_seq"]
            summary = comp_row["summary"]

            # 2. load only messages after the boundary and count user turns
            recent = await self._load_messages_after(db, session_id, boundary_seq)
            turns_after = sum(1 for m in recent if _is_turn_start(m))

            if turns_after < protect_turns:
                # not enough turns past the boundary → fall back to full history
                return await self._load_all_messages(db, session_id)

            # 3. enable compaction: summary + messages after the boundary
            ts = datetime.now(timezone.utc)
            summary_msg = ModelRequest(
                parts=[UserPromptPart(
                    content=f"[Previous conversation summary]\n{summary}",
                    timestamp=ts,
                )],
                kind="request",
                timestamp=ts,
            )
            return [summary_msg] + recent

    async def get_max_message_seq(self, session_id: str) -> int:
        """Return the current maximum ``message_seq`` for the session, or ``-1`` if empty."""
        async with self._connect() as db:
            cursor = await db.execute(
                "SELECT COALESCE(MAX(message_seq), -1) FROM messages WHERE session_id = ?",
                (session_id,),
            )
            row = await cursor.fetchone()
            return row[0]

    async def _load_all_messages(self, db, session_id: str) -> list:
        cursor = await db.execute(
            "SELECT content FROM messages WHERE session_id = ? ORDER BY message_seq",
            (session_id,),
        )
        rows = await cursor.fetchall()
        return [_MessageAdapter.validate_json(row["content"].encode()) for row in rows]

    async def _load_messages_after(self, db, session_id: str, boundary_seq: int) -> list:
        cursor = await db.execute(
            "SELECT content FROM messages WHERE session_id = ? AND message_seq > ? ORDER BY message_seq",
            (session_id, boundary_seq),
        )
        rows = await cursor.fetchall()
        return [_MessageAdapter.validate_json(row["content"].encode()) for row in rows]

    async def save_messages(self, session_id: str, messages: list) -> None:
        """Append a batch of new messages (this turn's delta) to the ``messages`` table.

        Each message gets the next ``message_seq`` starting from the current
        maximum + 1.
        """
        async with self._connect() as db:
            cursor = await db.execute(
                "SELECT COALESCE(MAX(message_seq), -1) + 1 AS next_seq FROM messages WHERE session_id = ?",
                (session_id,),
            )
            row = await cursor.fetchone()
            seq = row["next_seq"]
            for msg in messages:
                role = _infer_role(msg)
                content = _MessageAdapter.dump_json(msg).decode()
                await db.execute(
                    "INSERT INTO messages (entry_id, session_id, message_seq, role, content) VALUES (?, ?, ?, ?, ?)",
                    (str(uuid4()), session_id, seq, role, content),
                )
                seq += 1
            await db.commit()

    async def apply_compaction(self, session_id: str, summary: str, boundary_seq: int) -> None:
        """Write a compaction record into the separate ``compactions`` table."""
        async with self._connect() as db:
            await db.execute(
                "INSERT INTO compactions (session_id, boundary_seq, summary) VALUES (?, ?, ?)",
                (session_id, boundary_seq, summary),
            )
            await db.commit()
