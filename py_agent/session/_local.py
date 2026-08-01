"""SQLite session implementation."""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, cast
from uuid import uuid4

import aiosqlite
from pydantic_ai.messages import ModelRequest, UserPromptPart

from py_agent.session._shared import _infer_role, _MessageAdapter
from py_agent.types import SessionManager

# SQLite schema
# sessions: one row per session, with creation time and custom metadata.
# messages: one row per message, indexed by (session_id, message_seq).
#   - entry_id: unique message identifier (UUID).
#   - message_seq: monotonic sequence number starting at 0.
#   - role: message role (user / assistant / tool).
#   - content: JSON string serialized by _MessageAdapter.
# compactions: one row per compaction, with boundary_seq marking the summarized range.
# system_prompts: one row per system prompt write, ordered by id DESC for latest retrieval.
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

CREATE INDEX IF NOT EXISTS idx_messages_session_role
    ON messages(session_id, role, message_seq);

CREATE TABLE IF NOT EXISTS compactions (
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    boundary_seq INTEGER NOT NULL,
    summary TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_compactions_session_boundary
    ON compactions(session_id, boundary_seq);

CREATE TABLE IF NOT EXISTS system_prompts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    system_prompt TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_system_prompts_session
    ON system_prompts (session_id, id DESC);
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
    async def _connect(self) -> AsyncIterator[aiosqlite.Connection]:
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

    async def create_session(
        self, *, metadata: dict[str, Any] | None = None
    ) -> str:
        """Insert a row into ``sessions`` and return the new ``session_id``."""
        sid = str(uuid4())
        async with self._connect() as db:
            await db.execute(
                "INSERT INTO sessions (session_id, metadata) VALUES (?, ?)",
                (sid, json.dumps(metadata or {})),
            )
            await db.commit()
        return sid

    async def ensure_session(
        self, session_id: str, *, metadata: dict[str, Any] | None = None
    ) -> str:
        """Ensure the session row exists, creating it if necessary."""
        async with self._connect() as db:
            await db.execute(
                "INSERT OR IGNORE INTO sessions (session_id, metadata) VALUES (?, ?)",
                (session_id, json.dumps(metadata or {})),
            )
            await db.commit()
        return session_id

    async def load_history(
        self, session_id: str, *, protect_turns: int = 0
    ) -> list[Any]:
        """Load messages for the session.

        Walks backwards from the latest message to find the ``protect_turns``
        most recent user turns, then selects a compaction whose boundary falls
        before that protected region. Messages before the boundary are
        replaced by the compaction summary; messages after are loaded in full.

        Falls back to loading all messages when no eligible compaction exists.
        """
        async with self._connect() as db:
            # 1. get the most recent message seq
            cursor = await db.execute(
                "SELECT COALESCE(MAX(message_seq), -1) FROM messages WHERE session_id = ?",
                (session_id,),
            )
            row = await cursor.fetchone()
            max_seq = int(cast(aiosqlite.Row, row)[0])
            if max_seq < 0:
                return []

            # 2. walk backwards to find the cutoff sequence that protects
            #    the last ``protect_turns`` user turns
            cutoff_seq = await self._find_cutoff_seq(
                db, session_id, max_seq, protect_turns
            )

            # 3. pick a compaction whose boundary falls strictly before the
            #    protected region
            cursor = await db.execute(
                "SELECT boundary_seq, summary FROM compactions "
                "WHERE session_id = ? AND boundary_seq < ? "
                "ORDER BY boundary_seq DESC LIMIT 1",
                (session_id, cutoff_seq),
            )
            comp_row = await cursor.fetchone()

            if comp_row is None:
                return await self._load_all_messages(db, session_id)

            boundary_seq = comp_row["boundary_seq"]
            summary = comp_row["summary"]
            recent = await self._load_messages_after(db, session_id, boundary_seq)

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

    async def _find_cutoff_seq(
        self,
        db: aiosqlite.Connection,
        session_id: str,
        max_seq: int,
        protect_turns: int,
    ) -> int:
        """Find the sequence number after *protect_turns* user turns.

        Uses the ``role`` column (an exact classification written by
        ``_infer_role``: ``'user'`` is equivalent to ``_is_turn_start``) so
        the lookup is a single indexed query instead of scanning and
        decoding every message.

        Returns ``max_seq + 1`` when *protect_turns* is 0 (no protection —
        every compaction is eligible). Returns 0 when there aren't enough
        turns to satisfy *protect_turns* (fall back to full history).
        """
        if protect_turns <= 0:
            return max_seq + 1

        cursor = await db.execute(
            "SELECT message_seq FROM messages "
            "WHERE session_id = ? AND message_seq <= ? AND role = 'user' "
            "ORDER BY message_seq DESC LIMIT 1 OFFSET ?",
            (session_id, max_seq, protect_turns - 1),
        )
        row = await cursor.fetchone()
        if row is None:
            return 0
        return int(row["message_seq"])

    async def get_max_message_seq(self, session_id: str) -> int:
        """Return the current maximum ``message_seq`` for the session, or ``-1`` if empty."""
        async with self._connect() as db:
            cursor = await db.execute(
                "SELECT COALESCE(MAX(message_seq), -1) FROM messages WHERE session_id = ?",
                (session_id,),
            )
            row = await cursor.fetchone()
            return int(cast(aiosqlite.Row, row)[0])

    async def _load_all_messages(
        self, db: aiosqlite.Connection, session_id: str
    ) -> list[Any]:
        cursor = await db.execute(
            "SELECT content FROM messages WHERE session_id = ? ORDER BY message_seq",
            (session_id,),
        )
        rows = await cursor.fetchall()
        return [_MessageAdapter.validate_json(row["content"].encode()) for row in rows]

    async def _load_messages_after(
        self, db: aiosqlite.Connection, session_id: str, boundary_seq: int
    ) -> list[Any]:
        cursor = await db.execute(
            "SELECT content FROM messages WHERE session_id = ? AND message_seq > ? ORDER BY message_seq",
            (session_id, boundary_seq),
        )
        rows = await cursor.fetchall()
        return [_MessageAdapter.validate_json(row["content"].encode()) for row in rows]

    async def save_messages(
        self, session_id: str, messages: list[Any]
    ) -> None:
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
            seq = int(cast(aiosqlite.Row, row)["next_seq"])
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

    async def save_system_prompt(
        self, session_id: str, system_prompt: str
    ) -> None:
        """Persist the system prompt for the session."""
        async with self._connect() as db:
            await db.execute(
                "INSERT INTO system_prompts (session_id, system_prompt) VALUES (?, ?)",
                (session_id, system_prompt),
            )
            await db.commit()

    async def load_system_prompt(
        self, session_id: str
    ) -> str | None:
        """Load the most recent system prompt for the session, if any."""
        async with self._connect() as db:
            cursor = await db.execute(
                "SELECT system_prompt FROM system_prompts "
                "WHERE session_id = ? ORDER BY id DESC LIMIT 1",
                (session_id,),
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            return cast(str, row["system_prompt"])
