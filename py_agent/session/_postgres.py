"""PostgreSQL session implementation."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

from psycopg_pool import AsyncConnectionPool
from pydantic_ai.messages import ModelRequest, UserPromptPart

from py_agent.session._shared import _infer_role, _is_turn_start, _MessageAdapter
from py_agent.types import SessionManager


# PostgreSQL schema
# sessions: one row per session, metadata stored as JSONB.
# messages: one row per message.
#   - entry_id is auto-generated with gen_random_uuid().
#   - content uses JSONB; psycopg converts between Python dict and JSONB.
#   - indexed by (session_id, message_seq).
# compactions: one row per compaction, with boundary_seq marking the summarized range.
PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT now(),
    metadata JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS messages (
    entry_id TEXT PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    message_seq INT NOT NULL,
    role TEXT NOT NULL,
    content JSONB NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_messages_session_seq_pg
    ON messages(session_id, message_seq);

CREATE TABLE IF NOT EXISTS compactions (
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    boundary_seq INTEGER NOT NULL,
    summary TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);
"""


class PostgresSessionManager(SessionManager):
    """PostgreSQL-backed multi-turn session manager."""

    def __init__(
        self,
        *,
        # PostgreSQL connection URL, e.g. postgresql://user:pass@host/db
        pg_url: str,
        # minimum idle connections in the pool
        pool_size: int = 5,
        # maximum extra connections beyond pool_size (total = pool_size + max_overflow)
        max_overflow: int = 10,
    ):
        self._pg_url = pg_url
        self._pool_size = pool_size
        self._max_overflow = max_overflow
        # lazy pool initialization; first _get_pool() call creates the real connections
        self._pool: AsyncConnectionPool | None = None

    async def _get_pool(self) -> AsyncConnectionPool:
        """Get (or lazily create) the connection pool.

        Creates the pool, opens it, and ensures the schema exists on first call.
        """
        if self._pool is None:
            self._pool = AsyncConnectionPool(
                self._pg_url,
                # min_size: keep warm connections to avoid cold-start latency
                min_size=self._pool_size,
                # max_size: hard cap; connections above pool_size are recycled after use
                max_size=self._pool_size + self._max_overflow,
                # open=False defers actual connection establishment until open() is called
                open=False,
            )
            await self._pool.open()
            # ensure schema exists before serving requests
            async with self._pool.connection() as conn:
                await conn.execute(PG_SCHEMA)
        return self._pool

    # SessionManager interface implementation

    async def create_session(self, *, metadata: dict | None = None) -> str:
        """Insert a row into ``sessions`` and return the new ``session_id``."""
        pool = await self._get_pool()
        sid = str(uuid4())
        async with pool.connection() as conn:
            await conn.execute(
                "INSERT INTO sessions (session_id, metadata) VALUES (%s, %s)",
                (sid, json.dumps(metadata or {})),
            )
        return sid

    async def ensure_session(self, session_id: str, *, metadata: dict | None = None) -> str:
        """Ensure the session row exists, creating it if necessary."""
        pool = await self._get_pool()
        async with pool.connection() as conn:
            await conn.execute(
                "INSERT INTO sessions (session_id, metadata) VALUES (%s, %s) ON CONFLICT (session_id) DO NOTHING",
                (session_id, json.dumps(metadata or {})),
            )
        return session_id

    async def load_history(self, session_id: str, *, protect_turns: int = 0) -> list:
        """Load messages for the session.

        Checks compaction first: if no compaction record exists, loads all
        messages. If a record exists and at least ``protect_turns`` user turns
        have passed the boundary, loads only messages with ``seq > boundary``
        and prepends a summary.
        """
        pool = await self._get_pool()
        async with pool.connection() as conn:
            # 1. fetch the latest compaction
            cursor = await conn.execute(
                "SELECT boundary_seq, summary FROM compactions WHERE session_id = %s ORDER BY boundary_seq DESC LIMIT 1",
                (session_id,),
            )
            comp_row = await cursor.fetchone()

            if comp_row is None:
                return await self._load_all_messages(conn, session_id)

            boundary_seq = comp_row[0]
            summary = comp_row[1]

            # 2. load only messages after the boundary and count user turns
            recent = await self._load_messages_after(conn, session_id, boundary_seq)
            turns_after = sum(1 for m in recent if _is_turn_start(m))

            if turns_after < protect_turns:
                return await self._load_all_messages(conn, session_id)

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
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cursor = await conn.execute(
                "SELECT COALESCE(MAX(message_seq), -1) FROM messages WHERE session_id = %s",
                (session_id,),
            )
            return (await cursor.fetchone())[0]

    async def _load_all_messages(self, conn, session_id: str) -> list:
        cursor = await conn.execute(
            "SELECT content FROM messages WHERE session_id = %s ORDER BY message_seq",
            (session_id,),
        )
        rows = await cursor.fetchall()
        return [_deserialize_pg_message(row[0]) for row in rows]

    async def _load_messages_after(self, conn, session_id: str, boundary_seq: int) -> list:
        cursor = await conn.execute(
            "SELECT content FROM messages WHERE session_id = %s AND message_seq > %s ORDER BY message_seq",
            (session_id, boundary_seq),
        )
        rows = await cursor.fetchall()
        return [_deserialize_pg_message(row[0]) for row in rows]

    async def save_messages(self, session_id: str, messages: list) -> None:
        """Append this turn's delta messages to the ``messages`` table.

        ``message_seq`` starts at the current maximum + 1 and increments for
        each message in the batch.
        """
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cursor = await conn.execute(
                "SELECT COALESCE(MAX(message_seq), -1) + 1 FROM messages WHERE session_id = %s",
                (session_id,),
            )
            seq = (await cursor.fetchone())[0]
            for msg in messages:
                role = _infer_role(msg)
                content = _MessageAdapter.dump_json(msg).decode()
                await conn.execute(
                    "INSERT INTO messages (session_id, message_seq, role, content) VALUES (%s, %s, %s, %s)",
                    (session_id, seq, role, content),
                )
                seq += 1

    async def apply_compaction(self, session_id: str, summary: str, boundary_seq: int) -> None:
        """Write a compaction record into the separate ``compactions`` table."""
        pool = await self._get_pool()
        async with pool.connection() as conn:
            await conn.execute(
                "INSERT INTO compactions (session_id, boundary_seq, summary) VALUES (%s, %s, %s)",
                (session_id, boundary_seq, summary),
            )

    async def close(self) -> None:
        """Close the connection pool and release all connections."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None


# JSONB deserialization helper
# psycopg may return a Python dict for JSONB columns (when auto-deserialization is enabled)
# or a raw JSON string (depending on configuration). This helper handles both cases.

def _deserialize_pg_message(data) -> object:
    """Deserialize a JSONB value (dict or string) from psycopg into a message object."""
    if isinstance(data, dict):
        return _MessageAdapter.validate_python(data)
    return _MessageAdapter.validate_json(data.encode())
