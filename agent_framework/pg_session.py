"""PostgreSQL session adapter."""
from __future__ import annotations

from uuid import uuid4

import ujson
from psycopg_pool import AsyncConnectionPool

from agent_framework.session import _infer_role, _MessageAdapter
from agent_framework.types import SessionManager

PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT now(),
    metadata JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS messages (
    entry_id TEXT PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    turn_index INT NOT NULL,
    role TEXT NOT NULL,
    content JSONB NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_messages_session_turn_pg
    ON messages(session_id, turn_index);
"""


class PostgresSessionManager(SessionManager):
    def __init__(
        self,
        *,
        pg_url: str,
        pool_size: int = 5,
        max_overflow: int = 10,
    ):
        self._pg_url = pg_url
        self._pool_size = pool_size
        self._max_overflow = max_overflow
        self._pool: AsyncConnectionPool | None = None

    async def _get_pool(self) -> AsyncConnectionPool:
        if self._pool is None:
            self._pool = AsyncConnectionPool(
                self._pg_url,
                min_size=self._pool_size,
                max_size=self._pool_size + self._max_overflow,
                open=False,
            )
            await self._pool.open()
            async with self._pool.connection() as conn:
                await conn.execute(PG_SCHEMA)
        return self._pool

    async def create_session(self, *, metadata: dict | None = None) -> str:
        pool = await self._get_pool()
        sid = str(uuid4())
        async with pool.connection() as conn:
            await conn.execute(
                "INSERT INTO sessions (session_id, metadata) VALUES (%s, %s)",
                (sid, ujson.dumps(metadata or {})),
            )
        return sid

    async def load_history(self, session_id: str) -> list:
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cursor = await conn.execute(
                "SELECT content FROM messages WHERE session_id = %s AND role != 'compaction' ORDER BY turn_index",
                (session_id,),
            )
            rows = await cursor.fetchall()
        return [_deserialize_pg_message(row[0]) for row in rows]

    async def save_messages(self, session_id: str, messages: list) -> None:
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cursor = await conn.execute(
                "SELECT COALESCE(MAX(turn_index), -1) + 1 FROM messages WHERE session_id = %s",
                (session_id,),
            )
            turn = (await cursor.fetchone())[0]
            for msg in messages:
                role = _infer_role(msg)
                content = _MessageAdapter.dump_json(msg).decode()
                await conn.execute(
                    "INSERT INTO messages (session_id, turn_index, role, content) VALUES (%s, %s, %s, %s)",
                    (session_id, turn, role, content),
                )
                turn += 1

    async def apply_compaction(self, session_id: str, summary: str, boundary_entry_id: str) -> None:
        pool = await self._get_pool()
        async with pool.connection() as conn:
            await conn.execute(
                "INSERT INTO messages (session_id, turn_index, role, content) VALUES (%s, -1, %s, %s)",
                (session_id, "compaction", ujson.dumps({
                    "type": "compaction", "summary": summary, "boundary_entry_id": boundary_entry_id,
                })),
            )

    async def cleanup_stale_sessions(self, timeout_seconds: int | None = None) -> int:
        """Delete sessions older than ``timeout_seconds`` (defaults to 1 day)."""
        timeout = timeout_seconds if timeout_seconds is not None else 86400
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cursor = await conn.execute(
                "DELETE FROM sessions WHERE created_at < now() - make_interval(secs => %s)",
                (timeout,),
            )
            return cursor.rowcount

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None


def _deserialize_pg_message(data) -> object:
    """Deserialize a JSONB value that may be returned as str or dict."""
    if isinstance(data, dict):
        return _MessageAdapter.validate_python(data)
    return _MessageAdapter.validate_json(data.encode())
