"""PostgreSQL session adapter."""
from __future__ import annotations

from uuid import uuid4

import ujson
from psycopg_pool import AsyncConnectionPool

from agent_framework.types import SessionManager
from agent_framework.session import _infer_role

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
    def __init__(self, *, pg_url: str, serializer):
        self._pg_url = pg_url
        self._serializer = serializer
        self._pool: AsyncConnectionPool | None = None

    async def _get_pool(self) -> AsyncConnectionPool:
        if self._pool is None:
            self._pool = AsyncConnectionPool(self._pg_url, open=False)
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
                "SELECT content FROM messages WHERE session_id = %s ORDER BY turn_index",
                (session_id,),
            )
            rows = await cursor.fetchall()
        return [self._serializer.deserialize(row[0]) for row in rows]

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
                content = self._serializer.serialize(msg)
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
                    "type": "compaction", "summary": summary, "boundary": boundary_entry_id,
                })),
            )
