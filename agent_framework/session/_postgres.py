"""PostgreSQL session 实现。"""
from __future__ import annotations

from uuid import uuid4

import json
from psycopg_pool import AsyncConnectionPool

from agent_framework.session._shared import _infer_role, _is_turn_start, _MessageAdapter
from agent_framework.types import SessionManager
from pydantic_ai.messages import ModelRequest, UserPromptPart
from datetime import datetime, timezone


# ── PostgreSQL 建表语句 ──────────────────────────────────────────────
# sessions 表：每个 session 一行，metadata 用 JSONB。
# messages 表：每条消息一行。
#   - entry_id 用 gen_random_uuid() 自动生成。
#   - content 用 JSONB，psycopg 在 Python dict 和 JSONB 之间自动转换。
#   - 索引按 (session_id, message_seq)。
# compactions 表：每次压缩一条记录，用 boundary_seq 标记压缩覆盖范围。
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
    """PostgreSQL 持久化的多轮会话管理。"""

    def __init__(
        self,
        *,
        # PostgreSQL 连接 URL，例如 postgresql://user:pass@host/db。
        pg_url: str,
        # 连接池最小空闲连接数。
        pool_size: int = 5,
        # 连接池允许的最大额外连接数（总连接数 = pool_size + max_overflow）。
        max_overflow: int = 10,
    ):
        self._pg_url = pg_url
        self._pool_size = pool_size
        self._max_overflow = max_overflow
        # 连接池惰性初始化，首次调用 _get_pool() 时才真正连接。
        self._pool: AsyncConnectionPool | None = None

    async def _get_pool(self) -> AsyncConnectionPool:
        """获取（或惰性创建）连接池。

        首次调用时创建连接池，打开连接，执行建表语句。
        """
        if self._pool is None:
            self._pool = AsyncConnectionPool(
                self._pg_url,
                # min_size：保持的最小连接数，避免冷启动延迟。
                min_size=self._pool_size,
                # max_size：允许的最大连接数，超出 pool_size 的连接用完后会回收。
                max_size=self._pool_size + self._max_overflow,
                # open=False 表示先不建立连接，等 open() 调用时再连。
                open=False,
            )
            await self._pool.open()
            # 确保 schema 存在后再对外提供服务。
            async with self._pool.connection() as conn:
                await conn.execute(PG_SCHEMA)
        return self._pool

    # ── SessionManager 接口实现 ─────────────────────────────────

    async def create_session(self, *, metadata: dict | None = None) -> str:
        """在 sessions 表中插入一行，返回新 session_id。"""
        pool = await self._get_pool()
        sid = str(uuid4())
        async with pool.connection() as conn:
            await conn.execute(
                "INSERT INTO sessions (session_id, metadata) VALUES (%s, %s)",
                (sid, json.dumps(metadata or {})),
            )
        return sid

    async def ensure_session(self, session_id: str, *, metadata: dict | None = None) -> str:
        """确保 session 行存在：不存在则创建，存在则无操作。"""
        pool = await self._get_pool()
        async with pool.connection() as conn:
            await conn.execute(
                "INSERT INTO sessions (session_id, metadata) VALUES (%s, %s) ON CONFLICT (session_id) DO NOTHING",
                (session_id, json.dumps(metadata or {})),
            )
        return session_id

    async def load_history(self, session_id: str, *, protect_turns: int = 0) -> list:
        """加载 session 消息。

        先查 compaction：若无记录则加载全部；若有且 boundary 后攒够 protect_turns
        个 user turn，则只加载 seq > boundary 的消息并前置摘要。
        """
        pool = await self._get_pool()
        async with pool.connection() as conn:
            # 1. 先查最新 compaction
            cursor = await conn.execute(
                "SELECT boundary_seq, summary FROM compactions WHERE session_id = %s ORDER BY boundary_seq DESC LIMIT 1",
                (session_id,),
            )
            comp_row = await cursor.fetchone()

            if comp_row is None:
                return await self._load_all_messages(conn, session_id)

            boundary_seq = comp_row[0]
            summary = comp_row[1]

            # 2. 只加载 boundary 之后的消息，数 user turn
            recent = await self._load_messages_after(conn, session_id, boundary_seq)
            turns_after = sum(1 for m in recent if _is_turn_start(m))

            if turns_after < protect_turns:
                return await self._load_all_messages(conn, session_id)

            # 3. 启用 compaction：摘要 + boundary 后消息
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
        """返回当前 session 的最大 message_seq，无消息时返回 -1。"""
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
        """把这轮新增的消息批量写入 messages 表。

        message_seq 从当前最大值 +1 开始，同批消息按顺序递增。
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
        """写入 compaction 记录到独立的 compactions 表。"""
        pool = await self._get_pool()
        async with pool.connection() as conn:
            await conn.execute(
                "INSERT INTO compactions (session_id, boundary_seq, summary) VALUES (%s, %s, %s)",
                (session_id, boundary_seq, summary),
            )

    async def close(self) -> None:
        """关闭连接池，释放所有连接。"""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None


# ── JSONB 反序列化辅助 ────────────────────────────────────────────────
# psycopg 连接 JSONB 列时可能返回 Python dict（连接设置了自动反序列化）
# 或原始 JSON 字符串（取决于配置）。这个函数统一处理两种情况。

def _deserialize_pg_message(data) -> object:
    """把 psycopg 返回的 JSONB 值（可能是 dict 或 str）反序列化为消息对象。"""
    if isinstance(data, dict):
        return _MessageAdapter.validate_python(data)
    return _MessageAdapter.validate_json(data.encode())
