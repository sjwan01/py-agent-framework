"""PostgreSQL session 实现。"""
from __future__ import annotations

from uuid import uuid4

import json
from psycopg_pool import AsyncConnectionPool

from agent_framework.session._shared import _infer_role, _MessageAdapter
from agent_framework.types import SessionManager

# ── PostgreSQL 建表语句 ──────────────────────────────────────────────
# sessions 表：每个 session 一行，metadata 用 JSONB。
# messages 表：每条消息一行。
#   - entry_id 用 gen_random_uuid() 自动生成。
#   - content 用 JSONB，psycopg 在 Python dict 和 JSONB 之间自动转换。
#   - 索引按 (session_id, turn_index)。
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

    async def load_history(self, session_id: str) -> list:
        """按 turn_index 升序加载指定 session 的所有普通消息。

        psycopg 可能把 JSONB 返回为 Python dict 或字符串，由 _deserialize_pg_message 统一处理。

        TODO: 当前 load_history 只加载原 messages，没有把 compaction
        摘要重新注入为 system prompt。这意味着长对话即使做过 compaction，
        加载历史时仍然是原始（可能已超窗口）的消息列表。期望行为应该是
        加载 baseline + compaction summary + 未被压缩的 recent messages。
        """
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cursor = await conn.execute(
                "SELECT content FROM messages WHERE session_id = %s AND role != 'compaction' ORDER BY turn_index",
                (session_id,),
            )
            rows = await cursor.fetchall()
        return [_deserialize_pg_message(row[0]) for row in rows]

    async def save_messages(self, session_id: str, messages: list) -> None:
        """把这轮新增的消息批量写入 messages 表。

        turn_index 从当前最大值 +1 开始，同批消息按顺序递增。
        """
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
        """写入 compaction 摘要记录（turn_index=-1，role='compaction'）。"""
        pool = await self._get_pool()
        async with pool.connection() as conn:
            await conn.execute(
                "INSERT INTO messages (session_id, turn_index, role, content) VALUES (%s, -1, %s, %s)",
                (session_id, "compaction", json.dumps({
                    "type": "compaction",
                    "summary": summary,
                    "boundary_entry_id": boundary_entry_id,
                })),
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
