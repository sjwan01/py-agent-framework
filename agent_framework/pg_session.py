"""PostgreSQL session adapter.

功能和 session.py 的 LocalSessionManager 完全一样，只是后端换成了 PostgreSQL。
通过 psycopg_pool 管理连接池，content 列使用 JSONB 类型。
"""
from __future__ import annotations

from uuid import uuid4

import ujson
from psycopg_pool import AsyncConnectionPool

# _infer_role 和 _MessageAdapter 复用 session.py 的定义，避免重复。
from agent_framework.session import _infer_role, _MessageAdapter
from agent_framework.types import SessionManager

# ── PostgreSQL 建表语句 ──────────────────────────────────────────────
# sessions 表：每个 session 一行。
#   - metadata 用 JSONB，可以存任意结构化信息。
# messages 表：每条消息一行。
#   - entry_id 用 gen_random_uuid() 自动生成，不依赖应用层 UUID。
#   - content 用 JSONB，psycopg 会自动在 Python dict 和 JSONB 之间转换。
#   - 索引和 SQLite 版一样，按 (session_id, turn_index)。
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
    """PostgreSQL 持久化的多轮会话管理。

    通过 psycopg_pool.AsyncConnectionPool 管理连接，支持连接池大小和溢出配置。
    和 LocalSessionManager 接口完全一致，可以无缝切换。
    """

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
        之后的所有调用共享同一个连接池。
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
                (sid, ujson.dumps(metadata or {})),
            )
        return sid

    async def load_history(self, session_id: str) -> list:
        """按 turn_index 升序加载指定 session 的所有普通消息。

        role='compaction' 被排除（和 SQLite 版逻辑一致）。
        psycopg 可能把 JSONB 返回为 Python dict 或字符串，由 _deserialize_pg_message 统一处理。
        """
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cursor = await conn.execute(
                "SELECT content FROM messages WHERE session_id = %s AND role != 'compaction' ORDER BY turn_index",
                (session_id,),
            )
            rows = await cursor.fetchall()
            # row[0] 是 content 列，psycopg 返回的类型取决于数据：Python dict 或 str。
        return [_deserialize_pg_message(row[0]) for row in rows]

    async def save_messages(self, session_id: str, messages: list) -> None:
        """把这轮新增的消息批量写入 messages 表。

        turn_index 从当前最大值 +1 开始，同批消息按顺序递增。
        content 通过 _MessageAdapter 序列化为 JSON 字符串；psycopg
        会把字符串转成 JSONB 存进去。
        """
        pool = await self._get_pool()
        async with pool.connection() as conn:
            # 找到当前最大的 turn_index，+1 作为本轮起始 turn。
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
        """写入 compaction 摘要记录（turn_index=-1，role='compaction'）。

        这条记录不会被 load_history 加载，只用于审计/诊断。
        """
        pool = await self._get_pool()
        async with pool.connection() as conn:
            await conn.execute(
                "INSERT INTO messages (session_id, turn_index, role, content) VALUES (%s, -1, %s, %s)",
                (session_id, "compaction", ujson.dumps({
                    "type": "compaction",
                    "summary": summary,
                    "boundary_entry_id": boundary_entry_id,
                })),
            )

    # ── 运维 ────────────────────────────────────────────────────

    async def cleanup_stale_sessions(self, timeout_seconds: int | None = None) -> int:
        """删除超过 ``timeout_seconds`` 秒未活跃的 session（默认 1 天）。

        用 PostgreSQL 的 make_interval 函数计算过期时间。
        """
        timeout = timeout_seconds if timeout_seconds is not None else 86400
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cursor = await conn.execute(
                "DELETE FROM sessions WHERE created_at < now() - make_interval(secs => %s)",
                (timeout,),
            )
            return cursor.rowcount

    async def close(self) -> None:
        """关闭连接池，释放所有连接。

        调用后不能再使用这个 SessionManager。
        """
        if self._pool is not None:
            await self._pool.close()
            self._pool = None


# ── JSONB 反序列化辅助 ────────────────────────────────────────────────
# psycopg 连接 JSONB 列时可能返回 Python dict（当连接设置了自动反序列化）
# 或原始 JSON 字符串（取决于 psycopg 配置）。这个函数统一处理两种情况。

def _deserialize_pg_message(data) -> object:
    """把 psycopg 返回的 JSONB 值（可能是 dict 或 str）反序列化为消息对象。"""
    # 如果 psycopg 已经帮我们解析成 dict，直接用 validate_python。
    if isinstance(data, dict):
        return _MessageAdapter.validate_python(data)
    # 否则是 JSON 字符串，用 validate_json 解析。
    return _MessageAdapter.validate_json(data.encode())
