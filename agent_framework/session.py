"""Session adapters — SingleTurn (no-op) and Local (SQLite).

提供三种 SessionManager 实现，都实现了 ``agent_framework.types.SessionManager`` 接口：
- SingleTurnSessionManager：不持久化，每次都是全新会话。
- LocalSessionManager：基于 SQLite 的多轮持久化。
- PostgresSessionManager：基于 PostgreSQL 的多轮持久化（在 pg_session.py）。
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from uuid import uuid4

import aiosqlite
import ujson
from pydantic import TypeAdapter
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse

from agent_framework.types import SessionManager

# ── SQLite 建表语句 ──────────────────────────────────────────────────
# sessions 表：每个 session 一行，记录创建时间和自定义元数据。
# messages 表：每条消息一行，按 (session_id, turn_index) 索引。
#   - entry_id：消息唯一标识，UUID。
#   - turn_index：该消息属于第几轮对话（从 0 开始递增）。
#   - role：消息角色，user / assistant / tool / compaction。
#   - content：消息的 JSON 字符串（由 _MessageAdapter 序列化）。
#   - compaction 记录用 turn_index = -1，与普通消息区分开。
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

# ── 单消息序列化适配器 ───────────────────────────────────────────────
# DB 按“每行一条消息”存储，不是整个消息列表，所以用 TypeAdapter(ModelMessage)
# 而不是显式导出的 ModelMessagesTypeAdapter（那个是 list）。
# dump_json / validate_json 走 Pydantic 的 schema，保证与 SDK 版本一致。
_MessageAdapter: TypeAdapter[ModelMessage] = TypeAdapter(ModelMessage)


# ═══════════════════════════════════════════════════════════════════════
# SingleTurnSessionManager：单轮、无持久化
# ═══════════════════════════════════════════════════════════════════════

class SingleTurnSessionManager(SessionManager):
    """不持久化历史，每次 run() 都是全新会话。

    load_history 永远返回空列表，save_messages 是 no-op。
    """

    async def create_session(self, *, metadata: dict | None = None) -> str:
        # 生成一个 UUID 作为 session_id，但不落盘。
        return str(uuid4())

    async def load_history(self, session_id: str) -> list:
        # 没有历史，永远返回空。
        return []

    async def save_messages(self, session_id: str, messages: list) -> None:
        # 不持久化，什么都不做。
        pass

    async def apply_compaction(self, session_id: str, summary: str, boundary_entry_id: str) -> None:
        # 单轮模式不支持 compaction。
        raise NotImplementedError("SingleTurnSessionManager does not support compaction")


# ═══════════════════════════════════════════════════════════════════════
# LocalSessionManager：基于 SQLite 的多轮持久化
# ═══════════════════════════════════════════════════════════════════════

class LocalSessionManager(SessionManager):
    """SQLite 持久化，适合本地开发和测试。

    每个 session 存在 ``sessions`` 表的一行，每条消息存在 ``messages`` 表的一行。
    消息按 turn_index 排序加载，compaction 记录不会被加载进历史。
    """

    def __init__(self, *, db_path: str):
        # SQLite 文件路径。
        self._db_path = db_path
        # 避免每次连接都重新执行建表语句。
        self._schema_initialized = False

    @asynccontextmanager
    async def _connect(self):
        """获取一个 SQLite 连接的生命周期上下文管理器。

        首次连接时自动建表。连接关闭后不想被外部持有。
        """
        db = await aiosqlite.connect(self._db_path)
        # 让查询结果可以通过列名访问，例如 row["content"]。
        db.row_factory = aiosqlite.Row
        if not self._schema_initialized:
            await db.executescript(SCHEMA)
            await db.commit()
            self._schema_initialized = True
        try:
            yield db
        finally:
            await db.close()

    # ── SessionManager 接口实现 ─────────────────────────────────

    async def create_session(self, *, metadata: dict | None = None) -> str:
        """在 sessions 表中插入一行，返回新 session_id。"""
        sid = str(uuid4())
        async with self._connect() as db:
            await db.execute(
                "INSERT INTO sessions (session_id, metadata) VALUES (?, ?)",
                (sid, ujson.dumps(metadata or {})),
            )
            await db.commit()
        return sid

    async def load_history(self, session_id: str) -> list:
        """按 turn_index 升序加载指定 session 的所有普通消息。

        role='compaction' 的行被排除，因为这些是压缩摘要元数据，
        不参与 Agent 的消息历史。
        """
        async with self._connect() as db:
            cursor = await db.execute(
                "SELECT content FROM messages WHERE session_id = ? AND role != 'compaction' ORDER BY turn_index",
                (session_id,),
            )
            rows = await cursor.fetchall()
            # content 列存的是 JSON 字符串，通过 TypeAdapter 反序列化。
            return [_MessageAdapter.validate_json(row["content"].encode()) for row in rows]

    async def save_messages(self, session_id: str, messages: list) -> None:
        """把一批新消息（本轮 delta）追加到 messages 表。

        每条消息的 turn_index 从当前最大值 +1 开始，同一条消息的不同 role
        （user 输入、tool 返回、assistant 回复）都共享同一个 turn_index。
        """
        async with self._connect() as db:
            # 算出本 session 当前最大的 turn_index，新消息从下一个 turn 开始写。
            cursor = await db.execute(
                "SELECT COALESCE(MAX(turn_index), -1) + 1 AS next_turn FROM messages WHERE session_id = ?",
                (session_id,),
            )
            row = await cursor.fetchone()
            turn = row["next_turn"]
            for msg in messages:
                # 推断消息角色，写入 role 列（用于后续过滤和可读性）。
                role = _infer_role(msg)
                # 用 Pydantic TypeAdapter 把消息序列化为 JSON 字符串。
                content = _MessageAdapter.dump_json(msg).decode()
                await db.execute(
                    "INSERT INTO messages (entry_id, session_id, turn_index, role, content) VALUES (?, ?, ?, ?, ?)",
                    (str(uuid4()), session_id, turn, role, content),
                )
                turn += 1
            await db.commit()

    async def apply_compaction(self, session_id: str, summary: str, boundary_entry_id: str) -> None:
        """把 compaction 摘要存为一条特殊记录。

        这条记录的 role='compaction'，turn_index=-1，不会被 load_history 加载，
        只用于审计/诊断，不参与 Agent 的上下文。
        """
        async with self._connect() as db:
            await db.execute(
                "INSERT INTO messages (entry_id, session_id, turn_index, role, content) VALUES (?, ?, -1, ?, ?)",
                (str(uuid4()), session_id, "compaction", ujson.dumps({
                    "type": "compaction",
                    "summary": summary,
                    "boundary_entry_id": boundary_entry_id,
                })),
            )
            await db.commit()


# ── 角色推断 ──────────────────────────────────────────────────────────
# 根据消息类型和 parts 内容判断消息的角色，写入 DB 的 role 列。
# 这只是标记用途，不影响反序列化后的消息对象本身。

def _infer_role(msg) -> str:
    # ModelRequest：如果第一个 part 有 tool_name 属性，说明是 tool 返回消息，
    # 否则是用户消息。
    if isinstance(msg, ModelRequest):
        parts = msg.parts
        if parts and hasattr(parts[0], "tool_name"):
            return "tool"
        return "user"
    # ModelResponse：assistant 的回复。
    if isinstance(msg, ModelResponse):
        return "assistant"
    # 兜底。
    return "unknown"
