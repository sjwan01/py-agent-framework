"""SQLite session 实现。"""
from __future__ import annotations

from contextlib import asynccontextmanager
from uuid import uuid4

import aiosqlite
import json

from py_agent.session._shared import _infer_role, _is_turn_start, _MessageAdapter
from py_agent.types import SessionManager
from pydantic_ai.messages import ModelRequest, UserPromptPart
from datetime import datetime, timezone

# ── SQLite 建表语句 ──────────────────────────────────────────────────
# sessions 表：每个 session 一行，记录创建时间和自定义元数据。
# messages 表：每条消息一行，按 (session_id, message_seq) 索引。
#   - entry_id：消息唯一标识，UUID。
#   - message_seq：消息顺序号（从 0 开始递增）。
#   - role：消息角色，user / assistant / tool。
#   - content：消息的 JSON 字符串（由 _MessageAdapter 序列化）。
# compactions 表：每次压缩一条记录，用 boundary_seq 标记压缩覆盖范围。
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
    """SQLite 持久化，适合本地开发和测试。

    每个 session 存在 ``sessions`` 表的一行，每条消息存在 ``messages`` 表的一行。
    消息按 message_seq 排序加载，compaction 记录存在独立的 ``compactions`` 表。
    """

    def __init__(self, *, db_path: str):
        # SQLite 文件路径。
        self._db_path = db_path
        # 避免每次连接都重新执行建表语句。
        self._schema_initialized = False

    @asynccontextmanager
    async def _connect(self):
        """获取一个 SQLite 连接，首次连接时自动建表。"""
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
                (sid, json.dumps(metadata or {})),
            )
            await db.commit()
        return sid

    async def ensure_session(self, session_id: str, *, metadata: dict | None = None) -> str:
        """确保 session 行存在：不存在则创建，存在则无操作。"""
        async with self._connect() as db:
            await db.execute(
                "INSERT OR IGNORE INTO sessions (session_id, metadata) VALUES (?, ?)",
                (session_id, json.dumps(metadata or {})),
            )
            await db.commit()
        return session_id

    async def load_history(self, session_id: str, *, protect_turns: int = 0) -> list:
        """加载 session 消息。

        先查 compaction：若无记录则加载全部；若有且 boundary 后攒够 protect_turns
        个 user turn，则只加载 seq > boundary 的消息并前置摘要。
        """
        async with self._connect() as db:
            # 1. 先查最新 compaction
            cursor = await db.execute(
                "SELECT boundary_seq, summary FROM compactions WHERE session_id = ? ORDER BY boundary_seq DESC LIMIT 1",
                (session_id,),
            )
            comp_row = await cursor.fetchone()

            if comp_row is None:
                # 无 compaction → 加载全部
                return await self._load_all_messages(db, session_id)

            boundary_seq = comp_row["boundary_seq"]
            summary = comp_row["summary"]

            # 2. 只加载 boundary 之后的消息，数 user turn
            recent = await self._load_messages_after(db, session_id, boundary_seq)
            turns_after = sum(1 for m in recent if _is_turn_start(m))

            if turns_after < protect_turns:
                # 不足 protect_turns → 回退，加载全部
                return await self._load_all_messages(db, session_id)

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
        """把一批新消息（本轮 delta）追加到 messages 表。

        每条消息的 message_seq 从当前最大值 +1 开始。
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
        """写入 compaction 记录到独立的 compactions 表。"""
        async with self._connect() as db:
            await db.execute(
                "INSERT INTO compactions (session_id, boundary_seq, summary) VALUES (?, ?, ?)",
                (session_id, boundary_seq, summary),
            )
            await db.commit()
