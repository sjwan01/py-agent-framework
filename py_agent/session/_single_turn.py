"""单轮、无持久化 session 实现。"""
from __future__ import annotations

from uuid import uuid4

from py_agent.types import SessionManager


class SingleTurnSessionManager(SessionManager):
    """不持久化历史，每次 run() 都是全新会话。

    load_history 永远返回空列表，save_messages 是 no-op。
    """

    async def create_session(self, *, metadata: dict | None = None) -> str:
        # 生成一个 UUID 作为 session_id，但不落盘。
        return str(uuid4())

    async def ensure_session(self, session_id: str, *, metadata: dict | None = None) -> str:
        # 单轮模式不持久化，无需创建 session 行。
        return session_id

    async def load_history(self, session_id: str, *, protect_turns: int = 0) -> list:
        # 没有历史，永远返回空。
        return []

    async def save_messages(self, session_id: str, messages: list) -> None:
        # 不持久化，什么都不做。
        pass

    async def apply_compaction(self, session_id: str, summary: str, boundary_seq: int) -> None:
        # 单轮模式不支持 compaction。
        raise NotImplementedError("SingleTurnSessionManager does not support compaction")

    async def get_max_message_seq(self, session_id: str) -> int:
        return -1
