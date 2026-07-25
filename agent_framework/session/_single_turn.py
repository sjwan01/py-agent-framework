"""单轮、无持久化 session 实现。"""
from __future__ import annotations

from uuid import uuid4

from agent_framework.types import SessionManager


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
