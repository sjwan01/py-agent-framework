"""Single-turn, non-persistent session implementation."""
from __future__ import annotations

from typing import Any
from uuid import uuid4

from py_agent.types import SessionManager


class SingleTurnSessionManager(SessionManager):
    """No-persistence backend. Every ``run()`` is a fresh session.

    Implements ``SessionManager``. ``load_history`` returns ``[]``,
    ``save_messages`` is a no-op, and ``apply_compaction`` raises
    ``NotImplementedError``.
    """

    async def create_session(
        self, *, metadata: dict[str, Any] | None = None
    ) -> str:
        # generate a UUID as session_id without persisting it
        return str(uuid4())

    async def ensure_session(
        self, session_id: str, *, metadata: dict[str, Any] | None = None
    ) -> str:
        # single-turn mode does not persist, so no session row is needed
        return session_id

    async def load_history(
        self, session_id: str, *, protect_turns: int = 0
    ) -> list[Any]:
        # no history, always return empty
        return []

    async def save_messages(
        self, session_id: str, messages: list[Any]
    ) -> None:
        # no persistence, do nothing
        pass

    async def apply_compaction(self, session_id: str, summary: str, boundary_seq: int) -> None:
        # single-turn mode does not support compaction
        raise NotImplementedError("SingleTurnSessionManager does not support compaction")

    async def get_max_message_seq(self, session_id: str) -> int:
        return -1
