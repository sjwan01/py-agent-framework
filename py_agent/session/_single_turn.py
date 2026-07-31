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
        """Generate a fresh UUID session id without persisting it."""
        return str(uuid4())

    async def ensure_session(
        self, session_id: str, *, metadata: dict[str, Any] | None = None
    ) -> str:
        """Return the session id as-is; no session row is needed."""
        return session_id

    async def load_history(
        self, session_id: str, *, protect_turns: int = 0
    ) -> list[Any]:
        """Return an empty history for every session."""
        return []

    async def save_messages(
        self, session_id: str, messages: list[Any]
    ) -> None:
        """Persist nothing; single-turn mode has no storage."""
        pass

    async def apply_compaction(self, session_id: str, summary: str, boundary_seq: int) -> None:
        """Reject compaction; single-turn mode has no storage.

        Raises:
            NotImplementedError: Always, because single-turn mode does not
                support compaction.
        """
        raise NotImplementedError("SingleTurnSessionManager does not support compaction")

    async def get_max_message_seq(self, session_id: str) -> int:
        """Return -1; single-turn mode has no messages."""
        return -1

    async def save_system_prompt(
        self, session_id: str, system_prompt: str
    ) -> None:
        """Persist a system prompt (no-op: single-turn mode does not store)."""
        # single-turn mode does not persist system prompts
        pass

    async def load_system_prompt(
        self, session_id: str
    ) -> str | None:
        """Load the stored system prompt (always None for single-turn mode)."""
        # single-turn mode has no system prompt history
        return None
