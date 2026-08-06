"""Session adapters — one ABC + three concrete implementations."""
from __future__ import annotations

from py_agent.session._local import LocalSessionManager
from py_agent.session._postgres import PostgresSessionManager
from py_agent.session._single_turn import SingleTurnSessionManager
from py_agent.types import MessageRole, SessionManager  # re-export for convenience

__all__ = [
    "SessionManager",
    "SingleTurnSessionManager",
    "LocalSessionManager",
    "PostgresSessionManager",
    "MessageRole",
]
