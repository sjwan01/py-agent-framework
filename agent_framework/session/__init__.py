"""Session adapters — one ABC + three concrete implementations."""
from agent_framework.session._single_turn import SingleTurnSessionManager
from agent_framework.session._local import LocalSessionManager
from agent_framework.session._postgres import PostgresSessionManager
from agent_framework.types import SessionManager  # re-export for convenience

__all__ = [
    "SessionManager",
    "SingleTurnSessionManager",
    "LocalSessionManager",
    "PostgresSessionManager",
]
