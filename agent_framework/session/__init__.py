"""Session adapters — re-exported from sub-modules."""
from agent_framework.session._single_turn import SingleTurnSessionManager
from agent_framework.session._local import LocalSessionManager
from agent_framework.session._postgres import PostgresSessionManager
from agent_framework.session._shared import _infer_role, _MessageAdapter

__all__ = [
    "SingleTurnSessionManager",
    "LocalSessionManager",
    "PostgresSessionManager",
    "_infer_role",
    "_MessageAdapter",
]
