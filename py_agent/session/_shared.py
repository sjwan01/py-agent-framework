"""Session shared utilities — roles, serialization, turn detection."""
from __future__ import annotations

from pydantic import TypeAdapter
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    UserPromptPart,
)

from py_agent.types import MessageRole

# The DB stores one row per message, so use a single-message adapter.
_MessageAdapter: TypeAdapter[ModelMessage] = TypeAdapter(ModelMessage)


def _infer_role(msg: ModelMessage) -> MessageRole:
    """Infer the message role from its type and first part for the DB role column.

    This is only a marker for the DB; it does not affect the deserialized
    message object.
    """
    # ModelRequest with a first part that has tool_name is a tool return message.
    if isinstance(msg, ModelRequest):
        parts = msg.parts
        if parts and hasattr(parts[0], "tool_name"):
            return MessageRole.TOOL
        return MessageRole.USER
    # ModelResponse is an assistant reply.
    if isinstance(msg, ModelResponse):
        return MessageRole.ASSISTANT
    return MessageRole.UNKNOWN  # type: ignore[unreachable]


def _is_turn_start(msg: ModelMessage) -> bool:
    """Return True if ``msg`` marks the start of a new user turn.

    A turn starts when the message is a ``ModelRequest`` whose first part is a
    ``UserPromptPart``. This is roughly equivalent to "the user sent a new message".
    """
    if isinstance(msg, ModelRequest):
        return bool(msg.parts) and isinstance(msg.parts[0], UserPromptPart)
    return False
