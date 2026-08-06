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
    """Classify the message for the DB role column.

    The role is an exact classification, not a marker:

    - ``USER`` — a ``ModelRequest`` whose first part is a ``UserPromptPart``
      (exactly ``_is_turn_start``);
    - ``TOOL`` — a ``ModelRequest`` whose first part is a tool return;
    - ``UNKNOWN`` — any other ``ModelRequest`` (e.g. system-prompt-led
      messages or future part kinds), never confused with ``USER``;
    - ``ASSISTANT`` — a ``ModelResponse``.

    ``_find_cutoff_seq`` relies on ``role = 'user'`` being equivalent to
    ``_is_turn_start``. Keep this function delegating to ``_is_turn_start``
    and, if the turn-start definition ever changes, rewrite the stored
    ``role`` values as well.

    This classification does not affect the deserialized message object.
    """
    if isinstance(msg, ModelRequest):
        if _is_turn_start(msg):
            return MessageRole.USER
        if msg.parts and hasattr(msg.parts[0], "tool_name"):
            return MessageRole.TOOL
        return MessageRole.UNKNOWN
    # ModelResponse is an assistant reply.
    if isinstance(msg, ModelResponse):
        return MessageRole.ASSISTANT
    # Unreachable under the current ModelMessage union; kept as a defensive
    # catch-all for future message kinds.
    return MessageRole.UNKNOWN  # type: ignore[unreachable]


def _is_turn_start(msg: ModelMessage) -> bool:
    """Return True if ``msg`` marks the start of a new user turn.

    A turn starts when the message is a ``ModelRequest`` whose first part is a
    ``UserPromptPart``. This is roughly equivalent to "the user sent a new message".
    """
    if isinstance(msg, ModelRequest):
        return bool(msg.parts) and isinstance(msg.parts[0], UserPromptPart)
    return False
