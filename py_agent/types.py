"""Core types — ABCs, Protocols, and enums.

Zero implementation. Pure interface definitions.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from enum import StrEnum
from typing import Any, Protocol

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage

# SessionManager (external seam)

class SessionManager(ABC):
    """Interface that all session backends must implement.

    Implement this to build a custom persistence backend; the three built-in
    backends (`SingleTurnSessionManager`, `LocalSessionManager`,
    `PostgresSessionManager`) are working references. Messages are Pydantic AI
    ``ModelMessage`` objects (use ``TypeAdapter(ModelMessage)`` to
    serialize/deserialize them). ``AgentRunner`` calls these methods per turn;
    the contract for each is documented below.

    Attributes:
        create_session: Create a new session and return its id.
        load_history: Load messages for a session, respecting ``protect_turns`` for compaction-aware boundaries.
        save_messages: Persist a batch of messages for a session.
        apply_compaction: Store a compaction summary at the given ``boundary_seq``.
        get_max_message_seq: Return the highest message sequence number, or ``-1`` if empty.
        ensure_session: Create the session row if it does not already exist.
        save_system_prompt: Persist the current system prompt for a session.
        load_system_prompt: Load the most recently saved system prompt, if any.
    """

    @abstractmethod
    async def create_session(
        self, *, metadata: dict[str, Any] | None = None
    ) -> str:
        """Create a new session and return its identifier.

        Args:
            metadata: Arbitrary session metadata (e.g. language, tenant),
                stored for later inspection. Defaults to ``None``.

        Returns:
            A unique session id; pass it to ``AgentRunner.run(session_id=...)``
            to continue the session.
        """

    @abstractmethod
    async def load_history(
        self, session_id: str, *, protect_turns: int = 0
    ) -> list[ModelMessage]:
        """Load the session's messages for the next turn.

        A compaction summary (if any) replaces the messages it covers, so
        the returned list is the effective history the agent sees. Messages
        are ordered oldest to newest.

        Args:
            session_id: The session to load.
            protect_turns: Number of most recent user turns that must stay
                intact (not covered by a compaction summary). ``0`` means
                the latest compaction applies unconditionally.

        Returns:
            The effective message history.
        """

    @abstractmethod
    async def save_messages(
        self, session_id: str, messages: list[ModelMessage]
    ) -> None:
        """Append a batch of messages (this turn's delta) to the session.

        ``message_seq`` values continue from the current maximum + 1. The
        ``role`` column (see ``MessageRole``) is derived by the backend from
        each message and is what ``load_history``'s ``protect_turns`` relies
        on.

        Args:
            session_id: The session to append to.
            messages: The messages to persist, in order.
        """

    @abstractmethod
    async def apply_compaction(
        self, session_id: str, summary: str, boundary_seq: int
    ) -> None:
        """Record a compaction summary for the session.

        Write-only: this never deletes messages. ``load_history`` replaces
        messages at or before ``boundary_seq`` with the summary (only when
        the boundary falls before the protected region).

        Args:
            session_id: The session being compacted.
            summary: The generated summary text.
            boundary_seq: The message sequence boundary — messages up to and
                including this seq are summarized/overwritten on load.
        """

    @abstractmethod
    async def get_max_message_seq(self, session_id: str) -> int:
        """Return the highest ``message_seq`` for the session, or ``-1`` if empty.

        This is the raw maximum, unaffected by the compactions table — it
        serves as the boundary for the next compaction.
        """

    @abstractmethod
    async def ensure_session(
        self, session_id: str, *, metadata: dict[str, Any] | None = None
    ) -> str:
        """Ensure the session row exists, creating it if necessary.

        Called when resuming an existing session id.

        Returns:
            The session id (as passed in).
        """

    @abstractmethod
    async def save_system_prompt(
        self, session_id: str, system_prompt: str
    ) -> None:
        """Persist the session's system prompt (latest write wins).

        Stored prompts let a prompt-less ``AgentRunner`` reconnect to a
        session without re-supplying the prompt.
        """

    @abstractmethod
    async def load_system_prompt(
        self, session_id: str
    ) -> str | None:
        """Load the most recently saved system prompt for the session, if any.

        Returns:
            The stored prompt, or ``None`` if none was saved.
        """


# Extension Protocol

class Extension(Protocol):
    """Protocol for user-provided extensions that hook into the agent lifecycle.

    Extensions own the lifecycle: they observe and intercept events. They may
    also register SDK capabilities (the ``register_capabilities`` hook is
    optional) — tools and skills are declared on ``AgentRunner`` itself, not
    through extensions.

    A minimal lifecycle extension (e.g. auditing tool calls)::

        class Audit(Extension):
            async def on_agent_runner_event(self, event, data):
                if event == AgentRunnerEvent.TOOL_CALL:
                    audit_log(data["tool_name"], data["args"])
                return None   # no modification

    To intercept (not just observe), return a dict that patches ``data``:

    - ``TOOL_CALL``: ``{"block": true, "reason": ...}`` blocks the tool;
      ``{"args": {...}}`` rewrites its arguments before execution.
    - ``TOOL_RESULT``: ``{"content": ...}`` rewrites the tool's return value.
    - ``BEFORE_AGENT_RUN`` / ``SESSION_SAVE``: ``{"messages": ...}`` /
      ``{"delta_messages": ...}`` replace what is sent / persisted.
    - ``COMPACTION_TRIGGER``: ``{"cancel": true}`` vetoes compaction.

    Returning ``None`` leaves the event unchanged. See ``AgentRunnerEvent``
    for every event and its payload. Use ``register_capabilities`` only to
    add SDK capabilities (e.g. ``Skills``, ``PrefixTools``) — declare tools
    and skills on ``AgentRunner`` instead.

    Attributes:
        register_capabilities: Optional; return Pydantic AI ``AbstractCapability``
            instances (e.g. ``Skills``, ``PrefixTools``) the agent should use.
        on_agent_runner_event: Called during a run; receive ``AgentRunnerEvent`` values and return a dict to modify event data.
        on_agent_runner_event_stream: Optional async generator for streaming extensions.
    """

    async def register_capabilities(self) -> list[AbstractCapability[Any]]:
        """Optional hook: return Pydantic AI ``AbstractCapability`` instances.

        Extensions that only observe events can omit this — the default
        contributes nothing.
        """
        return []

    async def on_agent_runner_event(
        self, event: str, data: dict[str, Any]
    ) -> dict[str, Any] | None: ...
    async def on_agent_runner_event_stream(
        self, event: str, data: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        """Optional async generator for streaming extensions.

        If the extension does not override this, ``run()`` and ``run_stream()``
        ignore it.
        """
        if False:  # pragma: no cover - unreachable protocol-generator default
            yield {}  # type: ignore[unreachable]


ToolsetFailureHandler = Callable[
    [str, Exception], dict[str, Any] | None
]
"""Handler for a toolset whose connection or catalog failed to load.

- Return a ``dict`` to substitute this server's tools for this run (an empty
  dict drops them — partial degradation).
- Return ``None`` for the default behavior (warn + drop).
- Raise to fail the run (e.g. for a critical server).

The internal logic is free (retry, alert, decide); the signature is fixed:
``(toolset_id, exception)``. Any exception raised by this handler — a
mistyped signature (``TypeError``) included — propagates to the caller and
fails the run; write it correctly, it is intentionally not swallowed.
"""

# Data enums

class MessageRole(StrEnum):
    """Message role values stored in the database schema.

    Attributes:
        USER: The message came from the user.
        ASSISTANT: The message came from the model.
        TOOL: The message is a tool return value.
        UNKNOWN: The role could not be determined.
    """
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    UNKNOWN = "unknown"


# Event enums
#
# AgentRunnerEvent fires during a single run() / run_stream() execution.
#
#   run() / run_stream()
#   ├── SESSION_START
#   ├── load_history
#   ├── CONTEXT_PREPARE          # read-only: Extension observes context preparation output
#   ├── BEFORE_AGENT_RUN         # writable: Extension's only entry point to modify messages
#   ├── AGENT_START              # read-only: prompt + final messages entering the Agent
#   │   ├── (Pydantic AI tool loop)
#   │   │   ├── TOOL_START       # log/UI
#   │   │   ├── TOOL_CALL        # intercept: block / modify args
#   │   │   ├── TOOL_RESULT      # intercept: modify result
#   │   │   └── TOOL_END         # log/UI
#   │   └── TOKEN_STREAM         # per-token chunk during run_stream
#   ├── AGENT_END                # read-only: output + usage
#   ├── SESSION_SAVE             # writable: modify delta_messages before persistence
#   ├── COMPACTION_TRIGGER       # flagged by context preparation; cancellable
#   │   └── COMPACTION_SCHEDULED
#   └── SESSION_END
#

class AgentRunnerEvent(StrEnum):
    """Events fired during a single ``run()`` / ``run_stream()`` execution.

    Extensions receive these in ``on_agent_runner_event(event, data)``.
    Each event's ``data`` payload (all values are plain dicts):

    - ``SESSION_START`` / ``SESSION_END``: ``{"session_id": str}``
    - ``CONTEXT_PREPARE``: ``{"session_id": str, "messages": list, "needs_compaction": bool}``
    - ``BEFORE_AGENT_RUN``: ``{"session_id": str, "messages": list}`` — writable:
      replace ``messages`` to inject. Newly injected messages are persisted.
    - ``AGENT_START``: ``{"session_id": str, "prompt": str, "messages": list}``
    - ``AGENT_END`` / ``AFTER_AGENT_RUN``: ``{"session_id": str, "output": str, "usage": ...}``
    - ``TOKEN_STREAM``: ``{"session_id": str, "data": {"chunk": str}}`` (run_stream only)
    - ``TOOL_START`` / ``TOOL_END``: ``{"session_id": str, "name": str, "data": {"args" | "result": ...}}``
    - ``TOOL_CALL``: ``{"session_id": str, "tool_name": str, "tool_call_id": str, "args": dict}``
      — writable: ``{"block": true}`` blocks, ``{"args": ...}`` rewrites arguments.
    - ``TOOL_RESULT``: ``{"session_id": str, "tool_name": str, "tool_call_id": str, "content": ..., "is_error": bool}``
      — writable: ``{"content": ...}`` rewrites the result.
    - ``SESSION_SAVE``: ``{"session_id": str, "delta_messages": list}`` — writable:
      replace ``delta_messages`` before persistence.
    - ``COMPACTION_TRIGGER``: ``{"session_id": str}`` — return ``{"cancel": true}`` to veto.
    - ``COMPACTION_SCHEDULED``: ``{"session_id": str, "cancelled": bool}``

    Attributes:
        SESSION_START: A new session was created or reused.
        SESSION_END: The session is finished.
        CONTEXT_PREPARE: Context preparation has finished (read-only).
        BEFORE_AGENT_RUN: Last chance for extensions to modify messages.
        AGENT_START: The final prompt and messages are entering the model.
        AGENT_END: The model finished generating a response.
        TOKEN_STREAM: A single token chunk during streaming.
        TOOL_START: A tool is about to be invoked.
        TOOL_END: A tool finished executing.
        TOOL_CALL: Intercept point — block or modify tool arguments.
        TOOL_RESULT: Intercept point — modify the tool's return value.
        AFTER_AGENT_RUN: The agent finished but messages are not yet saved.
        SESSION_SAVE: Messages are about to be persisted — modify delta.
        COMPACTION_TRIGGER: Compaction was flagged; extensions may cancel.
        COMPACTION_SCHEDULED: Compaction was scheduled (or cancelled by an
            extension) — fired at scheduling time, before the background
            task runs. There is no completion event.
    """
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    CONTEXT_PREPARE = "context_prepare"
    BEFORE_AGENT_RUN = "before_agent_run"
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    TOKEN_STREAM = "token_stream"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    AFTER_AGENT_RUN = "after_agent_run"
    SESSION_SAVE = "session_save"
    COMPACTION_TRIGGER = "compaction_trigger"
    COMPACTION_SCHEDULED = "compaction_scheduled"



__all__ = [
    "SessionManager",
    "Extension",
    "MessageRole",
    "AgentRunnerEvent",
    "ToolsetFailureHandler",
]
