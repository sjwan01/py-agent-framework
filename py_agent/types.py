"""Core types — ABCs, Protocols, and enums.

Zero implementation. Pure interface definitions.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from enum import StrEnum
from typing import Any, Protocol

# SessionManager (external seam)

class SessionManager(ABC):
    """Interface that all session backends must implement.

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
    ) -> str: ...
    @abstractmethod
    async def load_history(
        self, session_id: str, *, protect_turns: int = 0
    ) -> list[Any]: ...
    @abstractmethod
    async def save_messages(
        self, session_id: str, messages: list[Any]
    ) -> None: ...
    @abstractmethod
    async def apply_compaction(
        self, session_id: str, summary: str, boundary_seq: int
    ) -> None: ...
    @abstractmethod
    async def get_max_message_seq(self, session_id: str) -> int: ...
    @abstractmethod
    async def ensure_session(
        self, session_id: str, *, metadata: dict[str, Any] | None = None
    ) -> str: ...
    @abstractmethod
    async def save_system_prompt(
        self, session_id: str, system_prompt: str
    ) -> None: ...
    @abstractmethod
    async def load_system_prompt(
        self, session_id: str
    ) -> str | None: ...


# Extension Protocol

class Extension(Protocol):
    """Protocol for user-provided extensions that hook into the agent lifecycle.

    Attributes:
        register_tool_sources: Called at initialization; return a list of
            pydantic-ai objects — ``Tool`` or ``AbstractToolset`` instances
            (e.g. ``MCPToolset``) — that the agent should use.
        on_agent_runner_event: Called during a run; receive ``AgentRunnerEvent`` values and return a dict to modify event data.
        on_agent_runner_event_stream: Optional async generator for streaming extensions.
    """

    async def register_tool_sources(self) -> list[Any]: ...
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
        if False:  # pragma: no cover
            yield {}  # type: ignore[unreachable]


ToolsetFailureHandler = Callable[
    [str, Exception], dict[str, Any] | None
]
"""Handler for a toolset whose tool catalog failed to load.

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
#   │   └── COMPACTION_APPLIED
#   └── SESSION_END
#

class AgentRunnerEvent(StrEnum):
    """Events fired during a single ``run()`` / ``run_stream()`` execution.

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
        COMPACTION_APPLIED: Compaction completed or was cancelled.
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
    COMPACTION_APPLIED = "compaction_applied"



__all__ = [
    "SessionManager",
    "Extension",
    "MessageRole",
    "AgentRunnerEvent",
    "ToolsetFailureHandler",
]
