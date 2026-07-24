"""Core types — ABCs, Protocols, and enums.

Zero implementation. Pure interface definitions.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Awaitable
from enum import StrEnum
from typing import Any, Protocol


# ── SessionManager (external seam) ────────────────────────────────────

class SessionManager(ABC):
    @abstractmethod
    async def create_session(self, *, metadata: dict | None = None) -> str: ...
    @abstractmethod
    async def load_history(self, session_id: str) -> list: ...
    @abstractmethod
    async def save_messages(self, session_id: str, messages: list) -> None: ...
    @abstractmethod
    async def apply_compaction(self, session_id: str, summary: str, boundary_entry_id: str) -> None: ...


# ── Extension Protocol ────────────────────────────────────────────────

class Extension(Protocol):
    async def register_tool_sources(self) -> list[Any]: ...
    async def on_tool_event(self, event: str, data: dict) -> dict | None: ...
    async def on_agent_runner_event(self, event: str, data: dict) -> dict | None: ...
    async def on_agent_runner_event_stream(
        self, event: str, data: dict
    ) -> AsyncIterator[dict]:
        """Optional async generator for streaming extensions.

        If the extension does not override this, ``run()`` and ``run_stream()``
        ignore it.
        """
        if False:  # pragma: no cover
            yield {}


# ── ToolSource ─────────────────────────────────────────────────────────

class ToolSource(ABC):
    @abstractmethod
    async def discover(self) -> list: ...
    @property
    @abstractmethod
    def source_type(self) -> str: ...
    @property
    @abstractmethod
    def source_id(self) -> str: ...
    @property
    def scope(self) -> str:
        """Visibility scope for this source. Defaults to global ('all')."""
        return "all"


# ── Event enums ───────────────────────────────────────────────────────

class ToolLifecycleEvent(StrEnum):
    TOOL_DISCOVERED = "tool_discovered"
    TOOL_CONFLICT = "tool_conflict"
    TOOL_REGISTERED = "tool_registered"
    TOOL_REMOVED = "tool_removed"


class AgentRunnerEvent(StrEnum):
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    CONTEXT_PREPARE = "context_prepare"
    BEFORE_AGENT_RUN = "before_agent_run"
    AGENT_RUN = "agent_run"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    AFTER_AGENT_RUN = "after_agent_run"
    SESSION_SAVE = "session_save"
    COMPACTION_TRIGGER = "compaction_trigger"
    COMPACTION_APPLIED = "compaction_applied"


# ── Event handler type ────────────────────────────────────────────────

ToolEventHandler = Callable[[str, dict], Awaitable[dict | None]]
