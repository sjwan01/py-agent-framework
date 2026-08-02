"""SESSION_SAVE event ordering and delta write-back tests.

Verifies that ``SESSION_SAVE`` fires *before* messages are persisted and
that extension modifications to ``delta_messages`` are actually written to
the database.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models.test import TestModel

from py_agent.runner import AgentRunner
from py_agent.session import LocalSessionManager
from py_agent.types import AgentRunnerEvent, SessionManager


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    """Return a temporary on-disk SQLite path."""
    return str(tmp_path / "test.db")


@pytest.fixture
def model() -> TestModel:
    """Return a deterministic Pydantic AI test model."""
    return TestModel()


class _SaveHookExtension:
    """Minimal extension that observes and rewrites the delta at SESSION_SAVE."""

    def __init__(self, session_manager: SessionManager, injected: list[ModelMessage]):
        self._sm = session_manager
        self._injected = injected
        self.seq_before_save: int | None = None

    async def on_agent_runner_event(
        self, event: str, data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Record the DB state at SESSION_SAVE and append a message to the delta."""
        if event == AgentRunnerEvent.SESSION_SAVE:
            # The event must fire before persistence: the DB must still be empty.
            self.seq_before_save = await self._sm.get_max_message_seq(
                data["session_id"]
            )
            data["delta_messages"] = list(data["delta_messages"]) + self._injected
            return {"delta_messages": data["delta_messages"]}
        return None


def _contents(messages: list[Any]) -> list[str]:
    """Extract all string part contents from a message list."""
    return [
        part.content
        for msg in messages
        for part in getattr(msg, "parts", ())
        if hasattr(part, "content")
    ]


def _injected_message() -> ModelMessage:
    """Return a user request that the extension appends to the delta."""
    ts = datetime.now(timezone.utc)
    return ModelRequest(
        parts=[UserPromptPart(content="injected by extension", timestamp=ts)],
        kind="request",
        timestamp=ts,
    )


class TestDeltaComposition:
    """The persisted delta is exactly SDK new_messages plus injected messages."""

    async def test_persisted_delta_combines_sdk_and_injected(
        self, db_path: str, model: TestModel
    ) -> None:
        """SDK new_messages exclude injections; persistence combines both."""
        sm = LocalSessionManager(db_path=db_path)

        class _Injector:
            async def on_agent_runner_event(
                self, event: str, data: dict[str, Any]
            ) -> dict[str, Any] | None:
                if event == AgentRunnerEvent.BEFORE_AGENT_RUN:
                    data["messages"] = list(data["messages"]) + [_injected_message()]
                    return {"messages": data["messages"]}
                return None

        runner = AgentRunner(
            model=model, session_manager=sm, system_prompt="sp",
            extensions=[_Injector()],
        )
        result = await runner.run("hello")

        # the SDK's own new_messages() excludes the injected message but
        # includes the user prompt
        sid, history, injected, _, _ = await runner._setup_run("again", result.session_id)
        agent = await runner._build_agent(sid, active_sp="sp")
        sdk_result = await agent.run("again", message_history=history)
        sdk_new = _contents(sdk_result.new_messages())
        assert "again" in sdk_new
        assert "injected by extension" not in sdk_new

        # the persisted delta combines the SDK-tracked messages and the injected one
        persisted = _contents(await sm.load_history(result.session_id, protect_turns=0))
        assert "hello" in persisted
        assert "injected by extension" in persisted
        assert any("injected by extension" in _contents([m]) for m in injected)


class TestSessionSaveOrdering:
    """SESSION_SAVE fires before persistence and can rewrite the delta."""

    async def test_fires_before_persist_and_injected_message_is_saved(
        self, db_path: str, model: TestModel
    ) -> None:
        """The event sees an empty DB and its delta rewrite is persisted."""
        sm = LocalSessionManager(db_path=db_path)
        ext = _SaveHookExtension(sm, [_injected_message()])
        runner = AgentRunner(
            model=model, session_manager=sm, system_prompt="sp", extensions=[ext]
        )

        result = await runner.run("hello")

        # ordering: SESSION_SAVE fired before any message was written
        assert ext.seq_before_save == -1

        # the extension-injected message was persisted
        history = await sm.load_history(result.session_id)
        assert "injected by extension" in _contents(history)

    async def test_rewrite_reflected_in_run_end(
        self, db_path: str, model: TestModel
    ) -> None:
        """RunResult.new_messages reports the persisted (rewritten) delta."""
        sm = LocalSessionManager(db_path=db_path)
        ext = _SaveHookExtension(sm, [_injected_message()])
        runner = AgentRunner(
            model=model, session_manager=sm, system_prompt="sp", extensions=[ext]
        )

        result = await runner.run("hello")

        assert "injected by extension" in _contents(result.new_messages)
