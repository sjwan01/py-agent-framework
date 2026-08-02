"""System prompt persistence and per-session isolation tests."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic_ai.models.test import TestModel

from py_agent.runner import AgentRunner
from py_agent.session import LocalSessionManager


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    """Return a temporary on-disk SQLite path."""
    return str(tmp_path / "test.db")


@pytest.fixture
def model() -> TestModel:
    """Return a deterministic Pydantic AI test model."""
    return TestModel()


class TestSystemPromptPerSessionIsolation:
    """System prompts are isolated per session."""

    async def test_each_session_gets_own_system_prompt(
        self, db_path: str, model: TestModel
    ) -> None:
        """Two sessions on the same runner write independent system prompts."""
        sm = LocalSessionManager(db_path=db_path)
        runner = AgentRunner(model=model, session_manager=sm, system_prompt="v1")

        await runner.run("hello", session_id="a")
        await runner.run("hello", session_id="b")

        sp_a = await sm.load_system_prompt("a")
        sp_b = await sm.load_system_prompt("b")

        assert sp_a == "v1"
        assert sp_b == "v1"


class TestSystemPromptRefresh:
    """System prompt changes write a new stored prompt row."""

    async def test_new_instance_new_prompt_refreshes_storage(
        self, db_path: str, model: TestModel
    ) -> None:
        """A new AgentRunner with a different prompt updates stored system_prompt."""
        sm = LocalSessionManager(db_path=db_path)
        runner1 = AgentRunner(model=model, session_manager=sm, system_prompt="v1")
        await runner1.run("hi", session_id="s1")

        assert await sm.load_system_prompt("s1") == "v1"

        runner2 = AgentRunner(model=model, session_manager=sm, system_prompt="v2")
        await runner2.run("hi", session_id="s1")

        assert await sm.load_system_prompt("s1") == "v2"


class TestSystemPromptFallback:
    """Empty system_prompt falls back to the stored prompt, or errors if absent."""

    async def test_none_prompt_falls_back_to_stored(
        self, db_path: str, model: TestModel
    ) -> None:
        """A reconnecting AgentRunner with no prompt uses the stored prompt."""
        sm = LocalSessionManager(db_path=db_path)
        runner1 = AgentRunner(
            model=model, session_manager=sm, system_prompt="stored-prompt"
        )
        await runner1.run("hi", session_id="s1")

        runner2 = AgentRunner(model=model, session_manager=sm, system_prompt=None)
        await runner2.run("hello again", session_id="s1")

        assert await sm.load_system_prompt("s1") == "stored-prompt"

    async def test_empty_prompt_without_storage_raises(
        self, db_path: str, model: TestModel
    ) -> None:
        """No system prompt and no stored prompt is rejected."""
        sm = LocalSessionManager(db_path=db_path)
        runner = AgentRunner(model=model, session_manager=sm, system_prompt=None)

        with pytest.raises(ValueError, match="system_prompt must be a non-empty string"):
            await runner.run("hi", session_id="s1")


class TestSystemPromptDirectApi:
    """Direct save/load behavior of LocalSessionManager."""

    async def test_save_and_load_roundtrip(self, db_path: str) -> None:
        """A saved system prompt can be loaded back intact."""
        sm = LocalSessionManager(db_path=db_path)
        sid = await sm.create_session()

        await sm.save_system_prompt(sid, "prompt")
        loaded = await sm.load_system_prompt(sid)

        assert loaded == "prompt"

    async def test_load_latest_returns_most_recent(self, db_path: str) -> None:
        """When multiple prompts exist, the latest is returned."""
        sm = LocalSessionManager(db_path=db_path)
        sid = await sm.create_session()

        await sm.save_system_prompt(sid, "first")
        await sm.save_system_prompt(sid, "second")

        loaded = await sm.load_system_prompt(sid)
        assert loaded == "second"
