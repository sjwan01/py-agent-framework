"""Single-turn session behavior tests.

Covers the one-liner promise (``AgentRunner(model=...)`` runs without a
system prompt) and the warning when explicit context/summarizer config is
silently ignored for single-turn sessions.
"""
from __future__ import annotations

import pytest
from pydantic_ai.models.test import TestModel

from py_agent.models import ContextConfig, SummarizerConfig
from py_agent.runner import AgentRunner


@pytest.fixture
def model() -> TestModel:
    """Return a deterministic Pydantic AI test model."""
    return TestModel()


class TestSingleTurnRun:
    """Single-turn runs always require a system prompt."""

    async def test_run_without_system_prompt_raises(
        self, model: TestModel
    ) -> None:
        """A system prompt is required even for single-turn runs."""
        runner = AgentRunner(model=model)
        with pytest.raises(ValueError, match="system_prompt must be a non-empty string"):
            await runner.run("hello")

    async def test_run_with_system_prompt_succeeds(
        self, model: TestModel
    ) -> None:
        """A single-turn run with a system prompt works."""
        runner = AgentRunner(model=model, system_prompt="instructions")
        result = await runner.run("hello")
        assert result.output
        assert result.session_id

    async def test_run_stream_with_system_prompt_succeeds(
        self, model: TestModel
    ) -> None:
        """run_stream works with a system prompt."""
        runner = AgentRunner(model=model, system_prompt="instructions")
        events = [event async for event in runner.run_stream("hello")]
        assert events[-1]["type"] == "run_end"
        assert events[-1]["output"]


class TestSingleTurnConfigWarning:
    """Explicit context/summarizer config on single-turn is warned about."""

    def test_explicit_config_warns(self, model: TestModel) -> None:
        """Passing config to a single-turn runner triggers on_warning."""
        warnings: list[str] = []

        def _warn(msg: str, exc: Exception | None = None) -> None:
            warnings.append(msg)

        runner = AgentRunner(
            model=model,
            context_config=ContextConfig(),
            summarizer_config=SummarizerConfig(),
            on_warning=_warn,
        )

        # the configs are still ignored; the user is told instead of silent loss
        assert runner._context_config is None
        assert runner._compaction_summarizer is None
        assert any("ignored" in w.lower() for w in warnings)

    def test_no_warning_without_config(self, model: TestModel) -> None:
        """Minimal single-turn construction emits no warning."""
        warnings: list[str] = []

        def _warn(msg: str, exc: Exception | None = None) -> None:
            warnings.append(msg)

        AgentRunner(model=model, on_warning=_warn)
        assert warnings == []

    def test_multi_turn_does_not_warn(self, model: TestModel) -> None:
        """Multi-turn sessions accept config without a warning."""
        from py_agent.session import LocalSessionManager

        warnings: list[str] = []

        def _warn(msg: str, exc: Exception | None = None) -> None:
            warnings.append(msg)

        AgentRunner(
            model=model,
            session_manager=LocalSessionManager(db_path=":memory:"),
            context_config=ContextConfig(),
            summarizer_config=SummarizerConfig(),
            on_warning=_warn,
        )
        assert warnings == []
