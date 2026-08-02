"""Validate ContextConfig passthrough + AgentRunner context_config handling."""
from __future__ import annotations

import pytest
from pydantic_ai.models.test import TestModel

from py_agent.models import ContextConfig, SummarizerConfig
from py_agent.runner import AgentRunner
from py_agent.session import LocalSessionManager, SingleTurnSessionManager
from tests.test_summarizer_config import _make_model


@pytest.fixture
def model() -> TestModel:
    """Return a deterministic Pydantic AI test model."""
    return TestModel()


class TestContextConfigFields:
    """ContextConfig Pydantic field validation."""

    def test_defaults(self) -> None:
        """Default values."""
        cfg = ContextConfig()
        assert cfg.context_window_cap == 128_000
        assert cfg.low_watermark_ratio == 0.6
        assert cfg.high_watermark_ratio == 0.75
        assert cfg.protect_turns == 5
        assert cfg.truncate_chars == 1_000

    def test_all_fields_set(self) -> None:
        """All fields explicitly set."""
        cfg = ContextConfig(
            context_window_cap=64_000,
            low_watermark_ratio=0.5,
            high_watermark_ratio=0.8,
            protect_turns=3,
            truncate_chars=500,
        )
        assert cfg.context_window_cap == 64_000
        assert cfg.low_watermark_ratio == 0.5
        assert cfg.high_watermark_ratio == 0.8
        assert cfg.protect_turns == 3
        assert cfg.truncate_chars == 500


class TestAgentRunnerContextConfig:
    """AgentRunner accepts ContextConfig for context management."""

    def test_context_config_with_config(self, model: TestModel) -> None:
        """Passing context_config creates a ContextConfig."""
        cfg = ContextConfig(context_window_cap=50_000, protect_turns=3)
        runner = AgentRunner(
            model=model,
            session_manager=LocalSessionManager(db_path=":memory:"),
            context_config=cfg,
        )
        assert runner._context_config is not None
        assert runner._protect_turns == 3
        assert runner._context_config.context_window_cap == 50_000


class TestAgentRunnerSummarizerFallback:
    """SummarizerConfig.model=None reuses the main model."""

    def test_summarizer_no_model_falls_back_to_main(self) -> None:
        """SummarizerConfig() without model uses the main model."""
        cfg = SummarizerConfig()
        assert cfg.model is None

    def test_summarizer_explicit_model(self) -> None:
        """SummarizerConfig(model=another) uses the specified model."""
        m = _make_model()
        cfg = SummarizerConfig(model=m)
        assert cfg.model is m


class TestAgentRunnerMinimal:
    """AgentRunner(model=my_model) works in one line."""

    def test_minimal_constructor(self, model: TestModel) -> None:
        """Minimal construction with only model."""
        runner = AgentRunner(model=model)
        assert runner._model is model
        assert isinstance(runner._session_manager, SingleTurnSessionManager)
        # Default values
        assert runner._max_tool_calls_per_turn == 5
        assert runner._parallel_tool_calls is False
        assert runner._thinking_enabled is True
        assert runner._thinking_level is None

    def test_thinking_params(self, model: TestModel) -> None:
        """thinking params are passed through."""
        runner = AgentRunner(
            model=model,
            thinking_enabled=False,
            thinking_level="high",
        )
        assert runner._thinking_enabled is False
        assert runner._thinking_level == "high"

    def test_tool_guardrails(self, model: TestModel) -> None:
        """Tool guardrail params are passed through."""
        runner = AgentRunner(
            model=model,
            max_tool_calls_per_turn=10,
            parallel_tool_calls=True,
        )
        assert runner._max_tool_calls_per_turn == 10
        assert runner._parallel_tool_calls is True


class TestAgentRunnerAutoConfig:
    """Multi-turn session auto-configures context config and summarizer."""

    def test_multi_turn_auto_context_config(self, model: TestModel) -> None:
        """LocalSessionManager -> auto-creates default ContextConfig."""
        runner = AgentRunner(
            model=model,
            session_manager=LocalSessionManager(db_path=":memory:"),
        )
        assert runner._context_config is not None
        assert runner._protect_turns == 5

    def test_multi_turn_auto_summarizer(self, model: TestModel) -> None:
        """LocalSessionManager -> auto-creates HarnessSummarizer using main model."""
        runner = AgentRunner(
            model=model,
            session_manager=LocalSessionManager(db_path=":memory:"),
        )
        assert runner._compaction_summarizer is not None

    def test_single_turn_no_auto_config(self, model: TestModel) -> None:
        """SingleTurn -> no auto context config or summarizer."""
        runner = AgentRunner(model=model)
        assert runner._context_config is None
        assert runner._compaction_summarizer is None
        assert runner._protect_turns == 0

    def test_single_turn_ignores_explicit_config(self, model: TestModel) -> None:
        """SingleTurn -> explicit config is ignored."""
        runner = AgentRunner(
            model=model,
            context_config=ContextConfig(),
            summarizer_config=SummarizerConfig(model=_make_model()),
        )
        assert runner._context_config is None
        assert runner._compaction_summarizer is None
        assert runner._protect_turns == 0
