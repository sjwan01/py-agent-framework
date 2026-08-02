"""Validate constructor invariants."""
from __future__ import annotations

import pytest
from pydantic_ai.models.test import TestModel

from py_agent.models import ContextConfig
from py_agent.runner import AgentRunner


@pytest.fixture
def model() -> TestModel:
    """Return a deterministic Pydantic AI test model."""
    return TestModel()


class TestContextConfigValidation:
    """ContextConfig validation."""

    def test_low_above_high_raises(self) -> None:
        """low_watermark_ratio > high_watermark_ratio raises ValueError."""
        with pytest.raises(ValueError, match="watermark"):
            ContextConfig(low_watermark_ratio=0.9, high_watermark_ratio=0.5)

    def test_low_zero_raises(self) -> None:
        """low_watermark_ratio <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="watermark"):
            ContextConfig(low_watermark_ratio=0.0)

    def test_high_one_raises(self) -> None:
        """high_watermark_ratio >= 1 raises ValueError."""
        with pytest.raises(ValueError, match="watermark"):
            ContextConfig(high_watermark_ratio=1.0)

    def test_protect_turns_negative_raises(self) -> None:
        """Negative protect_turns raises ValueError."""
        with pytest.raises(ValueError, match="protect_turns"):
            ContextConfig(protect_turns=-1)

    def test_context_window_cap_zero_raises(self) -> None:
        """context_window_cap <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="context_window_cap"):
            ContextConfig(context_window_cap=0)

    def test_context_window_cap_negative_raises(self) -> None:
        """Negative context_window_cap raises ValueError."""
        with pytest.raises(ValueError, match="context_window_cap"):
            ContextConfig(context_window_cap=-1)

    def test_truncate_chars_zero_raises(self) -> None:
        """truncate_chars <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="truncate_chars"):
            ContextConfig(truncate_chars=0)

    def test_truncate_chars_negative_raises(self) -> None:
        """Negative truncate_chars raises ValueError."""
        with pytest.raises(ValueError, match="truncate_chars"):
            ContextConfig(truncate_chars=-1)

    def test_valid_config_passes(self) -> None:
        """Valid config does not raise."""
        cfg = ContextConfig(
            low_watermark_ratio=0.6,
            high_watermark_ratio=0.75,
            protect_turns=5,
        )
        assert cfg is not None


class TestAgentRunnerValidation:
    """AgentRunner constructor validation."""

    def test_max_tool_calls_zero_raises(self, model: TestModel) -> None:
        """max_tool_calls_per_turn <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="max_tool_calls"):
            AgentRunner(model=model, max_tool_calls_per_turn=0)

    def test_max_tool_calls_negative_raises(self, model: TestModel) -> None:
        """Negative max_tool_calls raises ValueError."""
        with pytest.raises(ValueError, match="max_tool_calls"):
            AgentRunner(model=model, max_tool_calls_per_turn=-1)

    def test_valid_max_tool_calls_passes(self, model: TestModel) -> None:
        """Positive max_tool_calls does not raise."""
        runner = AgentRunner(model=model, max_tool_calls_per_turn=10)
        assert runner._max_tool_calls_per_turn == 10


class TestThinkingValidation:
    """thinking_level validation in _build_agent."""

    async def test_invalid_thinking_level_warns_on_run(
        self, model: TestModel
    ) -> None:
        """An invalid thinking_level warns and the run still succeeds."""
        warnings: list[str] = []

        def _warn(msg: str, exc: Exception | None = None) -> None:
            warnings.append(msg)

        runner = AgentRunner(
            model=model,
            system_prompt="sp",
            thinking_level="invalid",
            on_warning=_warn,
        )

        result = await runner.run("hi")

        assert result.output
        assert any("Invalid thinking_level" in w for w in warnings)

    async def test_valid_thinking_level_no_warning(
        self, model: TestModel
    ) -> None:
        """A valid thinking_level produces no warning."""
        warnings: list[str] = []

        def _warn(msg: str, exc: Exception | None = None) -> None:
            warnings.append(msg)

        runner = AgentRunner(
            model=model,
            system_prompt="sp",
            thinking_level="high",
            on_warning=_warn,
        )

        await runner.run("hi")

        assert warnings == []
