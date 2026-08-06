"""Validate SummarizerConfig → HarnessSummarizer field passthrough."""
from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.settings import ModelSettings

from py_agent._compaction import HarnessSummarizer
from py_agent.models import SummarizerConfig


class _FakeModel(Model):
    """Minimal concrete Model for testing SummarizerConfig."""

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None = None,
        model_request_parameters: ModelRequestParameters | None = None,
    ) -> ModelResponse:
        return ModelResponse(parts=[])

    @property
    def model_name(self) -> str:
        return "fake"

    @property
    def system(self) -> str:
        return "fake"


def _make_model() -> _FakeModel:
    """Return a minimal concrete model for tests."""
    return _FakeModel()


class TestSummarizerConfigFields:
    """SummarizerConfig Pydantic field validation."""

    def test_model_defaults_to_none(self) -> None:
        """model defaults to None (falls back to the main model)."""
        cfg = SummarizerConfig()
        assert cfg.model is None

    def test_model_rejects_non_model(self) -> None:
        """model only accepts Model instances or None."""
        for bad in [1, "gpt-4o", {}, []]:
            invalid: Any = bad
            with pytest.raises(ValidationError, match="is_instance_of"):
                SummarizerConfig(model=invalid)

    def test_defaults(self) -> None:
        """Default values."""
        cfg = SummarizerConfig(model=_make_model())
        assert cfg.max_output_tokens is None
        assert cfg.summary_prompt is None

    def test_all_fields_set(self) -> None:
        """All fields explicitly set."""
        m = _make_model()
        cfg = SummarizerConfig(
            model=m,
            max_output_tokens=300,
            summary_prompt="Custom prompt: {messages}",
        )
        assert cfg.model is m
        assert cfg.max_output_tokens == 300
        assert cfg.summary_prompt == "Custom prompt: {messages}"


class TestSummarizerConfigToHarnessSummarizer:
    """SummarizerConfig → HarnessSummarizer field passthrough."""

    def test_minimal_config(self) -> None:
        """Minimal config creates a summarizer."""
        m = _make_model()
        cfg = SummarizerConfig(model=m)
        s = HarnessSummarizer(
            model=m,
            max_output_tokens=cfg.max_output_tokens,
            summary_prompt=cfg.summary_prompt,
        )
        assert s._strategy is not None

    def test_full_config(self) -> None:
        """Full config creates a summarizer."""
        m = _make_model()
        cfg = SummarizerConfig(
            model=m,
            max_output_tokens=500,
            summary_prompt="Custom: {messages}",
        )
        s = HarnessSummarizer(
            model=m,
            max_output_tokens=cfg.max_output_tokens,
            summary_prompt=cfg.summary_prompt,
        )
        assert s._strategy is not None

    def test_custom_prompt_defaults_to_builtin(self) -> None:
        """summary_prompt=None uses the built-in template."""
        m = _make_model()
        cfg = SummarizerConfig(model=m)
        s = HarnessSummarizer(
            model=m,
            max_output_tokens=cfg.max_output_tokens,
            summary_prompt=cfg.summary_prompt,
        )
        # Built-in template contains section markers.
        strategy = s._strategy
        assert strategy is not None
