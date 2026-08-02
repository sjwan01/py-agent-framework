"""Regression tests for multi-chunk streaming output reconstruction.

Verifies that ``run()`` and ``run_stream()`` rebuild the full text from
incremental chunks rather than concatenating accumulated prefixes (which
duplicates text), and that TOKEN_STREAM events carry true increments.

Pydantic AI's ``TestModel`` emits a single chunk, so these tests use a fake
streaming model that yields three temporally separated chunks.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ModelResponseStreamEvent,
    TextPart,
)
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.settings import ModelSettings

from py_agent.runner import AgentRunner
from py_agent.types import AgentRunnerEvent

_CHUNKS = ["Hel", "lo ", "world"]


class _StreamedFakeResponse(StreamedResponse):
    """Streamed response that yields ``chunks`` one at a time.

    The 0.15s pause between chunks crosses Pydantic AI's 0.1s debounce
    window so each chunk surfaces as its own stream group instead of being
    merged into a single one.
    """

    def __init__(
        self, *, chunks: list[str], model_request_parameters: ModelRequestParameters
    ):
        super().__init__(model_request_parameters=model_request_parameters)
        self._chunks = chunks

    @property
    def model_name(self) -> str:
        return "fake-stream"

    @property
    def provider_url(self) -> str:
        return "https://fake.test"

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def timestamp(self) -> datetime:
        return datetime.now(timezone.utc)

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        """Yield one part start plus one delta per remaining chunk."""
        first, *rest = self._chunks
        for event in self._parts_manager.handle_text_delta(
            vendor_part_id=None, content=first
        ):
            yield event
        for delta in rest:
            await asyncio.sleep(0.15)
            for event in self._parts_manager.handle_text_delta(
                vendor_part_id=None, content=delta
            ):
                yield event


class _StreamingFakeModel(Model):
    """Model whose streaming path emits ``chunks`` incrementally."""

    def __init__(self, chunks: list[str]):
        self._chunks = chunks

    @property
    def model_name(self) -> str:
        return "fake-stream"

    @property
    def system(self) -> str:
        return "fake"

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None = None,
        model_request_parameters: ModelRequestParameters | None = None,
    ) -> ModelResponse:
        """Return the joined chunks as a single non-streamed response."""
        return ModelResponse(parts=[TextPart(content="".join(self._chunks))])

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: Any = None,
    ) -> AsyncIterator[StreamedResponse]:
        """Yield a response that streams the chunks incrementally."""
        yield _StreamedFakeResponse(
            chunks=self._chunks,
            model_request_parameters=model_request_parameters,
        )


class _TokenCollector:
    """Minimal extension that captures TOKEN_STREAM chunks."""

    def __init__(self) -> None:
        self.chunks: list[str] = []

    async def on_agent_runner_event(
        self, event: str, data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Record the chunk payload for TOKEN_STREAM events."""
        if event == AgentRunnerEvent.TOKEN_STREAM:
            self.chunks.append(data["data"]["chunk"])
        return None


class TestRunMultiChunkStream:
    """run() must reconstruct the full text without duplicating prefixes."""

    async def test_run_output_is_full_text(self) -> None:
        """Concatenating incremental chunks yields the full text exactly once."""
        runner = AgentRunner(model=_StreamingFakeModel(_CHUNKS), system_prompt="sp")
        result = await runner.run("hi")

        assert result.output == "Hello world"


class TestRunStreamMultiChunk:
    """run_stream() must emit increments and rebuild the full text."""

    async def test_token_stream_chunks_are_increments(self) -> None:
        """TOKEN_STREAM events carry true increments, not accumulated prefixes."""
        collector = _TokenCollector()
        runner = AgentRunner(
            model=_StreamingFakeModel(_CHUNKS),
            system_prompt="sp",
            extensions=[collector],
        )

        events = [event async for event in runner.run_stream("hi")]

        assert collector.chunks == ["Hel", "lo ", "world"]
        assert events[-1]["type"] == "run_end"
        assert events[-1]["output"] == "Hello world"
