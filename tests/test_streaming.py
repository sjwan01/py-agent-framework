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

from pydantic_ai import Tool as PydanticTool
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ModelResponseStreamEvent,
    TextPart,
)
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.function import (
    AgentInfo,
    DeltaToolCall,
    DeltaToolCalls,
    FunctionModel,
)
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


class TestStreamingCompleteness:
    """run_stream() completes when the model emits preamble + tool calls.

    The upstream defect (stop-at-first-output in the streaming path) truncates
    the run after a tool returns when the model emitted preamble text in the
    same response as the tool calls. These tests replay that exact shape with a
    deterministic FunctionModel: response 1 = preamble + tool call, response
    2 = summary. Regression: the summary must survive and token events must
    flow in preamble → tool → summary order.
    """

    async def test_run_stream_output_complete_with_preamble_and_tools(
        self,
    ) -> None:
        """Preamble + tool calls in response 1, summary in response 2: full output."""
        model = FunctionModel(stream_function=_preamble_then_summary_stream)
        collector = _TokenCollector()
        runner = AgentRunner(
            model=model,
            system_prompt="sp",
            tools=[PydanticTool(_tool_func, name="t")],
            extensions=[collector],
        )

        events = [event async for event in runner.run_stream("hi")]

        run_end = events[-1]
        assert run_end["type"] == "run_end"
        # the post-tool summary survives — no truncation
        assert run_end["output"] == "Summary: the tool ran."
        # both the preamble and the summary flowed as TOKEN_STREAM chunks
        assert collector.chunks == [
            "Preamble: about to call the tool. ",
            "Summary: the tool ran.",
        ]

    async def test_event_order_preamble_tool_summary(self) -> None:
        """run_with_events() yields token → tool_call → tool_result → token → run_end."""
        model = FunctionModel(stream_function=_preamble_then_summary_stream)
        runner = AgentRunner(
            model=model,
            system_prompt="sp",
            tools=[PydanticTool(_tool_func, name="t")],
        )

        events = [event async for event in runner.run_with_events("hi")]

        types = [event["type"] for event in events]
        assert types == ["token", "tool_call", "tool_result", "token", "run_end"]
        assert events[0]["chunk"] == "Preamble: about to call the tool. "
        assert events[3]["chunk"] == "Summary: the tool ran."
        assert events[-1]["output"] == "Summary: the tool ran."


class TestToolEventCompleteness:
    """Tool events arrive complete when text follows the tool call.

    B1: when the model emits text after a tool call, the consumer must still
    receive complete ``tool_call`` / ``tool_result`` events (payload fields
    intact, ordered between the preamble and summary text). B2: ``run_with_events()``
    output must be the complete final summary, not a truncated preamble.
    """

    async def test_tool_events_arrive_complete_with_following_text(self) -> None:
        """tool_call/tool_result payloads are complete and ordered after the preamble."""
        model = FunctionModel(stream_function=_preamble_then_summary_stream)
        runner = AgentRunner(
            model=model,
            system_prompt="sp",
            tools=[PydanticTool(_tool_func, name="t")],
        )

        events = [event async for event in runner.run_with_events("hi")]

        tool_call = next(e for e in events if e["type"] == "tool_call")
        assert tool_call["tool_name"] == "t"
        assert tool_call["tool_call_id"]
        assert tool_call["args"] == {"x": 1}

        tool_result = next(e for e in events if e["type"] == "tool_result")
        assert tool_result["tool_name"] == "t"
        assert tool_result["tool_call_id"] == tool_call["tool_call_id"]
        assert tool_result["content"] == "result 1"
        assert tool_result["is_error"] is False

        types = [e["type"] for e in events]
        # preamble token → tool_call → tool_result → summary token → run_end
        assert types.index("tool_call") > types.index("token")
        assert types.index("tool_call") < types.index("tool_result")
        assert types.index("tool_result") < types.index("run_end")

    async def test_run_with_events_output_complete(self) -> None:
        """run_with_events() delta stream joins to the full text; output is the summary."""
        model = FunctionModel(stream_function=_preamble_then_summary_stream)
        runner = AgentRunner(
            model=model,
            system_prompt="sp",
            tools=[PydanticTool(_tool_func, name="t")],
        )

        events = [event async for event in runner.run_with_events("hi")]

        tokens = [e["chunk"] for e in events if e["type"] == "token"]
        assert "".join(tokens) == (
            "Preamble: about to call the tool. " "Summary: the tool ran."
        )
        assert events[-1]["type"] == "run_end"
        assert events[-1]["output"] == "Summary: the tool ran."


def _tool_func(x: int = 1) -> str:
    """A plain tool returning its argument."""
    return f"result {x}"


async def _preamble_then_summary_stream(
    messages: list[ModelMessage], agent_info: AgentInfo
) -> AsyncIterator[str | DeltaToolCalls]:
    """Stream response 1 as preamble + tool call, response 2 as summary text."""
    if not any(isinstance(m, ModelResponse) for m in messages):
        yield "Preamble: about to call the tool. "
        yield {
            0: DeltaToolCall(name="t", json_args='{"x": 1}', tool_call_id="call_1")
        }
    else:
        yield "Summary: the tool ran."


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
