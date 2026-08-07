"""Event sequence tests for run() / run_stream().

Verifies the full agent lifecycle event order observed by extensions, the
run()/run_stream() difference (TOKEN_STREAM only fires when streaming), and
what the external consumer receives from run_stream().
"""
from __future__ import annotations

from typing import Any, AsyncIterator

import pytest
from pydantic_ai.models.test import TestModel

from py_agent.runner import AgentRunner
from py_agent.types import AgentRunnerEvent


class _EventRecorder:
    """Extension that records every agent-runner event it receives."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def on_agent_runner_event(
        self, event: str, data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Append the event name and payload."""
        self.events.append((event, data))
        return None


class _StreamingRelay:
    """Extension that relays TOKEN_STREAM chunks to the external consumer."""

    async def on_agent_runner_event(
        self, event: str, data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """No-op chain handler; this extension only streams."""
        return None

    async def on_agent_runner_event_stream(
        self, event: str, data: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        """Forward each TOKEN_STREAM chunk as a consumer-visible token event."""
        if event == AgentRunnerEvent.TOKEN_STREAM:
            yield {"type": "token", "chunk": data["data"]["chunk"]}


@pytest.fixture
def model() -> TestModel:
    """Return a deterministic Pydantic AI test model."""
    return TestModel()


class TestRunStreamEventSequence:
    """run_stream() fires the full lifecycle in order."""

    async def test_full_lifecycle_order(self, model: TestModel) -> None:
        """Extensions observe every lifecycle event in the expected order."""
        recorder = _EventRecorder()
        runner = AgentRunner(
            model=model, system_prompt="sp", extensions=[recorder]
        )

        events = [event async for event in runner.run_stream("hi")]

        names = [name for name, _ in recorder.events]
        assert names == [
            AgentRunnerEvent.SESSION_START,
            AgentRunnerEvent.CONTEXT_PREPARE,
            AgentRunnerEvent.BEFORE_AGENT_RUN,
            AgentRunnerEvent.AGENT_START,
            AgentRunnerEvent.TOKEN_STREAM,
            AgentRunnerEvent.AFTER_AGENT_RUN,
            AgentRunnerEvent.AGENT_END,
            AgentRunnerEvent.SESSION_SAVE,
            AgentRunnerEvent.SESSION_END,
        ]
        assert events[-1]["type"] == "run_end"

    async def test_session_id_is_consistent(self, model: TestModel) -> None:
        """Every event carries the same session_id as the final run_end."""
        recorder = _EventRecorder()
        runner = AgentRunner(
            model=model, system_prompt="sp", extensions=[recorder]
        )

        events = [event async for event in runner.run_stream("hi")]

        session_ids = {data.get("session_id") for _, data in recorder.events}
        assert session_ids == {events[-1]["session_id"]}


class TestTokenStreamOnlyInRunStream:
    """TOKEN_STREAM fires only when streaming, never in run()."""

    async def test_run_has_no_token_stream(self, model: TestModel) -> None:
        """run() never fires TOKEN_STREAM."""
        recorder = _EventRecorder()
        runner = AgentRunner(
            model=model, system_prompt="sp", extensions=[recorder]
        )

        await runner.run("hi")

        names = [name for name, _ in recorder.events]
        assert AgentRunnerEvent.TOKEN_STREAM not in names

    async def test_run_stream_fires_token_stream(self, model: TestModel) -> None:
        """run_stream() fires TOKEN_STREAM with the output chunk."""
        recorder = _EventRecorder()
        runner = AgentRunner(
            model=model, system_prompt="sp", extensions=[recorder]
        )

        [event async for event in runner.run_stream("hi")]

        names = [name for name, _ in recorder.events]
        assert AgentRunnerEvent.TOKEN_STREAM in names


class TestRunStreamConsumerView:
    """What the external consumer of run_stream() receives."""

    async def test_no_extensions_yields_only_run_end(self, model: TestModel) -> None:
        """Without extensions, the consumer only receives the final run_end."""
        runner = AgentRunner(model=model, system_prompt="sp")

        events = [event async for event in runner.run_stream("hi")]

        assert [event["type"] for event in events] == ["run_end"]

    async def test_chain_only_extension_yields_nothing_to_consumer(
        self, model: TestModel
    ) -> None:
        """Chain-mode events never reach the consumer without a stream hook."""
        runner = AgentRunner(
            model=model, system_prompt="sp", extensions=[_EventRecorder()]
        )

        events = [event async for event in runner.run_stream("hi")]

        # the recorder observes TOKEN_STREAM via on_agent_runner_event, but
        # chain-mode events are not forwarded; only run_end reaches the caller
        assert [event["type"] for event in events] == ["run_end"]

    async def test_streaming_extension_chunks_reach_consumer(
        self, model: TestModel
    ) -> None:
        """Chunks yielded by streaming extensions appear in the consumer stream."""
        runner = AgentRunner(
            model=model, system_prompt="sp", extensions=[_StreamingRelay()]
        )

        events = [event async for event in runner.run_stream("hi")]

        types = [event["type"] for event in events]
        # TestModel emits a single chunk
        assert types.count("token") == 1
        assert types[-1] == "run_end"

    async def test_run_end_payload(self, model: TestModel) -> None:
        """run_end carries output, session_id, new_messages, and usage."""
        runner = AgentRunner(model=model, system_prompt="sp")

        events = [event async for event in runner.run_stream("hi")]

        run_end = events[-1]
        assert run_end["type"] == "run_end"
        assert run_end["output"]
        assert run_end["session_id"]
        assert run_end["new_messages"]
        assert run_end["usage"] is not None
