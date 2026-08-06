"""Tests for run_with_events() — a direct event stream for bare consumers.

Verifies that a consumer with no extensions receives normalized
``token`` / ``tool_call`` / ``tool_result`` / ``run_end`` events in order,
that the thinking + streaming warning is inherited from ``run_stream()``,
that ``run_stream()`` behavior is unchanged, and that user extensions keep
observing events when the internal bridge is present.
"""
from __future__ import annotations

from typing import Any

from pydantic_ai import Tool as PydanticTool
from pydantic_ai.models.test import TestModel

from py_agent.runner import AgentRunner
from py_agent.types import AgentRunnerEvent


def _tool_func(x: int = 1) -> str:
    """A plain tool returning its argument."""
    return f"result {x}"


class _EventRecorder:
    """Extension recording every chain-mode event it receives."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def on_agent_runner_event(
        self, event: str, data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Append the event name and payload."""
        self.events.append((event, data))
        return None


class TestBareConsumerEventStream:
    """A consumer with no extensions receives the normalized event stream."""

    async def test_receives_all_event_types_in_order(self) -> None:
        """token, tool_call, tool_result, and run_end arrive in order."""
        runner = AgentRunner(
            model=TestModel(call_tools=["t"]),
            system_prompt="sp",
            tools=[PydanticTool(_tool_func, name="t")],
        )

        events = [event async for event in runner.run_with_events("hi")]

        types = [event["type"] for event in events]
        assert "tool_call" in types
        assert "tool_result" in types
        assert "token" in types
        assert types[-1] == "run_end"
        # tool events precede the final text chunk, which precedes run_end
        assert types.index("tool_call") < types.index("tool_result")
        assert types.index("tool_result") < types.index("token")
        assert types.index("token") < types.index("run_end")

    async def test_normalized_event_shapes(self) -> None:
        """Each event type carries its documented payload."""
        runner = AgentRunner(
            model=TestModel(call_tools=["t"]),
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

        token = next(e for e in events if e["type"] == "token")
        assert isinstance(token["chunk"], str)

        run_end = events[-1]
        assert run_end["session_id"]
        assert run_end["output"]
        assert run_end["new_messages"]
        assert run_end["usage"] is not None


class TestWarningInheritance:
    """run_with_events() inherits the thinking + streaming warning."""

    async def test_warns_once_on_first_iteration(self) -> None:
        """The first consumed run_with_events() warns; a second call does not."""
        warnings: list[str] = []

        def _warn(msg: str, exc: Exception | None = None) -> None:
            warnings.append(msg)

        runner = AgentRunner(model=TestModel(), system_prompt="sp", on_warning=_warn)

        [event async for event in runner.run_with_events("hi")]
        assert len(warnings) == 1

        [event async for event in runner.run_with_events("hi")]
        assert len(warnings) == 1

    async def test_warning_mentions_defect(self) -> None:
        """The warning names the pydantic-ai defect and the run() workaround."""
        warnings: list[str] = []

        def _warn(msg: str, exc: Exception | None = None) -> None:
            warnings.append(msg)

        runner = AgentRunner(model=TestModel(), system_prompt="sp", on_warning=_warn)

        [event async for event in runner.run_with_events("hi")]

        assert len(warnings) == 1
        assert "pydantic-ai" in warnings[0]
        assert "run()" in warnings[0]


class TestRunStreamUnchanged:
    """run_stream() keeps its existing contract."""

    async def test_bare_consumer_still_receives_only_run_end(self) -> None:
        """Without a streaming extension, run_stream() yields only run_end."""
        runner = AgentRunner(model=TestModel(), system_prompt="sp")

        events = [event async for event in runner.run_stream("hi")]

        assert [event["type"] for event in events] == ["run_end"]


class TestExtensionCoexistence:
    """User extensions keep working under run_with_events()."""

    async def test_chain_extension_still_observes_tool_call(self) -> None:
        """A recording extension sees TOOL_CALL while the bridge is active."""
        recorder = _EventRecorder()
        runner = AgentRunner(
            model=TestModel(call_tools=["t"]),
            system_prompt="sp",
            tools=[PydanticTool(_tool_func, name="t")],
            extensions=[recorder],
        )

        events = [event async for event in runner.run_with_events("hi")]

        names = [name for name, _ in recorder.events]
        assert AgentRunnerEvent.TOOL_CALL in names
        assert AgentRunnerEvent.TOOL_RESULT in names
        assert AgentRunnerEvent.TOKEN_STREAM in names
        # the consumer still receives the normalized events alongside the
        # extension's observations
        assert [e["type"] for e in events][-1] == "run_end"
        assert any(e["type"] == "tool_call" for e in events)


class TestNoToolPath:
    """run_with_events() without tools yields token + run_end only."""

    async def test_no_tool_call_without_tools(self) -> None:
        """A plain run has no tool events; token and run_end still arrive."""
        runner = AgentRunner(model=TestModel(), system_prompt="sp")

        events = [event async for event in runner.run_with_events("hi")]

        types = [event["type"] for event in events]
        assert "token" in types
        assert "tool_call" not in types
        assert "tool_result" not in types
        assert types[-1] == "run_end"
