"""Tests for tool lifecycle hooks — TOOL_CALL block semantics.

Verifies that the ``block`` field in TOOL_CALL returns honours only an
explicit ``True`` value, matching the documented ``{block: true}`` contract.
"""
from __future__ import annotations

from typing import Any

from pydantic_ai import Tool as PydanticTool
from pydantic_ai.models.test import TestModel

from py_agent.runner import AgentRunner
from py_agent.types import AgentRunnerEvent


class _ToolBlocker:
    """Extension returning a fixed vote at every TOOL_CALL event."""

    def __init__(self, vote: dict[str, Any]):
        self._vote = vote

    async def on_agent_runner_event(
        self, event: str, data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Return the configured vote for TOOL_CALL, abstain otherwise."""
        if event == AgentRunnerEvent.TOOL_CALL:
            return self._vote
        return None


def _make_runner(vote: dict[str, Any], called: list[bool]) -> AgentRunner:
    """Build a runner whose tool records its execution in ``called``."""

    def tool_func(x: int = 1) -> str:
        called.append(True)
        return f"result {x}"

    return AgentRunner(
        model=TestModel(call_tools=["t"]),
        system_prompt="sp",
        tools=[PydanticTool(tool_func, name="t")],
        extensions=[_ToolBlocker(vote)],
    )


class TestToolCallBlockStrictness:
    """TOOL_CALL honours only an explicit True block value."""

    async def test_true_block_prevents_tool_execution(self) -> None:
        """An explicit True block value stops the tool from running."""
        called: list[bool] = []
        runner = _make_runner({"block": True}, called)

        await runner.run("hi")

        assert called == []

    async def test_truthy_non_true_does_not_block(self) -> None:
        """Truthy but non-True values (1, 'yes') do not block the tool."""
        for vote in [{"block": 1}, {"block": "yes"}]:
            called: list[bool] = []
            runner = _make_runner(vote, called)

            await runner.run("hi")

            assert called == [True]


class _ToolEventRecorder:
    """Extension recording every tool-lifecycle event it receives."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def on_agent_runner_event(
        self, event: str, data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Append the event name and payload."""
        self.events.append((event, data))
        return None


def _tool_func(x: int = 1) -> str:
    """A plain tool returning its argument."""
    return f"result {x}"


class TestToolEventSequence:
    """Tool lifecycle events fire in order with the right payloads."""

    async def test_tool_events_fire_in_order_with_payloads(self) -> None:
        """A tool run fires TOOL_START → TOOL_CALL → TOOL_RESULT → TOOL_END."""
        recorder = _ToolEventRecorder()
        runner = AgentRunner(
            model=TestModel(call_tools=["t"]),
            system_prompt="sp",
            tools=[PydanticTool(_tool_func, name="t")],
            extensions=[recorder],
        )

        await runner.run("hi")

        names = [name for name, _ in recorder.events]
        tool_events = {
            AgentRunnerEvent.TOOL_START,
            AgentRunnerEvent.TOOL_CALL,
            AgentRunnerEvent.TOOL_RESULT,
            AgentRunnerEvent.TOOL_END,
        }
        tool_names = [n for n in names if n in tool_events]
        assert tool_names == [
            AgentRunnerEvent.TOOL_START,
            AgentRunnerEvent.TOOL_CALL,
            AgentRunnerEvent.TOOL_RESULT,
            AgentRunnerEvent.TOOL_END,
        ]
        # payloads carry the tool name through every stage
        for name, data in recorder.events:
            if name in (AgentRunnerEvent.TOOL_CALL, AgentRunnerEvent.TOOL_RESULT):
                assert data["tool_name"] == "t"
            elif name in (AgentRunnerEvent.TOOL_START, AgentRunnerEvent.TOOL_END):
                assert data["name"] == "t"


class TestToolResultRewrite:
    """Extensions may rewrite the tool's return value at TOOL_RESULT."""

    async def test_content_rewritten_before_model_sees_it(self) -> None:
        """A TOOL_RESULT content override replaces the tool's real output."""
        class _Rewriter:
            async def on_agent_runner_event(
                self, event: str, data: dict[str, Any]
            ) -> dict[str, Any] | None:
                if event == AgentRunnerEvent.TOOL_RESULT:
                    return {"content": "rewritten by extension"}
                return None

        runner = AgentRunner(
            model=TestModel(call_tools=["t"]),
            system_prompt="sp",
            tools=[PydanticTool(_tool_func, name="t")],
            extensions=[_Rewriter()],
        )

        result = await runner.run("hi")

        assert "rewritten by extension" in result.output
        assert "result 1" not in result.output


class TestToolCallArgsRewrite:
    """Extensions may rewrite the tool's arguments at TOOL_CALL."""

    async def test_args_rewritten_before_tool_runs(self) -> None:
        """A TOOL_CALL args override reaches the tool function."""
        class _Rewriter:
            async def on_agent_runner_event(
                self, event: str, data: dict[str, Any]
            ) -> dict[str, Any] | None:
                if event == AgentRunnerEvent.TOOL_CALL:
                    return {"args": {"x": 99}}
                return None

        runner = AgentRunner(
            model=TestModel(call_tools=["t"]),
            system_prompt="sp",
            tools=[PydanticTool(_tool_func, name="t")],
            extensions=[_Rewriter()],
        )

        result = await runner.run("hi")

        assert "result 99" in result.output


class TestToolCallBlockReason:
    """A blocked tool surfaces the extension's reason in the result."""

    async def test_blocked_message_includes_reason(self) -> None:
        """The model receives 'Tool call blocked: <reason>'."""
        called: list[bool] = []
        blocker = _ToolBlocker({"block": True, "reason": "not allowed"})

        def tool_func(x: int = 1) -> str:
            called.append(True)
            return "ran"

        runner = AgentRunner(
            model=TestModel(call_tools=["t"]),
            system_prompt="sp",
            tools=[PydanticTool(tool_func, name="t")],
            extensions=[blocker],
        )

        result = await runner.run("hi")

        assert called == []
        assert "Tool call blocked: not allowed" in result.output


class TestMaxToolCallsPerTurn:
    """The per-turn tool call cap returns a message instead of executing."""

    async def test_over_limit_returns_limit_message(self) -> None:
        """Calls beyond max_tool_calls_per_turn are not executed."""
        called: list[str] = []

        def make(name: str):
            def f(x: int = 1) -> str:
                called.append(name)
                return f"{name} ran"
            return f

        runner = AgentRunner(
            model=TestModel(call_tools=["t1", "t2"]),
            system_prompt="sp",
            max_tool_calls_per_turn=1,
            tools=[
                PydanticTool(make("t1"), name="t1"),
                PydanticTool(make("t2"), name="t2"),
            ],
        )

        result = await runner.run("hi")

        assert called == ["t1"]
        assert "Tool call limit (1) reached for this turn." in result.output
