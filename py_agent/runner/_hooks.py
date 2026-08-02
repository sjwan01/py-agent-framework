"""Pydantic AI Hooks construction for AgentRunner.

This module does one thing: build a ``Hooks`` instance that injects logic at
the three stages around a model's tool call. The returned hooks object is
placed into the Agent's capabilities list.

The three hook stages (in execution order):

1. ``before_tool_execute`` — before the tool runs.
   Fires ``TOOL_START`` so extensions can observe that a tool is about to run.

2. ``tool_execute`` — while the tool is actually running.
   Enforces the per-turn tool call limit, fires ``TOOL_CALL`` so extensions
   can block or modify arguments, then invokes ``handler(args)`` to run the tool.

3. ``after_tool_execute`` — after the tool finishes.
   Fires ``TOOL_RESULT`` so extensions can modify the return value, then fires
   ``TOOL_END`` so extensions can observe that the tool call completed.
"""
from __future__ import annotations

from typing import Any

from pydantic_ai.capabilities import Hooks

from py_agent.types import AgentRunnerEvent, Extension


def build_hooks(
    self: Any,
    session_id: str,
    *,
    pending: list[dict[str, Any]] | None = None,
    streamers: list[Extension] | None = None,
) -> Hooks:
    """Build a Pydantic AI ``Hooks`` instance.

    Args:
        self: ``AgentRunner`` instance (passed via class-attribute binding).
        session_id: Current session identifier.
        pending: Staging list for streaming events. ``run()`` passes an empty
            list; ``run_stream()`` passes a shared list that is drained to the
            external consumer after each event.
        streamers: List of extensions that should receive runtime events for
            streaming output.
    """
    pending = pending if pending is not None else []
    streamers = streamers if streamers is not None else []
    hooks = Hooks()

    # per-turn tool call counter shared by the three closures
    tool_calls = 0
    max_tool_calls = self._max_tool_calls_per_turn

    # Stage 1: before tool execution
    @hooks.on.before_tool_execute
    async def _on_tool_start(
        ctx: Any, *, call: Any, tool_def: Any, args: Any
    ) -> Any:
        """Notify extensions that the model decided to call ``call.tool_name``."""
        payload = {
            "session_id": session_id,
            "name": call.tool_name,
            "data": {"args": args},
        }
        # notify both chain-mode extensions and streaming extensions
        await self._fire(AgentRunnerEvent.TOOL_START, payload)
        await self._notify_streamers(streamers, AgentRunnerEvent.TOOL_START, payload, pending, on_warning=self._on_warning)
        return args

    # Stage 2: actual tool execution (blockable / arg mutable)
    @hooks.on.tool_execute
    async def _on_tool_call(
        ctx: Any, *, call: Any, tool_def: Any, args: Any, handler: Any
    ) -> Any:
        """Intercept the tool call: limit count, fire TOOL_CALL, then run.

        ``handler(args)`` is Pydantic AI's default execution logic; calling it
        actually invokes the underlying tool function.
        """
        nonlocal tool_calls
        tool_calls += 1

        # enforce per-turn tool call limit
        if tool_calls > max_tool_calls:
            return f"Tool call limit ({max_tool_calls}) reached for this turn."

        # fire TOOL_CALL so extensions can block or modify arguments
        call_data = {
            "session_id": session_id,
            "tool_name": call.tool_name,
            "tool_call_id": call.tool_call_id,
            "args": dict(args),
        }
        call_result = await self._fire(AgentRunnerEvent.TOOL_CALL, call_data)

        # extension chose to block
        if call_result.get("block") is True:
            reason = call_result.get("reason", "Blocked by extension")
            return f"Tool call blocked: {reason}"

        # extension modified arguments, use the new ones
        if "args" in call_result:
            args = call_result["args"]

        # actually invoke the tool function
        return await handler(args)

    # Stage 3: after tool execution completes
    @hooks.on.after_tool_execute
    async def _on_tool_result(
        ctx: Any, *, call: Any, tool_def: Any, args: Any, result: Any
    ) -> Any:
        """Tool execution finished: fire TOOL_RESULT, allow mutation, fire TOOL_END."""
        # TOOL_RESULT: extensions may modify content / is_error
        result_data = {
            "session_id": session_id,
            "tool_name": call.tool_name,
            "tool_call_id": call.tool_call_id,
            "content": result,
            "is_error": False,
        }
        result_data = await self._fire(AgentRunnerEvent.TOOL_RESULT, result_data)
        # A present ``content`` key wins — including an explicit None, which
        # extensions may use to clear the result. "No modification" is
        # expressed by returning None (no dict) or omitting ``content``, not by
        # ``{"content": None}``.
        content = result_data.get("content", result)

        # TOOL_END: notify extensions that the tool call completed
        payload = {
            "session_id": session_id,
            "name": call.tool_name,
            "data": {"result": content},
        }
        await self._fire(AgentRunnerEvent.TOOL_END, payload)
        await self._notify_streamers(streamers, AgentRunnerEvent.TOOL_END, payload, pending, on_warning=self._on_warning)

        return content

    return hooks
