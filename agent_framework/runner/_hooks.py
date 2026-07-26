"""Pydantic AI Hooks construction for AgentRunner."""
from __future__ import annotations

from pydantic_ai.capabilities import Hooks

from agent_framework.types import AgentRunnerEvent


def build_hooks(self, session_id: str, *, pending=None, streamers=None):
    """Build Pydantic AI hooks for tool execution and event forwarding."""
    pending = pending if pending is not None else []
    streamers = streamers if streamers is not None else []
    hooks = Hooks()
    tool_calls = 0
    max_tool_calls = self._settings.max_tool_calls_per_turn

    @hooks.on.before_tool_execute
    async def _on_tool_start(ctx, *, call, tool_def, args):
        payload = {
            "session_id": session_id,
            "name": call.tool_name,
            "data": {"args": args},
        }
        await self._fire(AgentRunnerEvent.TOOL_START, payload)
        await self._notify_streamers(streamers, AgentRunnerEvent.TOOL_START, payload, pending)
        return args

    @hooks.on.tool_execute
    async def _on_tool_call(ctx, *, call, tool_def, args, handler):
        nonlocal tool_calls
        tool_calls += 1
        if tool_calls > max_tool_calls:
            return f"Tool call limit ({max_tool_calls}) reached for this turn."

        call_data = {
            "session_id": session_id,
            "tool_name": call.tool_name,
            "tool_call_id": call.tool_call_id,
            "args": dict(args),
        }
        call_result = await self._fire(AgentRunnerEvent.TOOL_CALL, call_data)
        if call_result.get("block"):
            reason = call_result.get("reason", "Blocked by extension")
            return f"Tool call blocked: {reason}"
        if "args" in call_result:
            args = call_result["args"]
        return await handler(args)

    @hooks.on.after_tool_execute
    async def _on_tool_result(ctx, *, call, tool_def, args, result):
        result_data = {
            "session_id": session_id,
            "tool_name": call.tool_name,
            "tool_call_id": call.tool_call_id,
            "content": result,
            "is_error": False,
        }
        result_data = await self._fire(AgentRunnerEvent.TOOL_RESULT, result_data)
        content = result_data.get("content", result)
        payload = {
            "session_id": session_id,
            "name": call.tool_name,
            "data": {"result": content},
        }
        await self._fire(AgentRunnerEvent.TOOL_END, payload)
        await self._notify_streamers(streamers, AgentRunnerEvent.TOOL_END, payload, pending)
        return content

    return hooks
