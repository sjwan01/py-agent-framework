"""Pydantic AI Hooks construction for AgentRunner.

这个模块只做一件事：构建 Hooks 实例，定义在模型调用工具的前、中、后
三个阶段插入什么逻辑。返回的 hooks 对象会被放入 Agent 的 capabilities
列表中。

Hooks 的三个阶段（按执行顺序）：

1. before_tool_execute —— 工具调用前
   → 发 TOOL_START 事件（Extension 可观测到"马上要调用工具了"）

2. tool_execute —— 工具真正执行时
   → 检查调用次数是否超限
   → 发 TOOL_CALL 事件（Extension 可阻止调用 / 修改参数）
   → 调用 handler(args) 真正执行工具

3. after_tool_execute —— 工具执行完成后
   → 发 TOOL_RESULT 事件（Extension 可修改返回值）
   → 发 TOOL_END 事件（Extension 可观测到"工具调用完成"）
"""
from __future__ import annotations

from typing import Any

from pydantic_ai.capabilities import Hooks

from py_agent.types import AgentRunnerEvent


def build_hooks(
    self,
    session_id: str,
    *,
    pending: list[Any] | None = None,
    streamers: list[Any] | None = None,
):
    """构建 Pydantic AI Hooks 实例。

    self   — AgentRunner 实例（通过类属性绑定传入）
    pending — 流式事件的暂存列表。run() 传空列表；run_stream() 传
              共享列表，在每次 fire 后 drain 给外部消费者
    streamers — 所有 Extension 的列表。事件同时推给它们（用于 yield
                流式输出）
    """
    pending = pending if pending is not None else []
    streamers = streamers if streamers is not None else []
    hooks = Hooks()

    # 本轮工具调用计数器（nonlocal 变量，三个闭包共享）
    tool_calls = 0
    max_tool_calls = self._max_tool_calls_per_turn

    # ── 阶段 1：工具调用前 ──────────────────────────────────
    @hooks.on.before_tool_execute
    async def _on_tool_start(ctx, *, call, tool_def, args):
        """通知 Extension：模型决定调用 tool_def.name，参数是 args。"""
        payload = {
            "session_id": session_id,
            "name": call.tool_name,
            "data": {"args": args},
        }
        # 同时用 fire（给老 Extension 的 on_agent_runner_event）和
        # notify_streamers（给流式 Extension 的 on_agent_runner_event_stream）
        await self._fire(AgentRunnerEvent.TOOL_START, payload)
        await self._notify_streamers(streamers, AgentRunnerEvent.TOOL_START, payload, pending)
        return args

    # ── 阶段 2：工具真正执行（可阻止 / 可改参数）────────────
    @hooks.on.tool_execute
    async def _on_tool_call(ctx, *, call, tool_def, args, handler):
        """拦截工具调用：限次数 → 发 TOOL_CALL 事件 → 执行工具。

        handler(args) 是 Pydantic AI 的默认执行逻辑，调用它才算真正
        执行了工具函数。
        """
        nonlocal tool_calls
        tool_calls += 1

        # 超限：不执行工具，返回一句说明给模型看
        if tool_calls > max_tool_calls:
            return f"Tool call limit ({max_tool_calls}) reached for this turn."

        # 发 TOOL_CALL 事件给 Extension。
        # Extension 可以返回 {"block": True} 阻止执行，
        # 或者返回 {"args": {...}} 修改参数。
        call_data = {
            "session_id": session_id,
            "tool_name": call.tool_name,
            "tool_call_id": call.tool_call_id,
            "args": dict(args),
        }
        call_result = await self._fire(AgentRunnerEvent.TOOL_CALL, call_data)

        # Extension 选择阻止
        if call_result.get("block"):
            reason = call_result.get("reason", "Blocked by extension")
            return f"Tool call blocked: {reason}"

        # Extension 修改了参数 → 用新参数执行
        if "args" in call_result:
            args = call_result["args"]

        # 真正执行工具函数
        return await handler(args)

    # ── 阶段 3：工具执行完成后 ──────────────────────────────
    @hooks.on.after_tool_execute
    async def _on_tool_result(ctx, *, call, tool_def, args, result):
        """工具执行完毕：发 TOOL_RESULT → Extension 可改结果 → 发 TOOL_END。"""
        # TOOL_RESULT：Extension 可以修改 content / is_error
        result_data = {
            "session_id": session_id,
            "tool_name": call.tool_name,
            "tool_call_id": call.tool_call_id,
            "content": result,
            "is_error": False,
        }
        result_data = await self._fire(AgentRunnerEvent.TOOL_RESULT, result_data)
        content = result_data.get("content", result)

        # TOOL_END：通知 Extension 工具调用完成（含最终返回值）
        payload = {
            "session_id": session_id,
            "name": call.tool_name,
            "data": {"result": content},
        }
        await self._fire(AgentRunnerEvent.TOOL_END, payload)
        await self._notify_streamers(streamers, AgentRunnerEvent.TOOL_END, payload, pending)

        return content

    return hooks
