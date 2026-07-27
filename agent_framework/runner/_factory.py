"""Lifecycle and discovery for AgentRunner.

这个模块包含 AgentRunner 的工具注册、compaction 触发和 Extension
自动发现。所有函数的第一个参数 `self` 都是 AgentRunner 实例。
"""
from __future__ import annotations

from agent_framework.tools import LocalToolSource, ToolLifecycle
from agent_framework.types import ToolLifecycleEvent


# ── 工具注册（懒初始化，首次 run 时调用）─────────────────────

async def ensure_tool_lifecycle(self):
    """懒初始化 ToolLifecycle，注册所有来源的工具。

    这个函数在第一次 run() / run_stream() 时被调用，之后不再执行。
    注册顺序：
    1. 创建 ToolLifecycle 实例
    2. 把 Extension 的 on_tool_event handler 注册到所有工具事件
    3. 注册构造时传的 raw tools（作为 LocalToolSource）
    4. 调用每个 Extension 的 register_tool_sources()，注册它们的
       ToolSource（Local/MCP/Subagent）
    """
    # 已经初始化过，直接返回
    if self._tool_lifecycle_initialized:
        return self._tool_lifecycle

    # 首次创建：如果没有 tools 也没有 extensions，不需要 ToolLifecycle
    if self._tool_lifecycle is None:
        if self._raw_tools or self._extensions:
            self._tool_lifecycle = ToolLifecycle(on_warning=self._on_warning)
        else:
            self._tool_lifecycle_initialized = True
            return None

    # 先订阅 Extension 的工具事件处理器（register 之前订阅，这样
    # TOOL_CONFLICT 等事件在注册时就能被 Extension 拦截处理）
    for ext in self._extensions:
        handler = getattr(ext, "on_tool_event", None)
        if handler is None:
            continue
        for event in ToolLifecycleEvent:
            self._tool_lifecycle.on(event, handler)

    # 注册构造时传的 raw tools
    if self._raw_tools:
        await self._tool_lifecycle.add_source(LocalToolSource(self._raw_tools))

    # 调用每个 Extension 的 register_tool_sources()
    # Extension 可以在这里返回 LocalToolSource、MCPServerSource 等
    for ext in self._extensions:
        register = getattr(ext, "register_tool_sources", None)
        if register is None:
            continue
        try:
            sources = await register()
        except Exception as exc:  # pragma: no cover - fail-open
            self._on_warning(
                f"Extension {type(ext).__name__} register_tool_sources failed: {exc}",
                exc,
            )
            continue
        for src in sources or []:
            await self._tool_lifecycle.add_source(src)

    self._tool_lifecycle_initialized = True
    return self._tool_lifecycle


# ── Compaction 触发 ──────────────────────────────────────────

async def trigger_compaction(self, session_id: str) -> None:
    """异步压缩当前 session 的历史消息。

    在 _finalize_run 中通过 asyncio.create_task 触发，不阻塞这一轮
    的响应。流程：
    1. 取当前最大的 message_seq（这就是压缩边界——该 seq 及之前的
       全部消息都被压缩覆盖）
    2. load_history（protect_turns=0，如有旧的 compaction 摘要会
       被加载）→ 全部消息
    3. summarizer 总结全部消息（若为 None 则跳过 LLM 总结）
    4. 把摘要写入 compactions 表
    """
    try:
        # 取 raw 最大值，不受 compaction 表影响
        boundary_seq = await self._session_manager.get_max_message_seq(session_id)
        if boundary_seq < 0:
            return

        summarizer = self._compaction_summarizer
        if summarizer is None:
            return  # 未配置 SummarizerConfig，跳过 LLM 压缩

        # protect_turns=0 → 永远直接返回摘要 + 全量消息
        messages = await self._session_manager.load_history(session_id)
        summary = await summarizer.summarize(messages)

        await self._session_manager.apply_compaction(
            session_id,
            summary=summary,
            boundary_seq=boundary_seq,
        )
    except Exception as exc:  # pragma: no cover - fail-open
        self._on_warning(f"Compaction failed for session {session_id}: {exc}", exc)



