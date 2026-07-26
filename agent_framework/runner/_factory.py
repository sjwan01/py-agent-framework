"""Init-time factories, lifecycle, and discovery for AgentRunner.

这个模块包含 AgentRunner 的初始化/工厂方法、工具注册、compaction
触发、和 Extension 自动发现。所有函数的第一个参数 `self` 都是
AgentRunner 实例。
"""
from __future__ import annotations

import os
import tempfile

from agent_framework.compaction import HarnessSummarizer
from agent_framework.context import ContextManager
from agent_framework.session import LocalSessionManager, PostgresSessionManager
from agent_framework.tools import LocalToolSource, ToolLifecycle
from agent_framework.types import ToolLifecycleEvent


# ── 工厂函数（__init__ 时的默认值创建）───────────────────────

def default_context_manager(self) -> ContextManager | None:
    """创建默认的 ContextManager，参数全部来自 Settings。"""
    return ContextManager(
        context_window_cap=self._settings.context_window,
        low_watermark_ratio=self._settings.low_watermark_ratio,
        high_watermark_ratio=self._settings.high_watermark_ratio,
        protect_turns=self._settings.protect_turns,
        truncate_chars=self._settings.truncate_tool_result_chars,
    )


def default_compaction_summarizer(self, model):
    """创建 Harness 压缩总结器，使用指定模型。"""
    from agent_framework.compaction import HarnessSummarizer
    return HarnessSummarizer(model=model, settings=self._settings)


def default_session_manager(self):
    """选择 SessionManager 适配器。

    优先级：Postgres（如果配了 postgres_url）→ SQLite（临时文件）。

    SQLite 临时文件在 AgentRunner 进程退出时自动删除。
    """
    if self._settings.postgres_url:
        return PostgresSessionManager(
            pg_url=self._settings.postgres_url,
            pool_size=self._settings.pg_pool_size,
            max_overflow=self._settings.pg_max_overflow,
        )
    db_path = self._settings.sqlite_path
    if db_path is None:
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
    return LocalSessionManager(db_path=db_path)


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
    3. summarizer 总结全部消息
    4. 把摘要写入 compactions 表
    """
    try:
        # 取 raw 最大值，不受 compaction 表影响
        boundary_seq = await self._session_manager.get_max_message_seq(session_id)
        if boundary_seq < 0:
            return

        # protect_turns=0 → 永远直接返回摘要 + 全量消息
        messages = await self._session_manager.load_history(session_id)

        summarizer = self._compaction_summarizer
        summary = await summarizer.summarize(messages)

        await self._session_manager.apply_compaction(
            session_id,
            summary=summary,
            boundary_seq=boundary_seq,
        )
    except Exception as exc:  # pragma: no cover - fail-open
        self._on_warning(f"Compaction failed for session {session_id}: {exc}", exc)



