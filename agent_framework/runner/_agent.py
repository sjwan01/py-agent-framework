"""AgentRunner — 框架的核心编排器。

入口：run(prompt) 和 run_stream(prompt)。两个方法共享完全相同的
生命周期流程，唯一的区别是 run() 把结果打包成一个 RunResult 返回，
run_stream() 把事件逐个 yield 给外部消费者。

一次 run 的完整生命周期：

  SESSION_START → load_history → context_prepare
  → BEFORE_AGENT_RUN → AGENT_START
  → [Pydantic AI 循环: 工具调用、token 生成]
  → AFTER_AGENT_RUN → AGENT_END
  → save_messages → [可选: compaction]
  → SESSION_END → run_end
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, Callable, cast

from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from agent_framework.models import (
    ContextManagerConfig,
    RunResult,
    SummarizerConfig,
)
from agent_framework._compaction import HarnessSummarizer
from agent_framework.context import BaselineState, ContextManager, PreparedContext
from agent_framework.session import SingleTurnSessionManager
from agent_framework.types import (
    SessionManager,
    AgentRunnerEvent,
)

from agent_framework.runner import _factory, _hooks, _internals

# Pydantic AI 支持的 thinking 深度级别。无效值会被忽略并回退。
_VALID_THINKING_LEVELS = {"minimal", "low", "medium", "high", "xhigh"}


class AgentRunner:
    """Orchestrates load → build → run → save."""

    # ── 类属性绑定：把子模块函数绑成实例方法 ──────────────────
    #
    # Python 技巧：module-level 函数第一个参数是 self，在这里通过
    # 类属性绑定，调用 self._fire(...) 时实际执行 _internals.fire(self, ...)。
    # 这个模式避免了多重继承或者把所有代码塞在一个文件里。

    # --- 来自 _internals.py（运行时工具函数）---
    _build_hooks = _hooks.build_hooks
    _messages_to_persist = staticmethod(_internals.messages_to_persist)
    _notify_streamers = staticmethod(_internals.notify_streamers)
    _drain_pending = staticmethod(_internals.drain_pending)
    _build_capabilities = _internals.build_capabilities
    _fire = _internals.fire
    _fire_notify = _internals.fire_notify
    _get_tools = _internals.get_tools

    # --- 来自 _factory.py（初始化/生命周期/发现）---
    _ensure_tool_lifecycle = _factory.ensure_tool_lifecycle
    _trigger_compaction = _factory.trigger_compaction

    # ── 构造函数 ─────────────────────────────────────────────

    def __init__(
        self,
        model: Model,
        *,
        system_prompt: str = "",
        thinking_enabled: bool = True,
        thinking_level: str | None = None,
        extensions: list[Any] | None = None,
        tools: list[Any] | tuple[()] = (),
        session_manager: SessionManager | None = None,
        context_manager_config: ContextManagerConfig | None = None,
        summarizer_config: SummarizerConfig | None = None,
        max_tool_calls_per_turn: int = 5,
        parallel_tool_calls: bool = False,
        hooks: Any = None,
        capabilities: list[Any] | None = None,
        on_warning: Callable[[str, Exception | None], None] | None = None,
    ):
        """构造 AgentRunner。

        最简单的用法只需 model：

            runner = AgentRunner(model=my_model)

        其他参数都是可选的：system_prompt、extensions、tools、
        session_manager、context_manager_config、summarizer_config 等。
        不传就用默认值（空 system prompt、SingleTurn session、
        无 context 管理、无 compaction）。
        """
        def _noop(msg: str, exc: Exception | None = None) -> None:
            pass

        self._model = model
        self._system_prompt = system_prompt
        self._thinking_enabled = thinking_enabled
        self._thinking_level = thinking_level
        self._scope = "main"

        self._extensions = extensions or []

        self._raw_tools = list(tools)
        self._tool_lifecycle = None
        self._tool_lifecycle_initialized = False

        self._session_manager = session_manager or SingleTurnSessionManager()

        self._context_manager: ContextManager | None = None
        if context_manager_config is not None:
            self._context_manager = ContextManager(
                context_window_cap=context_manager_config.context_window,
                low_watermark_ratio=context_manager_config.low_watermark_ratio,
                high_watermark_ratio=context_manager_config.high_watermark_ratio,
                protect_turns=context_manager_config.protect_turns,
                truncate_chars=context_manager_config.truncate_tool_result_chars,
            )
        self._protect_turns = (
            context_manager_config.protect_turns if context_manager_config else 5
        )

        self._compaction_summarizer: HarnessSummarizer | None = None
        if summarizer_config is not None:
            context_window = (
                context_manager_config.context_window
                if context_manager_config
                else 128_000
            )
            default_max = int(min(32_768, max(context_window * 0.1, 8_192)))
            self._compaction_summarizer = HarnessSummarizer(
                model=summarizer_config.model or model,
                max_output_tokens=summarizer_config.max_output_tokens
                or default_max,
                summary_prompt=summarizer_config.summary_prompt,
            )

        self._max_tool_calls_per_turn = max_tool_calls_per_turn
        self._parallel_tool_calls = parallel_tool_calls

        self._hooks = hooks
        self._capabilities = capabilities or []

        self._on_warning = on_warning or _noop

    # ── run() 和 run_stream() 的共享方法 ─────────────────────

    async def _setup_run(
        self, prompt: str, session_id: str | None
    ) -> tuple[str, list, list, bool]:
        """run() / run_stream() 共享的 run 前准备。

        返回值：(session_id, history, original_history, needs_compaction)

        其中：
        - history：经过 ContextManager 截断 + Extension 注入后的消息列表
        - original_history：从 DB 加载的原始消息列表（用于计算本轮 delta）
        - needs_compaction：ContextManager 判定是否需要压缩
        """
        # 第一步：创建或复用 session
        if session_id is None:
            session_id = await self._session_manager.create_session()
        else:
            await self._session_manager.ensure_session(session_id)

        # SESSION_START 事件
        await self._fire(AgentRunnerEvent.SESSION_START, {"session_id": session_id})

        # 第二步：加载历史消息（已含 compaction 判定逻辑）
        history = await self._session_manager.load_history(
            session_id, protect_turns=self._protect_turns
        )
        # 保存原始副本——之后 Extension 会在 history 上注入消息，
        # 我们需要 original_history 来计算本轮真正的新增消息
        original_history = list(history)

        # 第三步：ContextManager.prepare —— 截断、baseline diff 注入、
        #         判定是否需要 compaction（无 ContextManager 则跳过）
        needs_compaction = False
        if self._context_manager is not None:
            try:
                prepared = await self._context_manager.prepare(
                    history,
                    system_prompt=self._system_prompt,
                    current_state=BaselineState(),
                )
                history = prepared.messages
                needs_compaction = prepared.needs_compaction
            except Exception as exc:  # pragma: no cover - fail-open
                self._on_warning(f"ContextManager prepare failed: {exc}", exc)

        # 第四步：CONTEXT_PREPARE 事件（只读——Extension 的返回值被忽略，
        #          修改 messages 的唯一入口是下一步的 BEFORE_AGENT_RUN）
        ctx_data = {
            "session_id": session_id,
            "messages": history,
            "needs_compaction": needs_compaction,
        }
        await self._fire(AgentRunnerEvent.CONTEXT_PREPARE, ctx_data)

        # 第五步：BEFORE_AGENT_RUN 事件（Extension 在这里注入消息）
        before_data = {"session_id": session_id, "messages": history}
        before_result = await self._fire(AgentRunnerEvent.BEFORE_AGENT_RUN, before_data)
        if "messages" in before_result:
            history = before_result["messages"]

        # 第六步：AGENT_START 事件（只读——告诉 Extension 正式开始了）
        await self._fire(
            AgentRunnerEvent.AGENT_START,
            {"session_id": session_id, "prompt": prompt, "messages": history},
        )

        return session_id, history, original_history, needs_compaction

    async def _build_agent(
        self,
        session_id: str,
        *,
        pending: list | None = None,
        streamers: list | None = None,
    ) -> Agent:
        """构造本轮要用的 Pydantic AI Agent。

        - hooks：框架自己的 Hooks（工具拦截 + 事件分发）
        - capabilities：用户的 capabilities + 框架的 hooks
        - model_settings：并行工具调用开关 + thinking 配置
        """
        # 构建 Hooks（工具执行前/中/后的拦截逻辑）
        hooks = self._build_hooks(session_id, pending=pending, streamers=streamers)

        # 组装 capabilities 列表
        capabilities = self._build_capabilities() or []
        if hooks not in capabilities:
            capabilities.append(hooks)

        # 模型设置
        model_settings = ModelSettings(
            parallel_tool_calls=self._parallel_tool_calls,
        )
        # thinking 配置
        if self._thinking_enabled:
            level = self._thinking_level
            if level is not None and level not in _VALID_THINKING_LEVELS:
                self._on_warning(
                    f"Invalid thinking_level {level!r} ignored (valid: "
                    f"{', '.join(sorted(_VALID_THINKING_LEVELS))})",
                    None,
                )
                level = None
            model_settings["thinking"] = cast(Any, level if level is not None else True)

        return Agent(
            model=self._model,
            instructions=self._system_prompt,
            tools=await self._get_tools(),
            capabilities=capabilities or None,
            model_settings=model_settings,
        )

    async def _finalize_run(
        self,
        session_id: str,
        original_history: list,
        result,
        output: str,
        needs_compaction: bool,
        *,
        streamers: list | None = None,
        pending: list | None = None,
    ) -> AsyncIterator[dict]:
        """run() / run_stream() 共享的 run 后收尾。

        Agent 执行完毕后，按顺序：
        1. 发 AFTER_AGENT_RUN 和 AGENT_END 事件
        2. 保存本轮新增消息到 DB（增量）
        3. 触发 compaction（如果需要且未被取消）
        4. 发 SESSION_END 事件
        5. yield run_end（{"type": "run_end", ...}）

        这是一个 async generator——run_stream() 把中间事件 yield 给外部，
        run() 只消费最终的 run_end。
        """
        streamers = streamers if streamers is not None else []
        pending = pending if pending is not None else []

        # 计算本轮新增消息（原始历史 vs Agent 生成后的完整列表）
        delta_messages = self._messages_to_persist(original_history, result.all_messages())
        usage = result.usage

        # ── Agent 结束事件 ──
        payload = {"session_id": session_id, "output": output, "usage": usage}
        await self._fire(AgentRunnerEvent.AFTER_AGENT_RUN, payload)
        await self._notify_streamers(streamers, AgentRunnerEvent.AFTER_AGENT_RUN, payload, pending)
        await self._fire(AgentRunnerEvent.AGENT_END, payload)
        await self._notify_streamers(streamers, AgentRunnerEvent.AGENT_END, payload, pending)
        async for chunk in self._drain_pending(pending):
            yield chunk

        # ── 保存消息 ──
        await self._session_manager.save_messages(session_id, delta_messages)
        save_payload = {"session_id": session_id, "delta_messages": delta_messages}
        await self._fire(AgentRunnerEvent.SESSION_SAVE, save_payload)
        await self._notify_streamers(streamers, AgentRunnerEvent.SESSION_SAVE, save_payload, pending)
        async for chunk in self._drain_pending(pending):
            yield chunk

        # ── Compaction（如果需要且 Extension 没取消）──
        if needs_compaction:
            # 征询 Extension：是否取消本次压缩
            comp_result = await self._fire_notify(AgentRunnerEvent.COMPACTION_TRIGGER, {
                "session_id": session_id,
            })
            cancelled = comp_result.get("cancel", False)
            if not cancelled:
                # 异步触发，不阻塞当前轮次的响应
                asyncio.create_task(self._trigger_compaction(session_id))
            # 通知 Extension 压缩结果（取消还是已触发）
            applied_payload = {"session_id": session_id, "cancelled": bool(cancelled)}
            await self._fire(AgentRunnerEvent.COMPACTION_APPLIED, applied_payload)
            await self._notify_streamers(streamers, AgentRunnerEvent.COMPACTION_APPLIED, applied_payload, pending)
            async for chunk in self._drain_pending(pending):
                yield chunk

        # ── 会话结束 ──
        end_payload = {"session_id": session_id}
        await self._fire(AgentRunnerEvent.SESSION_END, end_payload)
        await self._notify_streamers(streamers, AgentRunnerEvent.SESSION_END, end_payload, pending)
        async for chunk in self._drain_pending(pending):
            yield chunk

        # ── 最终事件 ──
        yield {
            "type": "run_end",
            "session_id": session_id,
            "output": output,
            "new_messages": delta_messages,
            "usage": usage,
        }

    # ── 公开 API ─────────────────────────────────────────────

    async def run(self, prompt: str, *, session_id: str | None = None) -> RunResult:
        """执行一次对话，返回 RunResult。

        这是最常用的入口。内部流程：

           _setup_run → _build_agent → agent.run_stream（流式消费文本）
           → _finalize_run → 提取 run_end → 返回 RunResult

        流式消费只是实现细节——consumer 不需要知道底层走了 stream_text()。
        """
        # 1. 准备（加载历史、context 处理、Extension 注入）
        session_id, history, original_history, needs_compaction = await self._setup_run(
            prompt, session_id
        )
        # 2. 构造 Agent
        agent = await self._build_agent(session_id)

        # 3. 执行 Agent（始终走 run_stream，内部把 token 拼接成完整文本）
        output_parts: list[str] = []
        async with agent.run_stream(prompt, message_history=history) as result:
            async for text in result.stream_text(delta=False):
                output_parts.append(text)
                # 每个 token chunk 也触发 TOKEN_STREAM 事件
                # （但不走 notify_streamers——run() 不是流式 API）
                await self._fire(
                    AgentRunnerEvent.TOKEN_STREAM,
                    {
                        "session_id": session_id,
                        "data": {"chunk": text},
                    },
                )

        output = "".join(output_parts)

        # 4. 收尾（保存、compaction、事件）
        async for event in self._finalize_run(
            session_id, original_history, result, output, needs_compaction
        ):
            if event["type"] == "run_end":
                return RunResult(
                    output=event["output"],
                    session_id=event["session_id"],
                    new_messages=event["new_messages"],
                    usage=event["usage"],
                )

        # 不应该走到这里——_finalize_run 一定会 yield run_end
        raise RuntimeError("run did not produce a run_end event")  # pragma: no cover

    async def run_stream(
        self, prompt: str, *, session_id: str | None = None
    ) -> AsyncIterator[dict]:
        """执行一次对话，逐条 yield 事件给外部消费者。

        和 run() 的区别：
        - 构造 Agent 时传了 streamers——所有 Extension 都会收到运行时事件
        - token chunk、工具调用、生命周期事件的流式产出被逐个 yield
        - 最终事件同样是 {"type": "run_end", ...}
        """
        # 1. 准备（和 run() 完全一样）
        session_id, history, original_history, needs_compaction = await self._setup_run(
            prompt, session_id
        )

        # 2. 构造 Agent——传入 streamers，开启流式事件推送
        streamers = list(self._extensions)
        pending: list[dict] = []
        agent = await self._build_agent(
            session_id, pending=pending, streamers=streamers
        )

        # 3. 执行 Agent——每个 token chunk 都 notify + drain
        output_parts: list[str] = []
        async with agent.run_stream(prompt, message_history=history) as result:
            async for text in result.stream_text(delta=False):
                output_parts.append(text)
                # TOKEN_STREAM 事件：既发 fire（老 Extension），也发
                # notify_streamers（流式 Extension 的 yield 进入 pending）
                payload = {
                    "session_id": session_id,
                    "data": {"chunk": text},
                }
                await self._fire(AgentRunnerEvent.TOKEN_STREAM, payload)
                await self._notify_streamers(
                    streamers, AgentRunnerEvent.TOKEN_STREAM, payload, pending,
                )
                # drain：把 pending 里的 chunks yield 给外部消费者
                async for chunk in self._drain_pending(pending):
                    yield chunk

        output = "".join(output_parts)

        # 4. 收尾——yield 所有 post-agent 事件（包括最终的 run_end）
        async for event in self._finalize_run(
            session_id,
            original_history,
            result,
            output,
            needs_compaction,
            streamers=streamers,
            pending=pending,
        ):
            yield event
