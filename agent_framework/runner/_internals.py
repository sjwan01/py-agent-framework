"""Runtime helpers for AgentRunner.

供 _agent.py 中的 AgentRunner 类通过类属性绑定调用（如：
_fire = _internals.fire）。这些函数的第一个参数 `self` 就是
AgentRunner 实例本身，所以函数内部可以直接访问 self._settings、
self._extensions 等属性。
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from pydantic_ai.messages import ModelRequest, ModelResponse


def messages_to_persist(original_history: list, all_messages: list) -> list:
    """计算本轮应该存入 DB 的消息（纯差集）。

    Pydantic AI 的 result.new_messages() 会把 Extension 在
    BEFORE_AGENT_RUN 中注入到 message_history 的消息当作"旧历史"
    而排除。但 V2 需要这些注入消息也被持久化，以便下一轮加载时
    能恢复。SDK 没提供区分"原始加载历史"和"Extension 注入消息"
    的原语，所以这里手动算差集。

    差集逻辑：
    1. 先用 id() 做快速身份判断——Pydantic AI 不会深拷贝传入的
       message_history，所以同一个对象 id 不变。
    2. 如果消息被拷贝了（id 变了），退化为按内容键去重——比较
       (kind, run_id, parts摘要)。
    """
    # 第一层：按 Python 对象 id 过滤。
    original_ids = {id(m) for m in original_history}

    def _key(m) -> tuple:
        """Fallback：构造一条消息的稳定内容键。

        当对象被拷贝导致 id 不同时，用这个键来判断是不是同一条消息。
        """
        kind = "request" if isinstance(m, ModelRequest) else (
            "response" if isinstance(m, ModelResponse) else type(m).__name__
        )
        parts: list = []
        for part in getattr(m, "parts", ()):
            pk = getattr(part, "part_kind", None)
            if pk == "user-prompt":
                parts.append(("user-prompt", part.content))
            elif pk == "tool-return":
                parts.append(("tool-return", part.tool_name, part.tool_call_id, str(part.content)))
            elif pk == "text":
                parts.append(("text", part.content))
            elif pk == "tool-call":
                parts.append(("tool-call", part.tool_name, part.tool_call_id, str(part.args)))
            else:
                parts.append((str(pk), repr(part)))
        return (kind, getattr(m, "run_id", None), tuple(parts))

    original_keys = {_key(m) for m in original_history}
    return [
        m for m in all_messages
        if id(m) not in original_ids and _key(m) not in original_keys
    ]


async def notify_streamers(
    streamers: list, event: str, data: dict, pending: list
) -> None:
    """将运行时事件推给所有流式 Extension，收集它们 yield 的 chunks。

    流式 Extension 的 on_agent_runner_event_stream 方法是一个 async
    generator，它的 yield 值被追加到 pending 列表中，供 run_stream()
    本体在合适的时机 drain（yield 给外部消费者）。

    无 on_agent_runner_event_stream 的 Extension 被静默跳过。
    """
    for s in streamers:
        stream_fn = getattr(s, "on_agent_runner_event_stream", None)
        if stream_fn is None:
            continue
        try:
            async for chunk in stream_fn(event, data):
                pending.append(chunk)
        except Exception:  # pragma: no cover - fail-open
            pass


async def drain_pending(pending: list) -> AsyncIterator[dict]:
    """逐条 yield pending 中的所有 chunks，同时清空列表。

    用 async generator 实现，调用方用 async for 消费。每消费一条，
    列表就短一截，直到为空。
    """
    while pending:
        yield pending.pop(0)


def build_capabilities(self) -> list:
    """组装传给 Pydantic AI Agent 的 capabilities 列表。

    用户配置的 capabilities（从 AgentConfig 来）排前面，
    框架自己的 Hooks 在外面由调用方追加进列表。
    """
    capabilities = list(self._config.capabilities)
    if self._config.hooks is not None:
        capabilities.append(self._config.hooks)
    return capabilities


async def fire(self, event: str, data: dict) -> dict:
    """链式（chain）模式分发事件给所有 Extension。

    链式意味着：Extension A 收到数据 → 返回修改后的 dict →
    Extension B 收到 A 修改后的数据 → 返回再次修改后的 dict →
    最终结果返回给调用方。

    如果某个 Extension 崩溃，记录 warning 后继续下一个。
    """
    current = dict(data)
    for ext in self._extensions:
        try:
            r = await ext.on_agent_runner_event(event, current)
        except Exception as exc:  # pragma: no cover - fail-open
            self._on_warning(
                f"Extension {type(ext).__name__} handler for {event} failed: {exc}",
                exc,
            )
            continue
        if isinstance(r, dict):
            current.update(r)
    return current


async def fire_notify(
    self, event: str, data: dict, *, cancel_key: str = "cancel"
) -> dict:
    """通知（notify）模式分发事件给所有 Extension。

    与链式不同：所有 Extension 收到的是同一个只读快照，互相不看到
    彼此的修改。支持"取消"语义：只要有任意一个 Extension 返回
    {cancel_key: True}，最终结果就是 {cancel_key: True}。
    """
    snapshot = dict(data)
    cancelled = False
    for ext in self._extensions:
        try:
            r = await ext.on_agent_runner_event(event, snapshot)
        except Exception as exc:  # pragma: no cover - fail-open
            self._on_warning(
                f"Extension {type(ext).__name__} handler for {event} failed: {exc}",
                exc,
            )
            continue
        if isinstance(r, dict) and r.get(cancel_key):
            cancelled = True
    return {cancel_key: cancelled}


async def get_tools(self) -> list:
    """获取本轮 Agent 应使用的工具列表。

    优先走 ToolLifecycle（如果 Extension 注册了工具来源则走链式注册），
    没有 ToolLifecycle 时才回退到构造时传入的 raw tools。
    """
    lifecycle = await self._ensure_tool_lifecycle()
    if lifecycle is not None:
        return lifecycle.get_for_scope(self._scope)
    return self._raw_tools
