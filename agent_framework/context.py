"""ContextManager — watermark truncation and baseline freezing."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

from pydantic_ai.messages import ModelRequest, ToolReturnPart, UserPromptPart

from agent_framework.models import PreparedContext, BaselineState


class ContextManager:
    def __init__(
        self,
        *,
        low_watermark_ratio: float = 0.6,
        high_watermark_ratio: float = 0.75,
        protect_turns: int = 5,
        truncate_chars: int = 1_000,
        token_estimator=None,
        context_window_cap: int | None = None,
    ):
        self._low = low_watermark_ratio
        self._high = high_watermark_ratio
        self._protect = protect_turns
        self._truncate_chars = truncate_chars
        self._estimate = token_estimator or _default_estimate
        self._context_window_cap = context_window_cap
        self._frozen_baseline: str | None = None
        self._baseline_state: BaselineState | None = None

    async def prepare(
        self, messages: list, *, system_prompt: str, current_state: BaselineState,
    ) -> PreparedContext:
        # TODO: prepare 内部多次遍历 messages（估算、找 boundary、截断、再估算）。
        #       当历史很长时这是 O(n) 甚至 O(n*m) 的开销。可考虑：
        #       1) 单次遍历同时完成估算、boundary 标记和截断；
        #       2) 使用增量/缓存结构避免每轮重新扫描全部历史。

        # Work on a copy so the caller's list is not mutated and prepare is idempotent.
        messages = list(messages)

        if self._frozen_baseline is None:
            self._frozen_baseline = system_prompt
            self._baseline_state = current_state
        else:
            baseline = self._baseline_state or BaselineState()
            diff = _compute_diff(baseline, current_state)
            if diff:
                messages = _inject_diff(messages, diff)

        tokens = self._estimate(messages)
        context_cap = self._context_window_cap or 128_000
        low_mark = context_cap * self._low
        high_mark = context_cap * self._high

        if tokens > low_mark:
            boundary = _find_turn_boundary(messages, self._protect)
            messages = _truncate_old_tool_results(messages, boundary, self._truncate_chars)

        tokens_after = self._estimate(messages)
        return PreparedContext(
            messages=messages,
            needs_compaction=tokens_after > high_mark,
            tokens_used=tokens_after,
        )

    def find_compaction_boundary(self, messages: list) -> int:
        """Return the index of the first message that should be kept intact.

        Messages before this index are candidates for compaction.
        """
        return _find_turn_boundary(messages, self._protect)


# ── token 估算 ────────────────────────────────────────────────────────
# 把消息列表里所有文本内容的字符数加起来，除以 4 作为 token 估算。
# 这是一个非常粗糙的启发式方法（~4 chars/token），
# 准确 token 计数需要 tiktoken 或类似分词器。

def _default_estimate(messages: list) -> int:
    """粗略 token 估算：所有文本字符总数 ÷ 4，最少返回 1。"""
    total = 0
    # 遍历每条消息的每个 part，只统计字符串类型的 content。
    for msg in messages:
        for part in getattr(msg, "parts", ()):
            content = getattr(part, "content", None)
            if isinstance(content, str):
                total += len(content)
    # 至少返回 1，避免下游除以 0 之类的边界情况。
    return max(total // 4, 1)


# ── turn 边界查找 ─────────────────────────────────────────────────────
# 从消息列表末尾往前数 protect 个用户 turn，返回被保护范围之外的
# 第一个消息索引。这个索引之前的消息是"旧消息"，可以被截断或压缩。

def _find_turn_boundary(messages: list, protect: int) -> int:
    """从后往前找到第 protect 个用户 turn 的起始位置。

    返回值是保护范围之外的第一条消息的索引：
    - messages[boundary:] → 受保护的最近 protect 个 turn
    - messages[:boundary] → 可以被截断/压缩的旧消息
    """
    turns = 0
    # 从列表末尾往前扫描，每次遇到用户 turn 开头就计数 +1。
    for i in range(len(messages) - 1, -1, -1):
        if _is_turn_start(messages[i]):
            turns += 1
            # 数够了 protect 个 turn，返回当前位置作为"保护边界"。
            if turns >= protect:
                return i
    # 消息总数少于 protect 个 turn，返回 0 表示全部保护。
    return 0


def _is_turn_start(msg) -> bool:
    """判断一条消息是不是用户 turn 的起点。

    规则：ModelRequest 且第一个 part 是 UserPromptPart。
    这约等于"用户发了一条新消息"。
    """
    if isinstance(msg, ModelRequest):
        # parts 非空 且 第一个 part 是 UserPromptPart。
        return bool(msg.parts) and isinstance(msg.parts[0], UserPromptPart)
    return False


# ── 工具结果截断 ─────────────────────────────────────────────────────
# 把 boundary 之前的旧 tool result 内容截断到 max_chars 个字符。
# 只截普通的 ToolReturnPart，不动框架 typed 子类（如 ToolSearchReturnPart），
# 避免破坏结构性数据。tool-call / tool-result 的配对关系保持不变。

def _truncate_old_tool_results(messages: list, boundary: int, max_chars: int) -> list:
    """把 boundary 之前的 tool result 截断到 max_chars 字符，返回新列表。

    不修改原始 messages。只有精确的 ``ToolReturnPart``（非子类）
    且 content 是字符串且长度超过 max_chars 才会被截断。
    """
    out: list = []
    for i, msg in enumerate(messages):
        # 不是 ModelRequest 或在保护范围内的消息直接保留，不动。
        if not isinstance(msg, ModelRequest) or i >= boundary:
            out.append(msg)
            continue

        # 遍历消息的 parts，检查是否有需要截断的 tool result。
        new_parts: list = []
        changed = False
        for part in msg.parts:
            # 三个条件全部满足才截断：
            #   1. type(part) is ToolReturnPart — 精确类型，不碰子类
            #   2. content 是字符串
            #   3. 长度超过 max_chars
            if (
                type(part) is ToolReturnPart
                and isinstance(part.content, str)
                and len(part.content) > max_chars
            ):
                # 用 replace 重建 part（不可变风格），保留前 max_chars 个字符。
                new_parts.append(replace(part, content=part.content[:max_chars]))
                changed = True
            else:
                new_parts.append(part)

        # 如果这个消息的 parts 有变化，用 replace 重建整个消息。
        if changed:
            out.append(replace(msg, parts=new_parts))
        else:
            out.append(msg)

    return out


def _compute_diff(baseline: BaselineState, current: BaselineState) -> str:
    # TODO: diff 消息只列了 skill/tool 名称，没显示 description，也没处理
    #       工具/技能被删除的情况。pi-agent 的 diff 消息会包含描述，并明确
    #       标出 Removed/Added/Updated。应改为结构化 diff，包含描述和删除事件。
    lines = []
    for name, desc in current.skills.items():
        if name not in baseline.skills:
            lines.append(f'  Added skill "{name}"')
        elif baseline.skills[name] != desc:
            lines.append(f'  Updated skill "{name}"')
    for name, desc in current.tools.items():
        if name not in baseline.tools:
            lines.append(f'  Added tool "{name}"')
        elif baseline.tools[name] != desc:
            lines.append(f'  Updated tool "{name}"')
    return "\n".join(lines)


def _inject_diff(messages: list, diff: str) -> list:
    ts = datetime.now(timezone.utc)
    for i in range(len(messages) - 1, -1, -1):
        if _is_turn_start(messages[i]):
            messages.insert(i, ModelRequest(
                parts=[UserPromptPart(content=f"[System config changed]\n{diff}", timestamp=ts)],
                kind="request", timestamp=ts,
            ))
            break
    return messages
