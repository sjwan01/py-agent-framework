"""ContextManager — watermark truncation and baseline freezing."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

from pydantic import BaseModel, Field
from pydantic_ai.messages import ModelRequest, ToolReturnPart, UserPromptPart

from py_agent.session._shared import _is_turn_start


class PreparedContext(BaseModel):
    """ContextManager.prepare() 的输出。"""

    messages: list = Field(default_factory=list)
    needs_compaction: bool = False
    tokens_used: int = 0


class BaselineState(BaseModel):
    """基线时刻的 skill/tool/context 快照，用于 diff 注入。"""

    skills: dict[str, str] = Field(default_factory=dict)
    tools: dict[str, str] = Field(default_factory=dict)
    context: list[str] = Field(default_factory=list)


class ContextManager:
    def __init__(
        self,
        *,
        low_watermark_ratio: float = 0.6,
        high_watermark_ratio: float = 0.75,
        protect_turns: int = 5,
        truncate_chars: int = 1_000,
        context_window_cap: int = 128_000,
    ):
        self._low = low_watermark_ratio
        self._high = high_watermark_ratio
        self._protect = protect_turns
        self._truncate_chars = truncate_chars
        self._context_window_cap = context_window_cap
        self._frozen_baseline: str | None = None
        self._baseline_state: BaselineState | None = None

    async def prepare(
        self, messages: list, *, system_prompt: str, current_state: BaselineState,
    ) -> PreparedContext:
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

        context_cap = self._context_window_cap
        low_mark = context_cap * self._low
        high_mark = context_cap * self._high

        total_tokens, boundary = _estimate_and_find_boundary(messages, self._protect)

        if total_tokens <= low_mark:
            return PreparedContext(
                messages=messages,
                needs_compaction=total_tokens > high_mark,
                tokens_used=total_tokens,
            )

        messages, tokens_after = _truncate_and_estimate(
            messages, boundary, self._truncate_chars
        )
        return PreparedContext(
            messages=messages,
            needs_compaction=tokens_after > high_mark,
            tokens_used=tokens_after,
        )


# ── 单次正向遍历：估算 + turn boundary ─────────────────────────────────
# 合并原来的 _default_estimate 和 _find_turn_boundary。
# 正向遍历一次，同时累计字符数和记录每个 user turn 的起始索引。

def _estimate_and_find_boundary(messages: list, protect: int) -> tuple[int, int]:
    """单次正向遍历，返回 (total_tokens, boundary_index)。

    total_tokens: 所有文本字符数 ÷ 4 的粗略估算。
    boundary_index: 第 protect 个 user turn（从末尾数）的起始索引。
                    此索引之前的消息可以被截断。
    """
    total_chars = 0
    turn_starts: list[int] = []

    for i, msg in enumerate(messages):
        if _is_turn_start(msg):
            turn_starts.append(i)
        for part in getattr(msg, "parts", ()):
            content = getattr(part, "content", None)
            if isinstance(content, str):
                total_chars += len(content)

    tokens = max(total_chars // 4, 1)

    if len(turn_starts) >= protect:
        boundary = turn_starts[-protect]
    else:
        boundary = 0

    return tokens, boundary


# ── 单次正向遍历：截断 + 重新估算 ──────────────────────────────────────
# 合并原来的 _truncate_old_tool_results 和第二次 _estimate。
# 一边截断 boundary 之前的旧 tool results，一边累加截断后的字符数。

def _truncate_and_estimate(
    messages: list, boundary: int, max_chars: int
) -> tuple[list, int]:
    """截断旧 tool results 并返回 (new_messages, tokens_after)。

    不修改原始 messages。只有精确的 ``ToolReturnPart``（非子类）
    且 content 是字符串且长度超过 max_chars 才会被截断。
    """
    out: list = []
    total_chars = 0

    for i, msg in enumerate(messages):
        # 不在截断范围的 ModelRequest 需要遍历 parts 统计 token。
        if not isinstance(msg, ModelRequest) or i >= boundary:
            out.append(msg)
            for part in getattr(msg, "parts", ()):
                content = getattr(part, "content", None)
                if isinstance(content, str):
                    total_chars += len(content)
            continue

        new_parts: list = []
        changed = False
        for part in msg.parts:
            if (
                type(part) is ToolReturnPart
                and isinstance(part.content, str)
                and len(part.content) > max_chars
            ):
                truncated_content = part.content[:max_chars]
                new_parts.append(replace(part, content=truncated_content))
                changed = True
                total_chars += len(truncated_content)
            else:
                new_parts.append(part)
                content = getattr(part, "content", None)
                if isinstance(content, str):
                    total_chars += len(content)

        if changed:
            out.append(replace(msg, parts=new_parts))
        else:
            out.append(msg)

    return out, max(total_chars // 4, 1)


def _compute_diff(baseline: BaselineState, current: BaselineState) -> str:
    lines: list[str] = []

    for name, desc in current.skills.items():
        if name not in baseline.skills:
            lines.append(f'  Added skill "{name}": "{desc}"')
        elif baseline.skills[name] != desc:
            lines.append(f'  Updated skill "{name}": "{desc}"')
            # Note: old description available as baseline.skills[name] if needed
    for name in baseline.skills:
        if name not in current.skills:
            lines.append(f'  Removed skill "{name}": "{baseline.skills[name]}"')

    for name, desc in current.tools.items():
        if name not in baseline.tools:
            lines.append(f'  Added tool "{name}": "{desc}"')
        elif baseline.tools[name] != desc:
            lines.append(f'  Updated tool "{name}": "{desc}"')
            # Note: old description available as baseline.tools[name] if needed
    for name in baseline.tools:
        if name not in current.tools:
            lines.append(f'  Removed tool "{name}": "{baseline.tools[name]}"')

    baseline_ctx = set(baseline.context)
    current_ctx = set(current.context)
    for item in sorted(current_ctx - baseline_ctx):
        lines.append(f'  Added context "{item}"')
    for item in sorted(baseline_ctx - current_ctx):
        lines.append(f'  Removed context "{item}"')

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
