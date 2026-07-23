"""ContextManager — watermark truncation and baseline freezing."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json

from pydantic_ai.messages import ModelRequest, UserPromptPart

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
    ):
        self._low = low_watermark_ratio
        self._high = high_watermark_ratio
        self._protect = protect_turns
        self._truncate_chars = truncate_chars
        self._estimate = token_estimator or _default_estimate
        self._frozen_baseline: str | None = None
        self._baseline_state: BaselineState | None = None
        self._needs_refresh = False

    async def prepare(
        self, messages: list, *, system_prompt: str, current_state: BaselineState,
    ) -> PreparedContext:
        if self._needs_refresh or self._frozen_baseline is None:
            self._frozen_baseline = system_prompt
            self._baseline_state = current_state
            self._needs_refresh = False
        else:
            diff = _compute_diff(self._baseline_state, current_state)
            if diff:
                messages = _inject_diff(messages, diff)

        tokens = self._estimate(messages)
        context_cap = max(tokens, 128_000)
        low_mark = context_cap * self._low
        high_mark = context_cap * self._high

        if tokens > low_mark:
            boundary = _find_turn_boundary(messages, self._protect)
            _truncate_old_tool_results(messages, boundary, self._truncate_chars)

        tokens_after = self._estimate(messages)
        return PreparedContext(
            messages=messages,
            needs_compaction=tokens_after > high_mark,
            tokens_used=tokens_after,
        )


def _default_estimate(messages: list) -> int:
    return len(json.dumps([asdict(m) for m in messages], default=str)) // 4


def _find_turn_boundary(messages: list, protect: int) -> int:
    turns = 0
    for i in range(len(messages) - 1, -1, -1):
        if _is_turn_start(messages[i]):
            turns += 1
            if turns >= protect:
                return i
    return 0


def _is_turn_start(msg) -> bool:
    if isinstance(msg, ModelRequest):
        return bool(msg.parts) and isinstance(msg.parts[0], UserPromptPart)
    return False


def _truncate_old_tool_results(messages: list, boundary: int, max_chars: int) -> None:
    for i, msg in enumerate(messages[:boundary]):
        if not isinstance(msg, ModelRequest):
            continue
        new_parts = []
        for part in msg.parts:
            if hasattr(part, "tool_name") and hasattr(part, "content"):
                if isinstance(part.content, str) and len(part.content) > max_chars:
                    part = type(part)(
                        tool_name=part.tool_name,
                        content=part.content[:max_chars],
                        tool_call_id=part.tool_call_id,
                        timestamp=part.timestamp,
                    )
            new_parts.append(part)
        messages[i] = ModelRequest(
            parts=new_parts, kind=msg.kind, run_id=msg.run_id,
            conversation_id=msg.conversation_id, instructions=msg.instructions,
            timestamp=msg.timestamp,
        )


def _compute_diff(baseline: BaselineState, current: BaselineState) -> str:
    lines = []
    for name in current.skills:
        if name not in baseline.skills:
            lines.append(f'  Added skill "{name}"')
    for name in current.tools:
        if name not in baseline.tools:
            lines.append(f'  Added tool "{name}"')
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
