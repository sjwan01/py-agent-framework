"""ContextManager — watermark truncation and baseline freezing."""
from __future__ import annotations

from datetime import datetime, timezone

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
        # Work on a copy so the caller's list is not mutated and prepare is idempotent.
        messages = list(messages)

        if self._frozen_baseline is None:
            self._frozen_baseline = system_prompt
            self._baseline_state = current_state
        else:
            diff = _compute_diff(self._baseline_state, current_state)
            if diff:
                messages = _inject_diff(messages, diff)

        tokens = self._estimate(messages)
        context_cap = self._context_window_cap or 128_000
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

    def find_compaction_boundary(self, messages: list) -> int:
        """Return the index of the first message that should be kept intact.

        Messages before this index are candidates for compaction.
        """
        return _find_turn_boundary(messages, self._protect)


def _default_estimate(messages: list) -> int:
    """Default token estimator: characters // 4."""
    total = 0
    for msg in messages:
        for part in getattr(msg, "parts", ()):
            content = getattr(part, "content", None)
            if isinstance(content, str):
                total += len(content)
    return max(total // 4, 1)


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
        changed = False
        for part in msg.parts:
            if hasattr(part, "tool_name") and hasattr(part, "content"):
                if isinstance(part.content, str) and len(part.content) > max_chars:
                    part = type(part)(
                        tool_name=part.tool_name,
                        content=part.content[:max_chars],
                        tool_call_id=part.tool_call_id,
                        timestamp=part.timestamp,
                    )
                    changed = True
            new_parts.append(part)
        if changed:
            messages[i] = ModelRequest(
                parts=new_parts, kind=msg.kind, run_id=msg.run_id,
                conversation_id=msg.conversation_id, instructions=msg.instructions,
                timestamp=msg.timestamp,
            )


def _compute_diff(baseline: BaselineState, current: BaselineState) -> str:
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
