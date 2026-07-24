"""ContextManager — watermark truncation and baseline freezing."""
from __future__ import annotations

from datetime import datetime, timezone

from pydantic_ai.messages import ModelRequest, UserPromptPart

from agent_framework.models import PreparedContext, BaselineState


async def _clear_tool_results(messages: list, keep_pairs: int) -> list:
    """Use Harness ClearToolResults to blank old tool results.

    This keeps tool-call / tool-return pairing intact while reclaiming tokens
    from older tool outputs. ``keep_pairs`` is aligned with V2's ``protect_turns``
    concept: the most recent N matched pairs are preserved.
    """
    from pydantic_ai_harness.compaction import ClearToolResults

    # ``max_tokens=1`` forces the strategy to compact every time it is invoked;
    # V2 already gates invocation on its own watermark logic.
    strategy = ClearToolResults(keep_pairs=keep_pairs, max_tokens=1)
    return await strategy.compact(messages, None)  # type: ignore[arg-type]


class ContextManager:
    def __init__(
        self,
        *,
        low_watermark_ratio: float = 0.6,
        high_watermark_ratio: float = 0.75,
        protect_turns: int = 5,
        token_estimator=None,
        context_window_cap: int | None = None,
    ):
        self._low = low_watermark_ratio
        self._high = high_watermark_ratio
        self._protect = protect_turns
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
            baseline = self._baseline_state or BaselineState()
            diff = _compute_diff(baseline, current_state)
            if diff:
                messages = _inject_diff(messages, diff)

        tokens = self._estimate(messages)
        context_cap = self._context_window_cap or 128_000
        low_mark = context_cap * self._low
        high_mark = context_cap * self._high

        if tokens > low_mark:
            messages = await _clear_tool_results(messages, self._protect)

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
