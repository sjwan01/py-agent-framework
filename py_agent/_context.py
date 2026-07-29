"""Context preparation — watermark truncation and baseline diff injection."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolReturnPart,
)

from py_agent.models import BaselineState, ContextConfig, PreparedContext
from py_agent.session._shared import _is_turn_start


async def prepare(
    messages: list[Any],
    *,
    frozen_baseline: BaselineState | None,
    current_state: BaselineState,
    config: ContextConfig,
) -> PreparedContext:
    """Prepare messages for the agent run.

    Injects transient diff messages when the current state differs from the
    frozen baseline, then applies watermark truncation. Diff messages are not
    persisted; they live only in the returned message list.

    Args:
        messages: Raw message history loaded from the session backend.
        frozen_baseline: Last persisted baseline state, if any.
        current_state: State built from the current AgentRunner configuration.
        config: Immutable truncation / watermark configuration.

    Returns:
        Prepared context with possibly injected diff messages and a compaction flag.
    """
    # Work on a copy so the caller's list is not mutated and prepare stays idempotent.
    messages = list(messages)

    if frozen_baseline is not None:
        diff = _compute_diff(frozen_baseline, current_state)
        if diff:
            messages = _inject_diff(messages, diff)

    context_cap = config.context_window_cap
    low_mark = context_cap * config.low_watermark_ratio
    high_mark = context_cap * config.high_watermark_ratio

    total_tokens, boundary = _estimate_and_find_boundary(
        messages, config.protect_turns
    )

    if total_tokens <= low_mark:
        return PreparedContext(
            messages=messages,
            needs_compaction=total_tokens > high_mark,
            tokens_used=total_tokens,
        )

    messages, tokens_after = _truncate_and_estimate(
        messages, boundary, config.truncate_chars
    )
    return PreparedContext(
        messages=messages,
        needs_compaction=tokens_after > high_mark,
        tokens_used=tokens_after,
    )


# Single forward pass: estimate tokens and find the turn boundary.
# Merges the former _default_estimate and _find_turn_boundary helpers.
# Walks forward once, accumulating characters and recording each user turn start.


def _estimate_and_find_boundary(
    messages: list[Any], protect: int
) -> tuple[int, int]:
    """Return a rough token estimate and the truncation boundary.

    ``total_tokens`` is a rough estimate (total characters divided by 4).
    ``boundary`` is the start index of the ``protect``-th user turn from the end.
    Messages before ``boundary`` may be truncated.
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


# Single forward pass: truncate old tool results and re-estimate.
# Merges the former _truncate_old_tool_results and the second estimation pass.
# Truncates old tool results before the boundary while accumulating the truncated character count.


def _truncate_and_estimate(
    messages: list[Any], boundary: int, max_chars: int
) -> tuple[list[Any], int]:
    """Truncate old tool results and return the updated messages plus token count.

    Does not mutate the original ``messages``. Only exact ``ToolReturnPart``
    instances (not subclasses) whose ``content`` is a string longer than
    ``max_chars`` are truncated.
    """
    out: list[Any] = []
    total_chars = 0

    for i, msg in enumerate(messages):
        # messages outside the truncation range still need their parts counted for tokens
        if not isinstance(msg, ModelRequest) or i >= boundary:
            out.append(msg)
            for part in getattr(msg, "parts", ()):
                content = getattr(part, "content", None)
                if isinstance(content, str):
                    total_chars += len(content)
            continue

        new_parts: list[Any] = []
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
    """Compute a human-readable diff between two baseline states."""
    lines: list[str] = []

    for name, desc in current.skills.items():
        if name not in baseline.skills:
            lines.append(f'  Added skill "{name}": "{desc}"')
        elif baseline.skills[name] != desc:
            lines.append(f'  Updated skill "{name}": "{desc}"')
            # old description available as baseline.skills[name] if needed
    for name in baseline.skills:
        if name not in current.skills:
            lines.append(f'  Removed skill "{name}": "{baseline.skills[name]}"')

    for name, desc in current.tools.items():
        if name not in baseline.tools:
            lines.append(f'  Added tool "{name}": "{desc}"')
        elif baseline.tools[name] != desc:
            lines.append(f'  Updated tool "{name}": "{desc}"')
            # old description available as baseline.tools[name] if needed
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


def _inject_diff(messages: list[Any], diff: str) -> list[Any]:
    """Insert a transient diff message before the latest turn start.

    Uses a ``ModelResponse`` with a ``TextPart`` rather than a user request.
    Pydantic AI merges consecutive ``UserPromptPart`` messages into a single
    ``ModelRequest``, which forces the diff to be re-persisted; a response
    message stays separate and is excluded from persistence because it is
    captured in ``original_history`` after context preparation.
    """
    ts = datetime.now(timezone.utc)
    for i in range(len(messages) - 1, -1, -1):
        if _is_turn_start(messages[i]):
            messages.insert(
                i,
                ModelResponse(
                    parts=[
                        TextPart(
                            content=f"[System config changed]\n{diff}",
                            part_kind="text",
                        )
                    ],
                    kind="response",
                    timestamp=ts,
                ),
            )
            break
    return messages
