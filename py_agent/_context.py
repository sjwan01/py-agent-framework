"""Context preparation — watermark truncation."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from pydantic_ai.messages import ModelRequest, ToolReturnPart

from py_agent.models import ContextConfig, PreparedContext
from py_agent.session._shared import _is_turn_start


async def _prepare_context(
    messages: list[Any],
    *,
    config: ContextConfig,
) -> PreparedContext:
    """Prepare messages for the agent run.

    Applies watermark truncation to old tool results outside the protected
    turn region. The returned message list is a copy so the caller's list is
    not mutated.

    Args:
        messages: Raw message history loaded from the session backend.
        config: Immutable truncation / watermark configuration.

    Returns:
        Prepared context with truncated tool results and a compaction flag.
    """
    # Work on a copy so the caller's list is not mutated and prepare stays idempotent.
    messages = list(messages)

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
