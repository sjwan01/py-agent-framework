"""Context preparation — watermark truncation."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from pydantic_ai.messages import ModelRequest, ToolReturnPart

from py_agent.models import ContextConfig
from py_agent.session._shared import _is_turn_start


async def _prepare_context(
    messages: list[Any],
    *,
    config: ContextConfig,
) -> tuple[list[Any], bool]:
    """Prepare messages for the agent run.

    Applies watermark truncation to old tool results outside the protected
    turn region. The returned message list is a copy so the caller's list is
    not mutated.

    Args:
        messages: Raw message history loaded from the session backend.
        config: Immutable truncation / watermark configuration.

    Returns:
        A tuple of ``(prepared_messages, needs_compaction)``.
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
        # Below the low watermark: nothing to truncate, no compaction needed.
        return messages, False

    messages, tokens_after = _truncate_and_estimate(
        messages, boundary, config.truncate_chars
    )
    return messages, tokens_after > high_mark


# Single forward pass: estimate tokens and find the turn boundary.
# Merges the former _default_estimate and _find_turn_boundary helpers.
# Walks forward once, accumulating per-content token estimates and recording
# each user turn start.


def _estimate_tokens(text: str) -> int:
    """Estimate the token count of ``text`` with script-aware weighting.

    CJK characters cost roughly one token each; other characters roughly
    four per token. Without the CJK weighting, a Chinese conversation would
    be undercounted by up to 4x and watermark truncation would trigger far
    too late. This is a heuristic, not a tokenizer.

    Args:
        text: The string content to estimate.

    Returns:
        The estimated token count, or 0 for empty text.
    """
    if not text:
        return 0
    cjk = 0
    for ch in text:
        if (
            "\u3400" <= ch <= "\u4dbf"  # CJK Extension A
            or "\u4e00" <= ch <= "\u9fff"  # CJK Unified Ideographs
            or "\uff00" <= ch <= "\uffef"  # Fullwidth forms
        ):
            cjk += 1
    return cjk + (len(text) - cjk) // 4


def _estimate_and_find_boundary(
    messages: list[Any], protect: int
) -> tuple[int, int]:
    """Return a rough token estimate and the truncation boundary.

    ``total_tokens`` is estimated per content string via ``_estimate_tokens``.
    ``boundary`` is the start index of the ``protect``-th user turn from the
    end; messages before ``boundary`` may be truncated. ``protect <= 0`` means
    nothing is protected, so every message may be truncated.
    """
    total_tokens = 0
    turn_starts: list[int] = []

    for i, msg in enumerate(messages):
        if _is_turn_start(msg):
            turn_starts.append(i)
        for part in getattr(msg, "parts", ()):
            content = getattr(part, "content", None)
            if isinstance(content, str):
                total_tokens += _estimate_tokens(content)

    tokens = max(total_tokens, 1)

    if protect <= 0:
        # protect=0 means "protect nothing": every message may be truncated.
        # (turn_starts[-0] would resolve to turn_starts[0], which would
        # protect everything — the opposite of the intent.)
        boundary = len(messages)
    elif len(turn_starts) >= protect:
        boundary = turn_starts[-protect]
    else:
        # fewer turns than protect_turns: fall back to full history, no truncation
        boundary = 0

    return tokens, boundary


# Single forward pass: truncate old tool results and re-estimate.
# Merges the former _truncate_old_tool_results and the second estimation pass.
# Truncates old tool results before the boundary while accumulating the
# estimated token count of every content string.


def _truncate_content(content: str, max_chars: int) -> str:
    """Truncate ``content`` and mark how many characters were removed.

    The cut point falls back to the last complete line when possible, so
    multi-line JSON results end on a field boundary. The appended marker
    reports the removed character count so the model knows the result is
    incomplete instead of mistaking a partial value for the full one.

    Args:
        content: The tool result content to truncate.
        max_chars: Maximum characters to keep before the marker.

    Returns:
        The truncated content plus a ``...[truncated: N chars]`` marker.
    """
    cut = content.rfind("\n", 0, max_chars)
    head = content[:cut] if cut > 0 else content[:max_chars]
    removed = len(content) - len(head)
    return f"{head}\n...[truncated: {removed} chars]"


def _truncate_and_estimate(
    messages: list[Any], boundary: int, max_chars: int
) -> tuple[list[Any], int]:
    """Truncate old tool results and return the updated messages plus token count.

    Does not mutate the original ``messages``. Only exact ``ToolReturnPart``
    instances (not subclasses) whose ``content`` is a string longer than
    ``max_chars`` are truncated via ``_truncate_content``.
    """
    out: list[Any] = []
    total_tokens = 0

    for i, msg in enumerate(messages):
        # messages outside the truncation range still need their parts counted for tokens
        if not isinstance(msg, ModelRequest) or i >= boundary:
            out.append(msg)
            for part in getattr(msg, "parts", ()):
                content = getattr(part, "content", None)
                if isinstance(content, str):
                    total_tokens += _estimate_tokens(content)
            continue

        new_parts: list[Any] = []
        changed = False
        for part in msg.parts:
            if (
                type(part) is ToolReturnPart
                and isinstance(part.content, str)
                and len(part.content) > max_chars
            ):
                truncated_content = _truncate_content(part.content, max_chars)
                new_parts.append(replace(part, content=truncated_content))
                changed = True
                total_tokens += _estimate_tokens(truncated_content)
            else:
                new_parts.append(part)
                content = getattr(part, "content", None)
                if isinstance(content, str):
                    total_tokens += _estimate_tokens(content)

        if changed:
            out.append(replace(msg, parts=new_parts))
        else:
            out.append(msg)

    return out, max(total_tokens, 1)
