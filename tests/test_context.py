"""Tests for context preparation — watermark truncation and token estimation.

Covers the pure functions in ``py_agent._context``: script-aware token
estimation (``_estimate_tokens``) and watermark truncation
(``_prepare_context``).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from pydantic_ai.messages import (
    ModelRequest,
    ToolReturnPart,
    UserPromptPart,
)

from py_agent._context import _estimate_tokens, _prepare_context
from py_agent.models import ContextConfig


def _make_turn(user_text: str, tool_content: str) -> list[ModelRequest]:
    """Build one user turn: a user prompt followed by a tool result."""
    ts = datetime.now(timezone.utc)
    user = ModelRequest(
        parts=[UserPromptPart(content=user_text, timestamp=ts)],
        kind="request",
        timestamp=ts,
    )
    tool = ModelRequest(
        parts=[ToolReturnPart(tool_name="t", tool_call_id="1", content=tool_content)],
        kind="request",
        timestamp=ts,
    )
    return [user, tool]


def _tool_contents(messages: list[ModelRequest]) -> list[str]:
    """Extract the tool return contents, one per message in order."""
    return [
        str(part.content)
        for msg in messages
        for part in getattr(msg, "parts", ())
        if type(part) is ToolReturnPart
    ]


class TestEstimateTokens:
    """Script-aware token estimation."""

    def test_empty_string_is_zero(self) -> None:
        """Empty text contributes no tokens."""
        assert _estimate_tokens("") == 0

    def test_ascii_four_chars_per_token(self) -> None:
        """ASCII text is estimated at roughly four characters per token."""
        assert _estimate_tokens("a" * 400) == 100

    def test_cjk_one_token_per_char(self) -> None:
        """CJK characters cost roughly one token each, not one per four chars."""
        assert _estimate_tokens("汉" * 100) == 100

    def test_mixed_script_weighting(self) -> None:
        """CJK and ASCII parts are weighted independently."""
        assert _estimate_tokens("汉" * 10 + "a" * 40) == 20


class TestPrepareContext:
    """Watermark truncation behavior."""

    def _six_turn_history(self) -> list[ModelRequest]:
        """Six turns of (user, 2000-char tool result) pairs."""
        messages: list[ModelRequest] = []
        for i in range(6):
            messages.extend(_make_turn(f"u{i}", "x" * 2000))
        return messages

    async def test_below_low_watermark_untouched(self) -> None:
        """Below the low watermark: no truncation, no compaction flag."""
        config = ContextConfig(
            context_window_cap=4_000,
            low_watermark_ratio=0.5,
            high_watermark_ratio=0.75,
            protect_turns=2,
            truncate_chars=100,
        )
        messages = _make_turn("u0", "x" * 200)  # 50 tokens, far below the cap

        prepared, needs_compaction = await _prepare_context(messages, config=config)

        assert _tool_contents(prepared) == ["x" * 200]
        assert needs_compaction is False

    async def test_truncates_old_tool_results_but_protects_recent_turns(self) -> None:
        """Old tool results are truncated; the last two turns stay intact."""
        config = ContextConfig(
            context_window_cap=2_000,
            low_watermark_ratio=0.5,
            high_watermark_ratio=0.75,
            protect_turns=2,
            truncate_chars=100,
        )
        messages = self._six_turn_history()  # 6 × 500 tokens = 3000 > low mark

        prepared, needs_compaction = await _prepare_context(messages, config=config)

        # turns 0-3 truncated (100 chars + cut marker); turns 4-5 intact
        contents = _tool_contents(prepared)
        assert all(c.startswith("x" * 100) for c in contents[:4])
        assert all("[truncated: 1900 chars]" in c for c in contents[:4])
        assert contents[4:] == ["x" * 2000] * 2
        # reclaimed space keeps us below the high watermark
        assert needs_compaction is False

    async def test_multiline_json_cut_falls_back_to_complete_line(self) -> None:
        """Multi-line JSON results are cut at a line boundary and marked."""
        content = json.dumps(
            {"items": [{"id": i, "name": f"item-{i}"} for i in range(50)]},
            indent=2,
        )
        config = ContextConfig(
            context_window_cap=1_500,
            low_watermark_ratio=0.5,
            high_watermark_ratio=0.75,
            protect_turns=1,
            truncate_chars=200,
        )
        # turn 0 carries the JSON tool result; turn 1 is protected
        messages = _make_turn("u0", content) + _make_turn("u1", "y" * 2_000)

        prepared, _ = await _prepare_context(messages, config=config)

        truncated = _tool_contents(prepared)[0]
        cut = content.rfind("\n", 0, 200)
        expected = content[:cut] + f"\n...[truncated: {len(content) - cut} chars]"
        assert truncated == expected
        # turn 1's result is untouched
        assert _tool_contents(prepared)[1] == "y" * 2_000

    async def test_above_high_watermark_flags_compaction(self) -> None:
        """When truncation alone cannot stay under the high mark, flag compaction."""
        config = ContextConfig(
            context_window_cap=1_200,
            low_watermark_ratio=0.5,
            high_watermark_ratio=0.75,
            protect_turns=2,
            truncate_chars=100,
        )
        messages = self._six_turn_history()

        _, needs_compaction = await _prepare_context(messages, config=config)

        # 4 × 25 tokens (truncated) + 2 × 500 tokens (protected) = 1100 > 900
        assert needs_compaction is True

    async def test_original_messages_not_mutated(self) -> None:
        """The caller's message list is untouched by preparation."""
        config = ContextConfig(
            context_window_cap=2_000,
            low_watermark_ratio=0.5,
            high_watermark_ratio=0.75,
            protect_turns=2,
            truncate_chars=100,
        )
        messages = self._six_turn_history()

        await _prepare_context(messages, config=config)

        assert _tool_contents(messages) == ["x" * 2000] * 6

    async def test_protect_zero_truncates_everything(self) -> None:
        """protect_turns=0 protects nothing: all tool results truncate."""
        config = ContextConfig(
            context_window_cap=2_000,
            low_watermark_ratio=0.5,
            high_watermark_ratio=0.75,
            protect_turns=0,
            truncate_chars=100,
        )
        messages = self._six_turn_history()

        prepared, _ = await _prepare_context(messages, config=config)

        contents = _tool_contents(prepared)
        assert all("[truncated: 1900 chars]" in c for c in contents)

    async def test_history_without_user_turns_does_not_crash(self) -> None:
        """A history with no user turns stays intact instead of raising."""
        ts = datetime.now(timezone.utc)
        tool_only = [
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="t", tool_call_id="1", content="x" * 2_000
                    )
                ],
                kind="request",
                timestamp=ts,
            )
        ]
        config = ContextConfig(
            context_window_cap=2_000,
            low_watermark_ratio=0.5,
            high_watermark_ratio=0.75,
            protect_turns=5,
            truncate_chars=100,
        )

        prepared, _ = await _prepare_context(tool_only, config=config)

        # no turn start to anchor protection → conservative: nothing truncated
        assert _tool_contents(prepared) == ["x" * 2_000]
