"""Regression tests for session compaction loading.

These tests verify that ``LocalSessionManager.load_history`` selects the
correct compaction boundary instead of falling back to the full message list.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)

from py_agent.session import LocalSessionManager
from py_agent.session._shared import _infer_role, _is_turn_start
from py_agent.types import MessageRole


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    """Return a temporary on-disk SQLite path.

    ``:memory:`` cannot be used because aiosqlite creates a new in-memory
    database for every connection, while ``LocalSessionManager`` only runs
    schema creation on the first connect.
    """
    return str(tmp_path / "test.db")


def _make_turn(user_index: int) -> list[ModelMessage]:
    """Create one user/assistant turn as a list of two messages."""
    ts = datetime.now(timezone.utc)
    user_msg = ModelRequest(
        parts=[UserPromptPart(content=f"user {user_index}", timestamp=ts)],
        kind="request",
        timestamp=ts,
    )
    assistant_msg = ModelResponse(
        parts=[TextPart(content=f"assistant {user_index}", part_kind="text")],
        kind="response",
        timestamp=ts,
    )
    return [user_msg, assistant_msg]


class TestCompactionLoading:
    """Regression tests for compaction boundary selection."""

    async def test_load_history_without_compaction_returns_all_messages(self, db_path: str) -> None:
        """No compaction record exists → load the entire history."""
        mgr = LocalSessionManager(db_path=db_path)
        sid = await mgr.create_session()
        messages = []
        for i in range(3):
            messages.extend(_make_turn(i))
        await mgr.save_messages(sid, messages)

        loaded = await mgr.load_history(sid, protect_turns=2)

        assert len(loaded) == 6

    async def test_protect_turns_zero_uses_latest_compaction(self, db_path: str) -> None:
        """protect_turns=0 always selects the latest compaction boundary."""
        mgr = LocalSessionManager(db_path=db_path)
        sid = await mgr.create_session()
        messages = []
        for i in range(3):
            messages.extend(_make_turn(i))
        await mgr.save_messages(sid, messages)
        await mgr.apply_compaction(sid, "latest summary", boundary_seq=5)

        loaded = await mgr.load_history(sid, protect_turns=0)

        assert len(loaded) == 1  # summary only, no messages after boundary=5
        assert "latest summary" in loaded[0].parts[0].content

    async def test_latest_compaction_too_recent_falls_back_to_older(self, db_path: str) -> None:
        """If the latest compaction is inside the protected region, use an older one."""
        mgr = LocalSessionManager(db_path=db_path)
        sid = await mgr.create_session()
        messages = []
        for i in range(5):
            messages.extend(_make_turn(i))
        await mgr.save_messages(sid, messages)

        # Older compaction covers turns 0-1 (messages 0-3).
        await mgr.apply_compaction(sid, "older summary", boundary_seq=3)
        # Latest compaction covers all turns up to the 5th.
        await mgr.apply_compaction(sid, "latest summary", boundary_seq=9)

        # Add one more turn. With protect_turns=2, the latest compaction at 9
        # is too recent (only one turn after it), so load_history should fall
        # back to the older compaction at boundary=3.
        await mgr.save_messages(sid, _make_turn(5))
        loaded = await mgr.load_history(sid, protect_turns=2)

        # summary + messages after boundary=3 (seq 4-11 = 8 messages)
        assert len(loaded) == 9
        assert "older summary" in loaded[0].parts[0].content

    async def test_enough_turns_after_boundary_uses_latest_compaction(self, db_path: str) -> None:
        """If enough turns exist after the latest boundary, use it."""
        mgr = LocalSessionManager(db_path=db_path)
        sid = await mgr.create_session()
        messages = []
        for i in range(5):
            messages.extend(_make_turn(i))
        await mgr.save_messages(sid, messages)
        await mgr.apply_compaction(sid, "latest summary", boundary_seq=5)

        # Add two more turns → 2 turns after boundary=5, satisfying protect_turns=2.
        for i in range(5, 7):
            await mgr.save_messages(sid, _make_turn(i))

        loaded = await mgr.load_history(sid, protect_turns=2)

        assert "latest summary" in loaded[0].parts[0].content
        # messages after boundary=5: turns 3-6 = 8 messages
        assert len(loaded) == 9

    async def test_not_enough_total_turns_loads_all_messages(self, db_path: str) -> None:
        """If total turns < protect_turns, no compaction is eligible."""
        mgr = LocalSessionManager(db_path=db_path)
        sid = await mgr.create_session()
        messages = []
        for i in range(2):
            messages.extend(_make_turn(i))
        await mgr.save_messages(sid, messages)
        await mgr.apply_compaction(sid, "summary", boundary_seq=3)

        loaded = await mgr.load_history(sid, protect_turns=5)

        # protect_turns not satisfied → full history
        assert len(loaded) == 4

    async def test_empty_session_returns_empty_history(self, db_path: str) -> None:
        """An empty session produces an empty message list."""
        mgr = LocalSessionManager(db_path=db_path)
        sid = await mgr.create_session()

        loaded = await mgr.load_history(sid, protect_turns=2)

        assert loaded == []


class TestInferRoleStrict:
    """_infer_role classifies exactly: user ⟺ turn start."""

    @staticmethod
    def _request(*parts: ModelMessage) -> ModelRequest:
        """Build a ModelRequest whose parts are the given parts."""
        return ModelRequest(
            parts=list(parts),
            kind="request",
            timestamp=datetime.now(timezone.utc),
        )

    def test_user_prompt_first_is_user(self) -> None:
        """A request starting with UserPromptPart is USER."""
        msg = self._request(
            UserPromptPart(content="u", timestamp=datetime.now(timezone.utc))
        )
        assert _infer_role(msg) == MessageRole.USER

    def test_tool_return_first_is_tool(self) -> None:
        """A request starting with ToolReturnPart is TOOL."""
        msg = self._request(
            ToolReturnPart(tool_name="t", tool_call_id="1", content="c")
        )
        assert _infer_role(msg) == MessageRole.TOOL

    def test_system_prompt_first_is_unknown(self) -> None:
        """A request starting with SystemPromptPart is UNKNOWN, not USER."""
        msg = self._request(SystemPromptPart(content="sys"))
        assert _infer_role(msg) == MessageRole.UNKNOWN

    def test_system_then_user_is_unknown(self) -> None:
        """A system-prompt-led mixed request is not a turn start."""
        msg = self._request(
            SystemPromptPart(content="s"),
            UserPromptPart(content="u", timestamp=datetime.now(timezone.utc)),
        )
        assert _infer_role(msg) == MessageRole.UNKNOWN

    def test_response_is_assistant(self) -> None:
        """A ModelResponse is ASSISTANT."""
        msg = ModelResponse(
            parts=[TextPart(content="hi", part_kind="text")],
            kind="response",
            timestamp=datetime.now(timezone.utc),
        )
        assert _infer_role(msg) == MessageRole.ASSISTANT

    def test_empty_parts_request_is_unknown(self) -> None:
        """A ModelRequest with no parts is UNKNOWN, not a turn start."""
        msg = ModelRequest(
            parts=[], kind="request", timestamp=datetime.now(timezone.utc)
        )
        assert _is_turn_start(msg) is False
        assert _infer_role(msg) == MessageRole.UNKNOWN


class TestFindCutoffSeqRoleEquivalence:
    """The indexed role query matches scan-based turn detection."""

    @staticmethod
    async def _seed_mixed(mgr: LocalSessionManager) -> str:
        """Create a session mixing user/tool/unknown/assistant messages."""
        sid = await mgr.create_session()
        ts = datetime.now(timezone.utc)
        messages = [
            ModelRequest(parts=[UserPromptPart(content="u1", timestamp=ts)], kind="request", timestamp=ts),
            ModelRequest(parts=[ToolReturnPart(tool_name="t", tool_call_id="1", content="c")], kind="request", timestamp=ts),
            ModelRequest(parts=[SystemPromptPart(content="sys")], kind="request", timestamp=ts),
            ModelRequest(parts=[UserPromptPart(content="u2", timestamp=ts)], kind="request", timestamp=ts),
            ModelResponse(parts=[TextPart(content="r2", part_kind="text")], kind="response", timestamp=ts),
            ModelRequest(parts=[UserPromptPart(content="u3", timestamp=ts)], kind="request", timestamp=ts),
        ]
        await mgr.save_messages(sid, messages)
        return sid

    @staticmethod
    async def _cutoff(mgr: LocalSessionManager, sid: str, protect: int) -> int:
        """Run the cutoff query against the stored session."""
        async with mgr._connect() as db:
            return await mgr._find_cutoff_seq(db, sid, max_seq=5, protect_turns=protect)

    async def test_cutoff_matches_turn_start_scan(self, db_path: str) -> None:
        """Role-based cutoff equals the scan-based _is_turn_start cutoff."""
        mgr = LocalSessionManager(db_path=db_path)
        sid = await self._seed_mixed(mgr)

        # scan-based baseline: which seqs are turn starts
        loaded = await mgr.load_history(sid, protect_turns=0)
        turn_seqs = [i for i, m in enumerate(loaded) if _is_turn_start(m)]

        for protect in [1, 2, 3, 4]:
            expected = turn_seqs[-protect] if len(turn_seqs) >= protect else 0
            assert await self._cutoff(mgr, sid, protect) == expected

    async def test_role_column_matches_turn_start_exactly(self, db_path: str) -> None:
        """Stored role='user' rows are exactly the turn starts."""
        mgr = LocalSessionManager(db_path=db_path)
        sid = await self._seed_mixed(mgr)

        async with mgr._connect() as db:
            cursor = await db.execute(
                "SELECT message_seq, role FROM messages "
                "WHERE session_id = ? ORDER BY message_seq",
                (sid,),
            )
            stored = {
                int(r["message_seq"]): r["role"] for r in await cursor.fetchall()
            }
        loaded = await mgr.load_history(sid, protect_turns=0)
        turn_seqs = {i for i, m in enumerate(loaded) if _is_turn_start(m)}

        assert {seq for seq, role in stored.items() if role == "user"} == turn_seqs

    async def test_protect_zero_returns_max_plus_one(self, db_path: str) -> None:
        """protect_turns=0 makes every compaction eligible (max_seq + 1)."""
        mgr = LocalSessionManager(db_path=db_path)
        sid = await mgr.create_session()

        async with mgr._connect() as db:
            cutoff = await mgr._find_cutoff_seq(db, sid, max_seq=5, protect_turns=0)

        assert cutoff == 6
