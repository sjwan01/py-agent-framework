"""PostgreSQL session backend tests.

Requires a running Postgres and a dedicated test database. Set
``PG_TEST_URL`` to the connection URL (e.g.
``postgresql://user:pass@localhost:5432/py_agent_test``); tests are skipped
when the variable is unset.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, AsyncIterator

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel

from py_agent.models import ContextConfig
from py_agent.runner import AgentRunner
from py_agent.session import PostgresSessionManager
from py_agent.session._shared import _is_turn_start

PG_URL = os.environ.get("PG_TEST_URL")

pytestmark = pytest.mark.skipif(
    PG_URL is None, reason="PG_TEST_URL not set — skipping Postgres tests"
)


@pytest.fixture
async def mgr() -> AsyncIterator[PostgresSessionManager]:
    """Return a Postgres session manager, closed after the test."""
    assert PG_URL is not None
    session_mgr = PostgresSessionManager(pg_url=PG_URL)
    yield session_mgr
    await session_mgr.close()


async def _stored_roles(
    mgr: PostgresSessionManager, sid: str
) -> dict[int, str]:
    """Read back the stored (message_seq, role) pairs for a session."""
    pool = await mgr._get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT message_seq, role FROM messages "
                "WHERE session_id = %s ORDER BY message_seq",
                (sid,),
            )
            return {int(r[0]): str(r[1]) for r in await cur.fetchall()}


class TestPostgresRoundtrip:
    """Basic save/load behavior on Postgres."""

    async def test_save_and_load_history(self, mgr: PostgresSessionManager) -> None:
        """Messages saved to Postgres load back identically."""
        sid = await mgr.create_session()
        ts = datetime.now(timezone.utc)
        await mgr.save_messages(sid, [
            ModelRequest(
                parts=[UserPromptPart(content="hi", timestamp=ts)],
                kind="request",
                timestamp=ts,
            )
        ])

        loaded = await mgr.load_history(sid, protect_turns=0)

        assert len(loaded) == 1
        assert loaded[0].parts[0].content == "hi"

    async def test_system_prompt_roundtrip(self, mgr: PostgresSessionManager) -> None:
        """The latest saved system prompt loads back."""
        sid = await mgr.create_session()

        await mgr.save_system_prompt(sid, "first")
        await mgr.save_system_prompt(sid, "second")

        assert await mgr.load_system_prompt(sid) == "second"


class TestPostgresCompaction:
    """Compaction writes and loads on Postgres."""

    async def test_summary_loaded_after_compaction(
        self, mgr: PostgresSessionManager
    ) -> None:
        """A compaction summary is loaded with the messages after the boundary."""
        sid = await mgr.create_session()
        ts = datetime.now(timezone.utc)
        messages: list[Any] = []
        for i in range(3):
            messages.append(
                ModelRequest(
                    parts=[UserPromptPart(content=f"u{i}", timestamp=ts)],
                    kind="request",
                    timestamp=ts,
                )
            )
            messages.append(
                ModelResponse(
                    parts=[TextPart(content=f"r{i}", part_kind="text")],
                    kind="response",
                    timestamp=ts,
                )
            )
        await mgr.save_messages(sid, messages)
        await mgr.apply_compaction(sid, "pg summary", boundary_seq=3)

        loaded = await mgr.load_history(sid, protect_turns=0)

        assert "pg summary" in str(loaded[0].parts[0].content)
        assert len(loaded) == 3  # summary + messages after seq 3 (seq 4, 5)


class TestPostgresRunnerIntegration:
    """AgentRunner on the Postgres backend — the production deployment shape."""

    async def test_multi_turn_session_with_reconnect(
        self, mgr: PostgresSessionManager
    ) -> None:
        """A multi-turn conversation persists and reconnects on Postgres."""
        runner = AgentRunner(
            model=TestModel(), session_manager=mgr, system_prompt="sp"
        )
        r1 = await runner.run("hello")
        assert r1.output

        # second turn on the same session: stored system prompt is reused
        runner2 = AgentRunner(
            model=TestModel(), session_manager=mgr, system_prompt=None
        )
        r2 = await runner2.run("again", session_id=r1.session_id)
        assert r2.output
        assert r2.session_id == r1.session_id

        # two turns persisted: user prompt + model reply each
        loaded = await mgr.load_history(r1.session_id, protect_turns=0)
        assert len(loaded) == 4
        # the stored prompt lets a prompt-less runner reconnect
        assert await mgr.load_system_prompt(r1.session_id) == "sp"

    async def test_multi_turn_with_context_config(
        self, mgr: PostgresSessionManager
    ) -> None:
        """Context management config works end-to-end on Postgres."""
        runner = AgentRunner(
            model=TestModel(),
            session_manager=mgr,
            system_prompt="sp",
            context_config=ContextConfig(),
        )
        r1 = await runner.run("hello")
        r2 = await runner.run("again", session_id=r1.session_id)

        assert r2.output
        loaded = await mgr.load_history(r1.session_id, protect_turns=0)
        assert len(loaded) == 4
    """The role column is an exact classification on Postgres too."""

    async def test_role_user_matches_turn_start(
        self, mgr: PostgresSessionManager
    ) -> None:
        """Stored role='user' rows are exactly the turn starts."""
        sid = await mgr.create_session()
        ts = datetime.now(timezone.utc)
        messages = [
            ModelRequest(
                parts=[UserPromptPart(content="u1", timestamp=ts)],
                kind="request",
                timestamp=ts,
            ),
            ModelRequest(
                parts=[ToolReturnPart(tool_name="t", tool_call_id="1", content="c")],
                kind="request",
                timestamp=ts,
            ),
            ModelRequest(
                parts=[SystemPromptPart(content="sys")],
                kind="request",
                timestamp=ts,
            ),
            ModelRequest(
                parts=[UserPromptPart(content="u2", timestamp=ts)],
                kind="request",
                timestamp=ts,
            ),
        ]
        await mgr.save_messages(sid, messages)

        loaded = await mgr.load_history(sid, protect_turns=0)
        turn_seqs = {i for i, m in enumerate(loaded) if _is_turn_start(m)}

        stored = await _stored_roles(mgr, sid)
        assert {seq for seq, role in stored.items() if role == "user"} == turn_seqs

    async def test_cutoff_matches_turn_start_scan(
        self, mgr: PostgresSessionManager
    ) -> None:
        """The indexed role query matches scan-based turn detection."""
        sid = await mgr.create_session()
        ts = datetime.now(timezone.utc)
        messages = [
            ModelRequest(
                parts=[UserPromptPart(content="u1", timestamp=ts)],
                kind="request",
                timestamp=ts,
            ),
            ModelRequest(
                parts=[SystemPromptPart(content="sys")],
                kind="request",
                timestamp=ts,
            ),
            ModelRequest(
                parts=[UserPromptPart(content="u2", timestamp=ts)],
                kind="request",
                timestamp=ts,
            ),
        ]
        await mgr.save_messages(sid, messages)

        loaded = await mgr.load_history(sid, protect_turns=0)
        turn_seqs = [i for i, m in enumerate(loaded) if _is_turn_start(m)]
        assert turn_seqs == [0, 2]

        pool = await mgr._get_pool()
        async with pool.connection() as conn:
            for protect, expected in [(1, 2), (2, 0), (3, 0)]:
                cutoff = await mgr._find_cutoff_seq(
                    conn, sid, max_seq=2, protect_turns=protect
                )
                assert cutoff == expected
