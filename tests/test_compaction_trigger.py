"""Tests for the compaction trigger path.

Covers COMPACTION_TRIGGER / COMPACTION_SCHEDULED events, extension
cancellation, the in-flight dedup set, background execution, and failure
handling in ``trigger_compaction``. The load side (boundary selection) is
covered by ``test_compaction.py``.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pytest
from pydantic_ai.messages import ModelRequest, ToolReturnPart, UserPromptPart
from pydantic_ai.models.test import TestModel

from py_agent.models import ContextConfig
from py_agent.runner import AgentRunner
from py_agent.session import LocalSessionManager
from py_agent.types import AgentRunnerEvent


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    """Return a temporary on-disk SQLite path."""
    return str(tmp_path / "test.db")


@pytest.fixture
def model() -> TestModel:
    """Return a deterministic Pydantic AI test model."""
    return TestModel()


def _trigger_config() -> ContextConfig:
    """Small-cap config that crosses the high watermark after truncation."""
    return ContextConfig(
        context_window_cap=1_200,
        low_watermark_ratio=0.5,
        high_watermark_ratio=0.75,
        protect_turns=2,
        truncate_chars=100,
    )


class _FakeSummarizer:
    """Recordable summarizer returning a fixed summary (or failing)."""

    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self._fail = fail

    async def summarize(self, messages: list[Any]) -> str:
        """Count the call and return the fake summary (or raise)."""
        self.calls += 1
        if self._fail:
            raise RuntimeError("summarizer exploded")
        return "fake summary"


class _EmptySummarizer:
    """Summarizer that produces an empty string."""

    async def summarize(self, messages: list[Any]) -> str:
        """Return an empty summary, as when the model produces no text."""
        return ""


class _CompactionRecorder:
    """Extension that captures the COMPACTION_SCHEDULED payload."""

    def __init__(self) -> None:
        self.applied: dict[str, Any] | None = None

    async def on_agent_runner_event(
        self, event: str, data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Record COMPACTION_SCHEDULED; cancel nothing."""
        if event == AgentRunnerEvent.COMPACTION_SCHEDULED:
            self.applied = dict(data)
        return None


class _CancelCompaction:
    """Extension that votes to cancel every compaction."""

    def __init__(self) -> None:
        self.applied: dict[str, Any] | None = None

    async def on_agent_runner_event(
        self, event: str, data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Return a cancel vote at COMPACTION_TRIGGER; record the outcome."""
        if event == AgentRunnerEvent.COMPACTION_TRIGGER:
            return {"cancel": True}
        if event == AgentRunnerEvent.COMPACTION_SCHEDULED:
            self.applied = dict(data)
        return None


async def _seed_history(sm: LocalSessionManager, session_id: str) -> None:
    """Persist six turns of (user, 2000-char tool result) pairs."""
    messages: list[ModelRequest] = []
    ts = datetime.now(timezone.utc)
    for i in range(6):
        user = ModelRequest(
            parts=[UserPromptPart(content=f"u{i}", timestamp=ts)],
            kind="request",
            timestamp=ts,
        )
        tool = ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="t", tool_call_id=str(i), content="x" * 2_000
                )
            ],
            kind="request",
            timestamp=ts,
        )
        messages.extend([user, tool])
    await sm.save_messages(session_id, messages)


async def _wait_for(predicate: Callable[[], bool], timeout: float = 2.0) -> None:
    """Poll until ``predicate`` is true or the timeout elapses."""
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            pytest.fail("condition not met within timeout")
        await asyncio.sleep(0.01)


async def _wait_for_summary(sm: LocalSessionManager, session_id: str) -> None:
    """Wait until the fake summary appears in the loaded history."""
    async def has_summary() -> bool:
        loaded = await sm.load_history(session_id, protect_turns=0)
        return bool(loaded) and "fake summary" in str(loaded[0].parts[0].content)

    deadline = asyncio.get_running_loop().time() + 2.0
    while not await has_summary():
        if asyncio.get_running_loop().time() > deadline:
            pytest.fail("compaction did not complete within timeout")
        await asyncio.sleep(0.01)


class TestCompactionTriggeredByRun:
    """Crossing the high watermark triggers a background compaction."""

    async def test_summary_is_written_to_compactions(
        self, db_path: str, model: TestModel
    ) -> None:
        """The background task summarizes and the summary is loaded back."""
        sm = LocalSessionManager(db_path=db_path)
        sid = await sm.create_session()
        await _seed_history(sm, sid)
        fake = _FakeSummarizer()
        recorder = _CompactionRecorder()
        runner = AgentRunner(
            model=model,
            session_manager=sm,
            context_config=_trigger_config(),
            system_prompt="sp",
            extensions=[recorder],
        )
        runner._compaction_summarizer = fake

        await runner.run("hi", session_id=sid)

        await _wait_for_summary(sm, sid)
        assert fake.calls == 1
        assert recorder.applied == {"session_id": sid, "cancelled": False}

    async def test_pending_session_skips_duplicate_task(
        self, db_path: str, model: TestModel
    ) -> None:
        """A session already in the in-flight set does not spawn a second task."""
        sm = LocalSessionManager(db_path=db_path)
        sid = await sm.create_session()
        await _seed_history(sm, sid)
        fake = _FakeSummarizer()
        runner = AgentRunner(
            model=model,
            session_manager=sm,
            context_config=_trigger_config(),
            system_prompt="sp",
        )
        runner._compaction_summarizer = fake
        runner._compaction_pending.add(sid)  # simulate an in-flight compaction

        await runner.run("hi", session_id=sid)

        await asyncio.sleep(0.05)  # give any (erroneous) task a chance to run
        assert fake.calls == 0
        assert sid in runner._compaction_pending


class TestFireNotifyCancelStrictness:
    """fire_notify honours only an explicit True cancel value."""

    @staticmethod
    def _runner_with_voter(model: TestModel, vote: dict[str, Any] | None) -> AgentRunner:
        """Build a runner whose only extension returns ``vote`` at every event."""

        class _Voter:
            async def on_agent_runner_event(
                self, event: str, data: dict[str, Any]
            ) -> dict[str, Any] | None:
                return vote

        return AgentRunner(model=model, system_prompt="sp", extensions=[_Voter()])

    async def test_true_cancel_vetoes(self, model: TestModel) -> None:
        """An explicit True cancel value vetoes the compaction."""
        runner = self._runner_with_voter(model, {"cancel": True})

        result = await runner._fire_notify(
            AgentRunnerEvent.COMPACTION_TRIGGER, {}
        )

        assert result == {"cancel": True}

    async def test_truthy_non_true_does_not_veto(self, model: TestModel) -> None:
        """Truthy but non-True values (1, 'yes') do not veto."""
        for vote in [{"cancel": 1}, {"cancel": "yes"}]:
            runner = self._runner_with_voter(model, vote)

            result = await runner._fire_notify(
                AgentRunnerEvent.COMPACTION_TRIGGER, {}
            )

            assert result == {"cancel": False}


class TestCompactionCancellation:
    """Extensions can veto compaction at COMPACTION_TRIGGER."""

    async def test_cancel_vote_skips_summarization(
        self, db_path: str, model: TestModel
    ) -> None:
        """A cancel vote blocks the summary write and reports cancelled."""
        sm = LocalSessionManager(db_path=db_path)
        sid = await sm.create_session()
        await _seed_history(sm, sid)
        cancel = _CancelCompaction()
        fake = _FakeSummarizer()
        runner = AgentRunner(
            model=model,
            session_manager=sm,
            context_config=_trigger_config(),
            system_prompt="sp",
            extensions=[cancel],
        )
        runner._compaction_summarizer = fake

        await runner.run("hi", session_id=sid)

        await asyncio.sleep(0.05)  # give any (erroneous) task a chance to run
        assert fake.calls == 0
        assert cancel.applied == {"session_id": sid, "cancelled": True}
        loaded = await sm.load_history(sid, protect_turns=0)
        assert not any("fake summary" in str(m.parts[0].content) for m in loaded)


class TestTriggerCompactionDirect:
    """trigger_compaction edge cases, invoked directly (no background task)."""

    async def test_summarizer_failure_warns_and_clears_pending(
        self, db_path: str, model: TestModel
    ) -> None:
        """A failing summarizer warns and unblocks future compactions."""
        sm = LocalSessionManager(db_path=db_path)
        sid = await sm.create_session()
        await _seed_history(sm, sid)
        warnings: list[str] = []

        def _warn(msg: str, exc: Exception | None = None) -> None:
            warnings.append(msg)

        runner = AgentRunner(
            model=model,
            session_manager=sm,
            context_config=_trigger_config(),
            system_prompt="sp",
            on_warning=_warn,
        )
        runner._compaction_summarizer = _FakeSummarizer(fail=True)
        runner._compaction_pending.add(sid)

        await runner._trigger_compaction(sid)

        assert any("Compaction failed" in w for w in warnings)
        assert sid not in runner._compaction_pending

    async def test_empty_session_skips_summarization(
        self, db_path: str, model: TestModel
    ) -> None:
        """A session with no messages never calls the summarizer."""
        sm = LocalSessionManager(db_path=db_path)
        sid = await sm.create_session()  # no messages
        fake = _FakeSummarizer()
        runner = AgentRunner(
            model=model,
            session_manager=sm,
            context_config=_trigger_config(),
            system_prompt="sp",
        )
        runner._compaction_summarizer = fake

        await runner._trigger_compaction(sid)

        assert fake.calls == 0

    async def test_no_summarizer_skips_silently(
        self, db_path: str, model: TestModel
    ) -> None:
        """A disabled summarizer makes the trigger a no-op."""
        sm = LocalSessionManager(db_path=db_path)
        sid = await sm.create_session()
        await _seed_history(sm, sid)
        runner = AgentRunner(
            model=model,
            session_manager=sm,
            context_config=_trigger_config(),
            system_prompt="sp",
        )
        runner._compaction_summarizer = None

        # must not raise even though summarization is impossible
        await runner._trigger_compaction(sid)

    async def test_empty_summary_skips_write_and_warns(
        self, db_path: str, model: TestModel
    ) -> None:
        """An empty summary is not written and triggers a warning."""
        sm = LocalSessionManager(db_path=db_path)
        sid = await sm.create_session()
        await _seed_history(sm, sid)
        warnings: list[str] = []

        def _warn(msg: str, exc: Exception | None = None) -> None:
            warnings.append(msg)

        runner = AgentRunner(
            model=model,
            session_manager=sm,
            context_config=_trigger_config(),
            system_prompt="sp",
            on_warning=_warn,
        )
        runner._compaction_summarizer = _EmptySummarizer()

        await runner._trigger_compaction(sid)

        assert any("empty summary" in w.lower() for w in warnings)
        loaded = await sm.load_history(sid, protect_turns=0)
        assert not any("fake summary" in str(m.parts[0].content) for m in loaded)
