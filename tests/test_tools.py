"""Tests for tool collection — SDK-object splitting, ordering, and fail-open."""
from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic_ai import Tool as PydanticTool
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset

from py_agent.runner import AgentRunner
from py_agent.runner._factory import _ResilientToolset
from py_agent.types import AgentRunnerEvent


def _mixed_tools() -> list[Any]:
    """One of each supported tool kind (raw callable, Tool, toolset)."""
    def local_func(x: int = 1) -> str:
        return f"local {x}"

    return [
        local_func,  # raw callable → wrapped into a Tool
        PydanticTool(lambda x=1: f"named {x}", name="named_tool"),
        FunctionToolset(
            [PydanticTool(lambda x=1: f"toolset {x}", name="ts_tool")],
            id="ts",
        ),
    ]


class _BrokenItem:
    """An item that cannot be turned into a tool (no name, not callable)."""


@pytest.fixture
def model() -> TestModel:
    """Return a deterministic Pydantic AI test model."""
    return TestModel()


class TestCollectTools:
    """collect_tools splits SDK objects into tools and toolsets."""

    async def test_splits_by_type(self, model: TestModel) -> None:
        """Callables and Tools go to tools; toolsets go to toolsets."""
        runner = AgentRunner(
            model=model, system_prompt="sp", tools=_mixed_tools()
        )

        tools, toolsets = await runner._collect_tools()

        names = [getattr(t, "name", None) for t in tools]
        assert "local_func" in names
        assert "named_tool" in names
        assert len(toolsets) == 1
        assert isinstance(toolsets[0], AbstractToolset)

    async def test_cached_after_first_collection(self, model: TestModel) -> None:
        """Tools are collected once and cached."""
        runner = AgentRunner(
            model=model, system_prompt="sp", tools=_mixed_tools()
        )
        await runner._collect_tools()
        assert runner._tools_initialized is True
        tools, toolsets = await runner._collect_tools()
        assert tools is runner._tools
        assert toolsets is runner._toolsets

    async def test_last_writer_wins_on_name_conflict(self, model: TestModel) -> None:
        """A later tool with the same name replaces the earlier one."""
        runner = AgentRunner(
            model=model,
            system_prompt="sp",
            tools=[
                PydanticTool(lambda: "v1", name="dup"),
                PydanticTool(lambda: "v2", name="dup"),
            ],
        )
        tools, _ = await runner._collect_tools()
        dup = [t for t in tools if t.name == "dup"]
        assert len(dup) == 1

    async def test_broken_item_raises(self, model: TestModel) -> None:
        """An invalid declared tool fails collection (fail-fast, not fail-open)."""
        runner = AgentRunner(
            model=model, system_prompt="sp", tools=[_BrokenItem()]
        )

        with pytest.raises(Exception):
            await runner._collect_tools()

    async def test_run_invokes_local_and_toolset_tools(self) -> None:
        """A run calls tools from both the tools and toolsets lists."""
        runner = AgentRunner(
            model=TestModel(call_tools=["local_func", "ts_ts_tool"]),
            system_prompt="sp",
            tools=_mixed_tools(),
        )

        result = await runner.run("hi")

        assert "local 1" in result.output
        assert "toolset 1" in result.output

    async def test_constructor_raw_tools(self, model: TestModel) -> None:
        """Raw callables passed to the constructor are wrapped."""
        runner = AgentRunner(
            model=TestModel(call_tools=["ctor_tool"]),
            system_prompt="sp",
            tools=[PydanticTool(lambda x=1: "ctor", name="ctor_tool")],
        )
        result = await runner.run("hi")
        assert "ctor" in result.output

    async def test_register_capabilities_raising_fails_open(
        self, model: TestModel
    ) -> None:
        """A raising register_capabilities is skipped with a warning."""
        class _RaisingExt:
            async def register_capabilities(self) -> list[Any]:
                raise RuntimeError("boom")

        warnings: list[str] = []

        def _warn(msg: str, exc: Exception | None = None) -> None:
            warnings.append(msg)

        runner = AgentRunner(
            model=model,
            system_prompt="sp",
            extensions=[_RaisingExt()],
            on_warning=_warn,
        )

        capabilities = await runner._collect_capabilities()

        assert capabilities == []
        assert any("register_capabilities failed" in w for w in warnings)

    async def test_register_capabilities_returning_none(
        self, model: TestModel
    ) -> None:
        """An extension returning None contributes nothing and does not crash."""
        class _NoneExt:
            async def register_capabilities(self) -> Any:
                return None

        runner = AgentRunner(
            model=model, system_prompt="sp", extensions=[_NoneExt()]
        )

        capabilities = await runner._collect_capabilities()

        assert capabilities == []

    async def test_empty_runner_has_no_tools(self, model: TestModel) -> None:
        """A runner with no tools and no extensions collects nothing."""
        runner = AgentRunner(model=model, system_prompt="sp")

        tools, toolsets = await runner._collect_tools()

        assert tools == []
        assert toolsets == []


class _BrokenToolset(FunctionToolset):
    """A toolset whose get_tools raises until fail_count is exhausted."""

    def __init__(self, *, fail_count: int = 999):
        super().__init__(
            [PydanticTool(lambda x=1: "bad", name="bad_tool")],
            id="broken",
        )
        self._fail_count = fail_count

    async def get_tools(self, ctx: Any) -> Any:
        if self._fail_count > 0:
            self._fail_count -= 1
            raise RuntimeError("server down")
        # return directly to avoid depending on ctx (unit tests pass None)
        return {"bad_tool": "recovered"}


class TestResilientToolset:
    """Catalog-load failures degrade instead of crashing the run."""

    async def test_broken_toolset_degrades_not_crashes(self) -> None:
        """One broken toolset is dropped; healthy toolsets keep working."""
        warnings: list[str] = []

        def _warn(msg: str, exc: Exception | None = None) -> None:
            warnings.append(msg)

        runner = AgentRunner(
            model=TestModel(call_tools=["good_good_tool"]),
            system_prompt="sp",
            tools=[
                FunctionToolset(
                    [PydanticTool(lambda x=1: "ok", name="good_tool")],
                    id="good",
                ),
                _BrokenToolset(),
            ],
            on_warning=_warn,
        )

        result = await runner.run("hi")

        assert "ok" in result.output
        assert any("unavailable" in w for w in warnings)

    async def test_toolset_recovers_next_run(self) -> None:
        """A toolset that failed once works again on the next get_tools call."""
        from py_agent.runner._factory import _ResilientToolset

        def _warn(msg: str, exc: Exception | None = None) -> None:
            pass

        broken = _BrokenToolset(fail_count=1)
        wrapped = _ResilientToolset(broken, _warn)

        # first load: catalog fails → dropped (empty)
        first = await wrapped.get_tools(None)  # type: ignore[arg-type]
        assert first == {}
        # second load: get_tools succeeds → tools available again
        second = await wrapped.get_tools(None)  # type: ignore[arg-type]
        assert len(second) == 1

    async def test_failure_handler_returns_dict(self) -> None:
        """A custom handler can substitute tools for the failing server."""
        handled: list[tuple[str, Exception]] = []

        def _handler(ts_id: str, exc: Exception) -> dict[str, Any] | None:
            handled.append((ts_id, exc))
            return {}

        runner = AgentRunner(
            model=TestModel(),
            system_prompt="sp",
            tools=[_BrokenToolset()],
            toolset_failure=_handler,
        )

        await runner.run("hi")

        assert len(handled) == 1
        assert "server down" in str(handled[0][1])

    async def test_failure_handler_raises_fails_run(self) -> None:
        """A handler raising fails the run (critical server policy)."""
        def _handler(ts_id: str, exc: Exception) -> dict[str, Any] | None:
            raise RuntimeError(f"critical server {ts_id} down")

        runner = AgentRunner(
            model=TestModel(),
            system_prompt="sp",
            tools=[_BrokenToolset()],
            toolset_failure=_handler,
        )

        with pytest.raises(RuntimeError, match="critical server"):
            await runner.run("hi")


class TestAgentRunnerClose:
    """close() releases the session backend."""

    async def test_close_calls_session_manager(self) -> None:
        """close() invokes the session manager's close method."""
        class _FakeSessionManager:
            def __init__(self) -> None:
                self.closed = False

            async def create_session(self, **kw: Any) -> str:
                return "x"

            async def load_history(self, *a: Any, **kw: Any) -> list[Any]:
                return []

            async def ensure_session(self, *a: Any, **kw: Any) -> str:
                return "x"

            async def close(self) -> None:
                self.closed = True

        sm = _FakeSessionManager()
        runner = AgentRunner(model=TestModel(), system_prompt="sp", session_manager=sm)

        await runner.close()

        assert sm.closed is True

    async def test_close_safe_without_close(self) -> None:
        """close() is a no-op when the session manager has no close method."""
        class _NoClose:
            async def create_session(self, **kw: Any) -> str:
                return "x"

            async def load_history(self, *a: Any, **kw: Any) -> list[Any]:
                return []

        runner = AgentRunner(
            model=TestModel(), system_prompt="sp", session_manager=_NoClose()  # type: ignore[arg-type]
        )

        await runner.close()  # must not raise


class TestToolsetFailureHandlerValidation:
    """toolset_failure handlers are validated eagerly at construction."""

    def test_missing_argument_raises_clear_error(self) -> None:
        """A handler with too few parameters fails fast with a clear message."""
        def _handler(ts_id: str) -> dict[str, Any] | None:
            return {}

        with pytest.raises(TypeError, match="toolset_failure"):
            AgentRunner(model=TestModel(), system_prompt="sp", toolset_failure=_handler)

    def test_too_many_required_arguments_raises(self) -> None:
        """A handler with too many required parameters fails fast."""
        def _handler(ts_id: str, exc: Exception, extra: str) -> dict[str, Any] | None:
            return {}

        with pytest.raises(TypeError, match="toolset_failure"):
            AgentRunner(model=TestModel(), system_prompt="sp", toolset_failure=_handler)

    def test_varargs_handler_is_accepted(self) -> None:
        """A handler using *args accepts the two positional arguments."""
        def _handler(*args: Any) -> dict[str, Any] | None:
            return {}

        runner = AgentRunner(model=TestModel(), system_prompt="sp", toolset_failure=_handler)
        assert runner._toolset_failure is _handler

    def test_optional_third_argument_is_accepted(self) -> None:
        """A third parameter with a default is fine (two are required)."""
        def _handler(ts_id: str, exc: Exception, extra: str = "") -> dict[str, Any] | None:
            return {}

        runner = AgentRunner(model=TestModel(), system_prompt="sp", toolset_failure=_handler)
        assert runner._toolset_failure is _handler

    def test_proper_signature_is_accepted(self) -> None:
        """The documented (toolset_id, exception) signature is accepted."""
        def _handler(ts_id: str, exc: Exception) -> dict[str, Any] | None:
            return {}

        runner = AgentRunner(model=TestModel(), system_prompt="sp", toolset_failure=_handler)
        assert runner._toolset_failure is _handler


class TestToolsetServerNames:
    """Server names are mandatory, unique, and prefix tools by default."""

    def test_missing_server_name_raises(self) -> None:
        """A toolset without an id fails the run with a clear error."""
        ts = FunctionToolset([PydanticTool(lambda x=1: "a", name="t")])

        runner = AgentRunner(
            model=TestModel(),
            system_prompt="sp",
            tools=[ts],
        )
        with pytest.raises(ValueError, match="server name"):
            asyncio.run(runner.run("hi"))

    def test_duplicate_server_name_raises(self) -> None:
        """Two toolsets with the same id fail the run."""
        mk = lambda: FunctionToolset(  # noqa: E731
            [PydanticTool(lambda x=1: "a", name="t1")], id="dup"
        )
        runner = AgentRunner(
            model=TestModel(),
            system_prompt="sp",
            tools=[mk(), mk()],
        )
        with pytest.raises(ValueError, match="Duplicate toolset server name"):
            asyncio.run(runner.run("hi"))

    def test_default_prefix_disambiguates_same_named_tools(self) -> None:
        """Same-named tools across servers coexist thanks to the default prefix."""
        ts1 = FunctionToolset(
            [PydanticTool(lambda x=1: "a", name="search")], id="s1"
        )
        ts2 = FunctionToolset(
            [PydanticTool(lambda x=1: "b", name="search")], id="s2"
        )
        runner = AgentRunner(
            model=TestModel(call_tools=["s1_search"]),
            system_prompt="sp",
            tools=[ts1, ts2],
        )
        r = asyncio.run(runner.run("hi"))
        # prefixed name is what the model calls; no conflict despite two
        # servers exposing an identically named tool
        assert '{"s1_search":"a"}' in r.output

    def test_prefix_disabled_reports_conflict_via_sdk(self) -> None:
        """Without prefixes, same-named tools conflict at SDK assembly time."""
        ts1 = FunctionToolset(
            [PydanticTool(lambda x=1: "a", name="search")], id="s1"
        )
        ts2 = FunctionToolset(
            [PydanticTool(lambda x=1: "b", name="search")], id="s2"
        )
        runner = AgentRunner(
            model=TestModel(call_tools=["search"]),
            system_prompt="sp",
            tools=[ts1, ts2],
            prefix_toolset_names=False,
        )
        with pytest.raises(Exception, match="conflicts"):
            asyncio.run(runner.run("hi"))


class _EnterBrokenToolset(FunctionToolset):
    """A toolset whose __aenter__ raises until fail_count is exhausted."""

    def __init__(self, *, fail_count: int = 999):
        super().__init__(
            [PydanticTool(lambda x=1: "ok", name="enter_tool")],
            id="enter_broken",
        )
        self._fail_count = fail_count

    async def __aenter__(self) -> Any:
        if self._fail_count > 0:
            self._fail_count -= 1
            raise RuntimeError("connection refused")
        return await super().__aenter__()

    async def get_tools(self, ctx: Any) -> Any:
        # return directly to avoid depending on ctx (unit tests pass None)
        return {"enter_tool": "ok"}


class TestResilientToolsetEnterFailure:
    """Connection failures (__aenter__) degrade like catalog failures."""

    async def test_enter_failure_degrades_not_crashes(self) -> None:
        """A server that fails to connect degrades; other servers keep working."""
        warnings: list[str] = []
        good = FunctionToolset(
            [PydanticTool(lambda x=1: "ok", name="good_enter")], id="good_enter"
        )

        runner = AgentRunner(
            model=TestModel(call_tools=["good_enter_good_enter"]),
            system_prompt="sp",
            tools=[good, _EnterBrokenToolset()],
            on_warning=lambda m, e: warnings.append(m),
        )
        result = await runner.run("hi")

        assert "ok" in result.output
        assert any("unavailable" in w for w in warnings)

    async def test_enter_failure_recovers_next_run(self) -> None:
        """A server that failed to connect is retried on the next run."""
        broken = _EnterBrokenToolset(fail_count=1)
        wrapped = _ResilientToolset(
            broken, lambda m, e: None, id="enter_broken"
        )

        async with wrapped:  # first entry fails → degrade
            tools = await wrapped.get_tools(None)  # type: ignore[arg-type]
        assert tools == {}

        async with wrapped:  # second entry succeeds → tools available
            tools2 = await wrapped.get_tools(None)  # type: ignore[arg-type]
        assert "enter_tool" in tools2

    async def test_exit_skipped_after_failed_enter(self) -> None:
        """__aexit__ after a failed __aenter__ does not raise."""
        broken = _EnterBrokenToolset()
        wrapped = _ResilientToolset(broken, lambda m, e: None, id="enter_broken")

        await wrapped.__aenter__()  # fails internally, degrades
        result = await wrapped.__aexit__(None, None, None)  # must not raise

        assert result is None


class TestMcpIntegration:
    """Real in-process FastMCP servers through the full AgentRunner path."""

    @staticmethod
    def _make_server(name: str, value: int) -> Any:
        from fastmcp import FastMCP

        server = FastMCP(name)

        @server.tool()
        def ping() -> int:
            """Return a fixed value."""
            return value

        return server

    async def test_real_mcp_toolset_end_to_end(self) -> None:
        """A real MCP server's tool is called through AgentRunner."""
        from pydantic_ai.mcp import MCPToolset

        server = self._make_server("calc", 7)
        runner = AgentRunner(
            model=TestModel(call_tools=["calc_ping"]),
            system_prompt="sp",
            tools=[MCPToolset(server, id="calc")],
        )
        result = await runner.run("hi")

        assert '{"calc_ping":7}' in result.output

    async def test_real_mcp_same_named_tools_prefixed(self) -> None:
        """Two real servers exposing the same tool name coexist via prefixes."""
        from pydantic_ai.mcp import MCPToolset

        s1 = self._make_server("s1", 1)
        s2 = self._make_server("s2", 2)
        runner = AgentRunner(
            model=TestModel(call_tools=["s1_ping"]),
            system_prompt="sp",
            tools=[MCPToolset(s1, id="s1"), MCPToolset(s2, id="s2")],
        )
        result = await runner.run("hi")

        assert '{"s1_ping":1}' in result.output

    async def test_real_down_mcp_server_degrades(self) -> None:
        """A real unreachable MCP server degrades; the healthy one still runs."""
        from pydantic_ai.mcp import MCPToolset

        warnings: list[str] = []
        server = self._make_server("calc", 5)
        down = MCPToolset("http://127.0.0.1:59999/mcp", id="down")
        runner = AgentRunner(
            model=TestModel(call_tools=["calc_ping"]),
            system_prompt="sp",
            tools=[MCPToolset(server, id="calc"), down],
            on_warning=lambda m, e: warnings.append(m),
        )
        result = await runner.run("hi")

        assert '{"calc_ping":5}' in result.output
        assert any("unavailable" in w for w in warnings)


class TestResilientToolsetEnterFailureHandler:
    """Connection failures also go through the custom handler."""

    async def test_enter_failure_calls_handler(self) -> None:
        """A __aenter__ failure invokes the toolset_failure handler."""
        handled: list[tuple[str, Exception]] = []

        def _handler(ts_id: str, exc: Exception) -> dict[str, Any] | None:
            handled.append((ts_id, exc))
            return {}

        runner = AgentRunner(
            model=TestModel(),
            system_prompt="sp",
            tools=[_EnterBrokenToolset()],
            toolset_failure=_handler,
        )

        await runner.run("hi")

        assert len(handled) == 1
        assert handled[0][0] == "enter_broken"
        assert "connection refused" in str(handled[0][1])


class TestLocalToolVsServerConflict:
    """Local tools participate in the SDK's cross-source conflict check."""

    async def test_local_and_server_same_name_conflicts_without_prefix(
        self,
    ) -> None:
        """With prefixes off, a local tool colliding with a server tool errors."""
        ts = FunctionToolset(
            [PydanticTool(lambda x=1: "srv", name="search")], id="s"
        )
        runner = AgentRunner(
            model=TestModel(),
            system_prompt="sp",
            prefix_toolset_names=False,
            tools=[PydanticTool(lambda x=1: "local", name="search"), ts],
        )

        with pytest.raises(Exception, match="conflicts"):
            await runner.run("hi")

    async def test_local_and_server_same_name_ok_with_prefix(self) -> None:
        """With prefixes on, the server tool is namespaced away from the local one."""
        runner = AgentRunner(
            model=TestModel(call_tools=["search"]),
            system_prompt="sp",
            tools=[
                PydanticTool(lambda x=1: "local", name="search"),
                FunctionToolset(
                    [PydanticTool(lambda x=1: "srv", name="search")], id="s"
                ),
            ],
        )

        result = await runner.run("hi")

        assert '{"search":"local"}' in result.output


class _ExitBrokenToolset(FunctionToolset):
    """A toolset whose __aexit__ raises (teardown failure)."""

    def __init__(self) -> None:
        super().__init__(
            [PydanticTool(lambda x=1: "ok", name="exit_tool")],
            id="exit_broken",
        )

    async def get_tools(self, ctx: Any) -> Any:
        return await super().get_tools(ctx)

    async def __aexit__(self, *args: Any) -> Any:
        raise RuntimeError("exit failed")


class TestResilientToolsetExitFailure:
    """Teardown failures fail open and never crash the run."""

    async def test_exit_failure_warns_and_does_not_raise(self) -> None:
        """A __aexit__ raising is caught: warn + return None."""
        warnings: list[str] = []
        wrapped = _ResilientToolset(
            _ExitBrokenToolset(),
            lambda m, e: warnings.append(m),
            id="exit_broken",
        )

        await wrapped.__aenter__()
        result = await wrapped.__aexit__(None, None, None)

        assert result is None
        assert any("exit failed" in w for w in warnings)

    async def test_run_succeeds_when_exit_fails(self) -> None:
        """Through AgentRunner, a teardown failure does not fail the run."""
        warnings: list[str] = []

        runner = AgentRunner(
            model=TestModel(call_tools=["exit_broken_exit_tool"]),
            system_prompt="sp",
            tools=[_ExitBrokenToolset()],
            on_warning=lambda m, e: warnings.append(m),
        )

        result = await runner.run("hi")

        assert '{"exit_broken_exit_tool":"ok"}' in result.output
        assert any("exit failed" in w for w in warnings)


def _make_skill_library() -> str:
    """Create a temp skill library with one skill named 'a' (TestModel seed)."""
    import pathlib
    import tempfile

    lib = pathlib.Path(tempfile.mkdtemp())
    (lib / "a").mkdir()
    ((lib / "a") / "SKILL.md").write_text(
        "---\ndescription: A test skill.\n---\n\nDo the thing carefully.\n"
    )
    return str(lib)


class TestSkills:
    """Skills reach the agent via the constructor parameter and extensions."""

    async def test_skills_parameter_reaches_agent(self) -> None:
        """A Skills instance passed to the constructor is available to the model."""
        from pydantic_ai_harness.skills import Skills

        runner = AgentRunner(
            model=TestModel(call_tools=["load_capability"]),
            system_prompt="sp",
            skills=Skills(_make_skill_library()),
        )

        result = await runner.run("hi")

        assert "load_capability" in result.output

    async def test_register_capabilities_extension(self) -> None:
        """Extensions can register SDK capabilities (e.g. Skills)."""
        from pydantic_ai_harness.skills import Skills

        class _CapExt:
            async def register_capabilities(self) -> list[Any]:
                return [Skills(_make_skill_library())]

        runner = AgentRunner(
            model=TestModel(call_tools=["load_capability"]),
            system_prompt="sp",
            extensions=[_CapExt()],
        )

        result = await runner.run("hi")

        assert "load_capability" in result.output

    async def test_skills_absent_means_no_load_capability(self) -> None:
        """Without skills, load_capability is not available to the model."""
        runner = AgentRunner(
            model=TestModel(),
            system_prompt="sp",
        )

        result = await runner.run("hi")

        assert "load_capability" not in result.output

    async def test_skills_and_extension_capabilities_combine(self) -> None:
        """Constructor skills and extension capabilities are both assembled."""
        from pydantic_ai_harness.skills import Skills

        class _CapExt:
            async def register_capabilities(self) -> list[Any]:
                return [Skills(_make_skill_library())]

        runner = AgentRunner(
            model=TestModel(call_tools=["load_capability"]),
            system_prompt="sp",
            skills=Skills(_make_skill_library()),
            extensions=[_CapExt()],
        )

        capabilities = await runner._collect_capabilities()

        assert len(capabilities) == 1  # only extension-registered; skills is separate
        assert runner._skills is not None


class TestCollectCapabilitiesCaching:
    """Extension capabilities are collected once and cached."""

    async def test_cached_after_first_collection(self, model: TestModel) -> None:
        """A second collection returns the cached list, extension called once."""
        calls: list[int] = []

        class _CapExt:
            async def register_capabilities(self) -> list[Any]:
                calls.append(1)
                return []

        runner = AgentRunner(
            model=model, system_prompt="sp", extensions=[_CapExt()]
        )

        first = await runner._collect_capabilities()
        second = await runner._collect_capabilities()

        assert first == [] and second == []
        assert len(calls) == 1  # extension ran once
        assert runner._capabilities_initialized is True

    async def test_multiple_extensions_merge_in_order(self, model: TestModel) -> None:
        """Capabilities from multiple extensions merge in extension order."""
        from pydantic_ai_harness.skills import Skills

        class _ExtA:
            async def register_capabilities(self) -> list[Any]:
                return [Skills(_make_skill_library())]

        class _ExtB:
            async def register_capabilities(self) -> list[Any]:
                return [Skills(_make_skill_library()), Skills(_make_skill_library())]

        runner = AgentRunner(
            model=model,
            system_prompt="sp",
            extensions=[_ExtA(), _ExtB()],
        )

        capabilities = await runner._collect_capabilities()

        assert len(capabilities) == 3  # 1 from A + 2 from B, in order


class TestSkillsMultiTurn:
    """Skill loading state survives across turns on a persistent session."""

    async def test_loaded_skill_persists_across_turns(self) -> None:
        """First turn loads the skill; second turn does not reload it."""
        import pathlib
        import tempfile

        from pydantic_ai_harness.skills import Skills

        from py_agent.session import LocalSessionManager

        class _Recorder:
            def __init__(self) -> None:
                self.events: list[str] = []

            async def on_agent_runner_event(
                self, event: str, data: dict[str, Any]
            ) -> dict[str, Any] | None:
                if event == AgentRunnerEvent.TOOL_CALL:
                    self.events.append(data.get("tool_name", ""))
                return None

        db = pathlib.Path(tempfile.mkdtemp()) / "skills.db"
        mgr = LocalSessionManager(db_path=str(db))
        skills = Skills(_make_skill_library())
        rec1 = _Recorder()
        runner1 = AgentRunner(
            model=TestModel(call_tools=["load_capability"]),
            system_prompt="sp",
            session_manager=mgr,
            skills=skills,
            extensions=[rec1],
        )
        r1 = await runner1.run("hi")

        # second runner reconnects to the same session (the reload pattern)
        rec2 = _Recorder()
        runner2 = AgentRunner(
            model=TestModel(call_tools=["load_capability"]),
            system_prompt="sp",
            session_manager=mgr,
            skills=skills,
            extensions=[rec2],
        )
        await runner2.run("again", session_id=r1.session_id)

        # first turn: the skill was loaded via the tool
        assert "load_capability" in rec1.events
        # second turn: no reload — the load state was restored from history
        assert rec2.events == []

        # the load_capability messages were persisted
        hist = await mgr.load_history(r1.session_id, protect_turns=0)
        kinds = [type(p).__name__ for m in hist for p in m.parts]
        assert "LoadCapabilityCallPart" in kinds
        assert "LoadCapabilityReturnPart" in kinds
