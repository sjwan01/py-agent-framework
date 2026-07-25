"""Init-time factories, lifecycle, and discovery for AgentRunner."""
from __future__ import annotations

# TODO: 移除 logging。Agent 框架层不应包含业务无关的日志输出；
#       异常应直接抛出或交给上层/Extension 处理。
import importlib.util
import logging
import os
import tempfile
from pathlib import Path

from agent_framework.compaction import HarnessSummarizer
from agent_framework.context import ContextManager
from agent_framework.session import LocalSessionManager, PostgresSessionManager
from agent_framework.tools import LocalToolSource, ToolLifecycle
from agent_framework.types import ToolLifecycleEvent


def default_context_manager(self) -> ContextManager | None:
    """Create a default ContextManager using the configured context window."""
    return ContextManager(
        context_window_cap=self._settings.context_window,
        low_watermark_ratio=self._settings.low_watermark_ratio,
        high_watermark_ratio=self._settings.high_watermark_ratio,
        protect_turns=self._settings.protect_turns,
        truncate_chars=self._settings.truncate_tool_result_chars,
    )


def default_compaction_summarizer(self):
    """Create a default Harness-backed summarizer if a compaction model is configured."""
    if self._settings.compaction_model_id:
        return HarnessSummarizer(self._settings)
    return None


def default_session_manager(self):
    """Pick Postgres if configured, otherwise SQLite (temp file by default), otherwise single-turn."""
    if self._settings.postgres_url:
        return PostgresSessionManager(
            pg_url=self._settings.postgres_url,
            pool_size=self._settings.pg_pool_size,
            max_overflow=self._settings.pg_max_overflow,
        )
    db_path = self._settings.sqlite_path
    if db_path is None:
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
    return LocalSessionManager(
        db_path=db_path,
    )


async def ensure_tool_lifecycle(self):
    """Lazily build ToolLifecycle and register extension sources / raw tools."""
    if self._tool_lifecycle_initialized:
        return self._tool_lifecycle

    if self._tool_lifecycle is None:
        if self._raw_tools or self._extensions:
            self._tool_lifecycle = ToolLifecycle()
        else:
            self._tool_lifecycle_initialized = True
            return None

    # Subscribe extension tool event handlers before registering sources so
    # they can influence conflicts such as TOOL_CONFLICT.
    for ext in self._extensions:
        handler = getattr(ext, "on_tool_event", None)
        if handler is None:
            continue
        for event in ToolLifecycleEvent:
            self._tool_lifecycle.on(event, handler)

    if self._raw_tools:
        await self._tool_lifecycle.add_source(LocalToolSource(self._raw_tools))

    for ext in self._extensions:
        register = getattr(ext, "register_tool_sources", None)
        if register is None:
            continue
        try:
            sources = await register()
        except Exception as exc:  # pragma: no cover - fail-open
            logging.getLogger(__name__).warning(
                "Extension %s register_tool_sources failed: %s",
                type(ext).__name__, exc,
                exc_info=True,
            )
            continue
        for src in sources or []:
            await self._tool_lifecycle.add_source(src)

    self._tool_lifecycle_initialized = True
    return self._tool_lifecycle


async def trigger_compaction(self, session_id: str) -> None:
    try:
        boundary_seq = await self._session_manager.get_max_message_seq(session_id)
        if boundary_seq < 0:
            return

        messages = await self._session_manager.load_history(session_id)

        summarizer = self._compaction_summarizer
        if summarizer is None:
            summary = "Context compacted to fit window."
        else:
            summary = await summarizer.summarize(messages)

        await self._session_manager.apply_compaction(
            session_id,
            summary=summary,
            boundary_seq=boundary_seq,
        )
    except Exception as exc:  # pragma: no cover - fail-open
        logging.getLogger(__name__).warning(
            "Compaction failed for session %s: %s", session_id, exc,
            exc_info=True,
        )


def discover_extensions(paths: list[str]) -> list:
    """Discover Extension implementations from Python files in the given paths."""
    extensions = []
    for path_str in paths:
        p = Path(path_str)
        if not p.exists():
            continue
        for py_file in p.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            spec = importlib.util.spec_from_file_location(py_file.stem, str(py_file))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                for attr_name in dir(mod):
                    attr = getattr(mod, attr_name)
                    if isinstance(attr, type) and hasattr(attr, "on_agent_runner_event"):
                        if attr_name != "Extension":
                            try:
                                extensions.append(attr())
                            except TypeError:
                                pass
    return extensions
