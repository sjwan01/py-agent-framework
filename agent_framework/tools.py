"""ToolLifecycle — unified tool registry with conflict resolution."""
from __future__ import annotations

import inspect
from typing import Any

from pydantic_ai import Tool as PydanticTool

from agent_framework.types import (
    ToolLifecycleEvent,
    ToolSource,
    ToolEventHandler,
)


class LocalToolSource(ToolSource):
    def __init__(self, tools: list, *, scope: str = "all"):
        self._raw_tools = tools
        self._id = f"local_{id(self)}"
        self._scope = scope

    async def discover(self) -> list:
        result = []
        for t in self._raw_tools:
            if isinstance(t, PydanticTool):
                result.append(t)
            else:
                result.append(PydanticTool(t))
        return result

    @property
    def source_type(self) -> str:
        return "local"

    @property
    def source_id(self) -> str:
        return self._id

    @property
    def scope(self) -> str:
        return self._scope


class MCPServerSource(ToolSource):
    def __init__(self, *, server_name: str, client_factory, scope: str = "all"):
        self._server_name = server_name
        self._client_factory = client_factory
        self._client = None
        self._scope = scope

    async def discover(self) -> list:
        if self._client is None:
            self._client = self._client_factory()
        return list(await self._client.list_tools())

    @property
    def source_type(self) -> str:
        return "mcp"

    @property
    def source_id(self) -> str:
        return f"mcp_{self._server_name}"

    @property
    def scope(self) -> str:
        return self._scope


class SubagentToolSource(ToolSource):
    """Wrap an async runnable (e.g. a LangGraph graph) as a Pydantic AI Tool."""

    def __init__(
        self,
        name: str,
        runnable,
        *,
        description: str | None = None,
        scope: str = "subagent",
    ):
        self._name = name
        self._runnable = runnable
        self._description = description or f"Subagent tool: {name}"
        self._scope = scope

    async def discover(self) -> list:
        if inspect.iscoroutinefunction(self._runnable):
            tool_func = self._runnable
        else:
            async def tool_func(**kwargs):
                if hasattr(self._runnable, "ainvoke"):
                    return await self._runnable.ainvoke(kwargs)
                return await self._runnable(**kwargs)

        return [PydanticTool(tool_func, name=self._name, description=self._description)]

    @property
    def source_type(self) -> str:
        return "subagent"

    @property
    def source_id(self) -> str:
        return f"subagent_{self._name}"

    @property
    def scope(self) -> str:
        return self._scope


class ToolLifecycle:
    def __init__(self, *, on_warning=None):
        self._on_warning = on_warning or (lambda msg, exc=None: None)
        self._tools: dict[str, Any] = {}
        self._handlers: dict[str, list[ToolEventHandler]] = {}
        self.on(ToolLifecycleEvent.TOOL_CONFLICT, self._default_conflict_handler)

    def on(self, event: str, handler: ToolEventHandler) -> None:
        self._handlers.setdefault(event, []).append(handler)

    @staticmethod
    async def _default_conflict_handler(event: str, data: dict) -> dict | None:
        """Built-in dedup: local tools win over MCP tools."""
        if event != ToolLifecycleEvent.TOOL_CONFLICT:
            return None
        existing_source = data.get("existing_source")
        incoming_source = data.get("incoming_source")
        if existing_source == "local" and incoming_source == "mcp":
            return {"action": "keep"}
        if existing_source == "mcp" and incoming_source == "local":
            return {"action": "replace"}
        return {"action": "keep"}

    async def _fire(self, event: str, data: dict) -> dict:
        """Fire event to all subscribed handlers in chain mode.

        Each handler receives the current data (including updates from previous
        handlers). Non-None dict results are shallow-merged back.
        """
        current = dict(data)
        for handler in self._handlers.get(event, []):
            try:
                r = await handler(event, current)
            except Exception as exc:  # pragma: no cover - fail-open
                self._on_warning(
                    f"Tool event handler {getattr(handler, '__name__', repr(handler))} "
                    f"for {event} failed: {exc}",
                    exc,
                )
                continue
            if isinstance(r, dict):
                current.update(r)
        return current

    async def add_source(self, source: ToolSource) -> list[str]:
        tools = await source.discover()
        scope = getattr(source, "scope", "all")
        registered = []
        for tool in tools:
            name = tool.name
            await self._fire(ToolLifecycleEvent.TOOL_DISCOVERED, {
                "tool_name": name, "source_type": source.source_type,
                "source_id": source.source_id, "tool": tool, "scope": scope,
            })
            if name in self._tools:
                conflict_data = {
                    "tool_name": name,
                    "action": "keep",
                    "existing": self._tools[name]["tool"],
                    "existing_source": self._tools[name]["source_type"],
                    "incoming": tool,
                    "incoming_source": source.source_type,
                }
                resolution = await self._fire(ToolLifecycleEvent.TOOL_CONFLICT, conflict_data)
                if resolution.get("action") == "replace":
                    self._tools[name] = {
                        "tool": tool, "source_type": source.source_type,
                        "source_id": source.source_id, "scope": scope,
                    }
                    registered.append(name)
            else:
                self._tools[name] = {
                    "tool": tool, "source_type": source.source_type,
                    "source_id": source.source_id, "scope": scope,
                }
                registered.append(name)
                await self._fire(ToolLifecycleEvent.TOOL_REGISTERED, {
                    "tool_name": name, "source_type": source.source_type,
                    "scope": scope,
                })
        return registered

    def get_for_scope(self, scope: str | None = None) -> list:
        if scope is None:
            return [entry["tool"] for entry in self._tools.values()]
        return [
            entry["tool"] for entry in self._tools.values()
            if entry["scope"] in (scope, "all")
        ]
