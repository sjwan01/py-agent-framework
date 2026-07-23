"""ToolLifecycle — unified tool registry with conflict resolution."""
from __future__ import annotations

from typing import Any

from agent_framework.types import (
    ToolLifecycleEvent,
    ToolSource,
    ToolEventHandler,
)


class LocalToolSource(ToolSource):
    def __init__(self, tools: list):
        self._raw_tools = tools
        self._id = f"local_{id(self)}"

    async def discover(self) -> list:
        from pydantic_ai import Tool as PydanticTool
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


class MCPServerSource(ToolSource):
    def __init__(self, *, server_name: str, client_factory):
        self._server_name = server_name
        self._client_factory = client_factory
        self._client = None

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


class ToolLifecycle:
    def __init__(self):
        self._tools: dict[str, Any] = {}
        self._handlers: dict[str, list[ToolEventHandler]] = {}

    def on(self, event: str, handler: ToolEventHandler) -> None:
        self._handlers.setdefault(event, []).append(handler)

    async def _fire(self, event: str, data: dict) -> dict | None:
        for handler in self._handlers.get(event, []):
            result = await handler(event, data)
            if result is not None:
                return result
        return None

    async def add_source(self, source: ToolSource) -> list[str]:
        tools = await source.discover()
        registered = []
        for tool in tools:
            name = tool.name
            await self._fire(ToolLifecycleEvent.TOOL_DISCOVERED, {
                "tool_name": name, "source_type": source.source_type,
                "source_id": source.source_id, "tool": tool,
            })
            if name in self._tools:
                resolution = await self._fire(ToolLifecycleEvent.TOOL_CONFLICT, {
                    "tool_name": name,
                    "existing": self._tools[name]["tool"],
                    "existing_source": self._tools[name]["source_type"],
                    "incoming": tool,
                    "incoming_source": source.source_type,
                })
                if resolution and resolution.get("action") == "replace":
                    self._tools[name] = {
                        "tool": tool, "source_type": source.source_type,
                        "source_id": source.source_id,
                    }
                    registered.append(name)
            else:
                self._tools[name] = {
                    "tool": tool, "source_type": source.source_type,
                    "source_id": source.source_id,
                }
                registered.append(name)
                await self._fire(ToolLifecycleEvent.TOOL_REGISTERED, {
                    "tool_name": name, "source_type": source.source_type, "scope": "all",
                })
        return registered

    def get_for_scope(self) -> list:
        return [entry["tool"] for entry in self._tools.values()]
