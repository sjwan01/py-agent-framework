# py-agent

A stateful LLM agent framework on [pydantic-ai](https://ai.pydantic.dev): session
persistence, context-window management, and a lifecycle extension system for tools
and skills — with the smallest possible footprint.

Pydantic AI provides the Agent, Tools, and Hooks primitives. `py-agent` adds what it
does not: **session-level multi-turn persistence** (SQLite / PostgreSQL / single-turn),
**context-window management** (watermark truncation + LLM compaction), and a clean
extension seam for lifecycle behavior.

## Features

- **Session persistence** — three backends for three stages of the lifecycle:
  - `SingleTurnSessionManager` — stateless, zero setup, every run is fresh.
  - `LocalSessionManager` — SQLite, file-based, local prototyping with full multi-turn memory.
  - `PostgresSessionManager` — production: connection pooling, JSONB, concurrent access.
  - Reconnect to a session by passing its `session_id`; the stored system prompt lets a
    prompt-less runner resume without re-supplying it.
- **Context management** — dual watermarks keep long conversations inside the window:
  below the low watermark nothing happens; between watermarks old tool results are
  truncated; above the high watermark the distant past is summarized by an LLM. A single
  `protect_turns` value keeps the last N turns intact through both.
- **Extension system** — a `Protocol`-defined lifecycle: observe and intercept every
  event (tool calls, token streams, compaction votes, persistence). Tools and skills are
  declared on `AgentRunner`; extensions own the lifecycle and may register SDK
  capabilities.
- **Tools & MCP** — tools are plain pydantic-ai objects: `Tool`, raw callables, or
  `MCPToolset` with automatic multi-server disambiguation (`{server}_{tool}`), partial
  degradation when a server is down, and mandatory unique server names.
- **Skills** — on-demand skill libraries via pydantic-ai harness `Skills`: the model sees
  names + descriptions, loads the full body only when needed.

## Quick start

```bash
pip install -r requirements.txt
```

```python
from py_agent import AgentRunner

runner = AgentRunner(
    model=my_model,                 # any pydantic-ai model
    system_prompt="You are a helpful assistant.",
)
result = await runner.run("Hello!")
```

### Multi-turn with persistence

```python
from py_agent import AgentRunner
from py_agent.session import LocalSessionManager

runner = AgentRunner(
    model=my_model,
    system_prompt="You are a helpful assistant.",
    session_manager=LocalSessionManager(db_path="sessions.db"),
)
r1 = await runner.run("Remember: the answer is 42.")
r2 = await runner.run("What did I ask you to remember?", session_id=r1.session_id)
```

### Tools and skills

```python
from py_agent import AgentRunner
from py_agent.session import SingleTurnSessionManager
from pydantic_ai import Tool
from pydantic_ai.mcp import MCPToolset
from pydantic_ai_harness.skills import Skills

def web_search(q: str) -> str:
    """Search the web."""
    ...

runner = AgentRunner(
    model=my_model,
    system_prompt="You are a helpful assistant.",
    tools=[
        Tool(web_search),
        MCPToolset("http://search:8000/mcp", id="search"),   # server name required
    ],
    skills=Skills(["skills/"]),
    session_manager=SingleTurnSessionManager(),
)
```

### Extensions (lifecycle)

```python
from py_agent import AgentRunner
from py_agent.types import AgentRunnerEvent, Extension

class Audit(Extension):
    async def on_agent_runner_event(self, event, data):
        if event == AgentRunnerEvent.TOOL_CALL:
            audit_log(data["tool_name"], data["args"])
        return None

runner = AgentRunner(..., extensions=[Audit()])
```

## Public API

- `from py_agent import AgentRunner, RunResult, ContextConfig, SummarizerConfig`
- `from py_agent.session import SessionManager, SingleTurnSessionManager, LocalSessionManager, PostgresSessionManager`
- `from py_agent.types import Extension, AgentRunnerEvent, MessageRole, ToolsetFailureHandler`

## Inspiration

Architecture inspired by [Pi Agent](https://github.com/badlogic/pi-agent) — its
extension-driven lifecycle, progressive-disclosure skills, and reload-by-reconstruction
pattern. This is an independent implementation on pydantic-ai, not a port.
