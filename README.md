# py-agent-framework

[![PyPI version](https://img.shields.io/pypi/v/py-agent-framework)](https://pypi.org/project/py-agent-framework/)
[![Python versions](https://img.shields.io/pypi/pyversions/py-agent-framework)](https://pypi.org/project/py-agent-framework/)
[![License](https://img.shields.io/github/license/sjwan01/py-agent-framework)](LICENSE)
[![CI](https://github.com/sjwan01/py-agent-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/sjwan01/py-agent-framework/actions/workflows/ci.yml)

A stateful LLM agent framework built on [pydantic-ai](https://ai.pydantic.dev).
It adds what Pydantic AI does not provide out of the box: **session-level
multi-turn persistence**, **context-window management**, and a **lifecycle
extension system** — with the smallest possible footprint.

*Architecture inspired by [Pi Agent](https://github.com/badlogic/pi-agent) —
its extension-driven lifecycle, progressive-disclosure skills, and
reload-by-reconstruction pattern. This is an independent implementation on
pydantic-ai, not a port.*

## Why py-agent-framework?

Pydantic AI gives you `Agent`, `Tool`, and Hooks primitives. What it does *not*
give you is memory across turns:

| | Bare pydantic-ai | py-agent-framework |
|---|---|---|
| Multi-turn conversation | ❌ you re-supply all history every call | ✅ `session_id` in, history out |
| Long conversations | ❌ you hand-roll truncation/summarization | ✅ dual-watermark context management |
| Tool/session lifecycle hooks | ⚠️ low-level Hooks only | ✅ `Extension` protocol with 15 events |
| Storage | ❌ none | ✅ SQLite / PostgreSQL / stateless |

If you were about to write your own "load history → truncate → call model →
save" loop, that loop is this library.

## Features

- **Session persistence** — three backends for three stages of the lifecycle:
  - `SingleTurnSessionManager` — stateless, zero setup, every run is fresh.
  - `LocalSessionManager` — SQLite, file-based, local prototyping with full multi-turn memory.
  - `PostgresSessionManager` — production: connection pooling, JSONB, concurrent access.
  - Reconnect by passing `session_id`; the stored system prompt lets a prompt-less
    runner resume without re-supplying it.
- **Context management** — dual watermarks keep long conversations inside the window:
  below the low watermark nothing happens; between watermarks old tool results are
  truncated; above the high watermark the distant past is summarized by an LLM.
  A single `protect_turns` value keeps the last N turns intact through both.
- **Extension system** — a `Protocol`-defined lifecycle: observe and intercept every
  event (tool calls, token streams, compaction votes, persistence). Tools and skills
  are declared on `AgentRunner`; extensions own the lifecycle and may register SDK
  capabilities.
- **Tools & MCP** — tools are plain pydantic-ai objects: `Tool`, raw callables, or
  `MCPToolset` with automatic multi-server disambiguation (`{server}_{tool}`),
  partial degradation when a server is down, and mandatory unique server names.
- **Skills** — on-demand skill libraries via pydantic-ai harness `Skills`: the model
  sees names + descriptions, loads the full body only when needed.

## Installation

Requires **Python ≥ 3.11**.

```bash
pip install py-agent-framework
```

Optional extras:

```bash
pip install "py-agent-framework[postgres]"   # PostgreSQL backend
```

## Quick Start

```python
import asyncio
from pydantic_ai.models.openai import OpenAIModel

from py_agent import AgentRunner

model = OpenAIModel("gpt-4o")   # any pydantic-ai model

runner = AgentRunner(
    model=model,
    system_prompt="You are a helpful assistant.",
)

async def main() -> None:
    result = await runner.run("Hello!")
    print(result.output)

asyncio.run(main())
```

By default the runner is **stateless** — every `run()` starts a fresh session.
Pass a persistent `session_manager` to enable multi-turn memory.

## Multi-turn Persistence

```python
from py_agent import AgentRunner
from py_agent.session import LocalSessionManager

runner = AgentRunner(
    model=model,
    system_prompt="You are a helpful assistant.",
    session_manager=LocalSessionManager(db_path="sessions.db"),
)

r1 = await runner.run("Remember: the answer is 42.")
r2 = await runner.run("What did I ask you to remember?", session_id=r1.session_id)
# → "You asked me to remember that the answer is 42."
```

Swap the backend without touching the rest of your code:

```python
from py_agent.session import PostgresSessionManager  # requires: pip install "py-agent-framework[postgres]"

session_manager = PostgresSessionManager(pg_url="postgres://user:pass@localhost/db")
```

## Context Management

Long conversations stay inside your model's context window automatically:

- **Low watermark** (default `0.6 × context_window_cap`): old tool results are truncated.
- **High watermark** (default `0.75 × context_window_cap`): the distant past is
  summarized by an LLM (compaction runs in the background, so your turn is not blocked).
- `protect_turns` (default 5) keeps the most recent turns intact through both.

Tune it per runner:

```python
from py_agent import AgentRunner, ContextConfig, SummarizerConfig

runner = AgentRunner(
    model=model,
    system_prompt="You are a helpful assistant.",
    session_manager=LocalSessionManager(db_path="sessions.db"),
    context_config=ContextConfig(
        low_watermark_ratio=0.6,
        high_watermark_ratio=0.75,
        protect_turns=5,
        truncate_chars=1_000,
        context_window_cap=128_000,
    ),
    summarizer_config=SummarizerConfig(),  # defaults: reuse main model
)
```

## Tools, MCP & Skills

```python
from pydantic_ai import Tool
from pydantic_ai.mcp import MCPToolset
from pydantic_ai_harness.skills import Skills

from py_agent import AgentRunner
from py_agent.session import SingleTurnSessionManager


def web_search(q: str) -> str:
    """Search the web."""
    ...


runner = AgentRunner(
    model=model,
    system_prompt="You are a helpful assistant.",
    tools=[
        Tool(web_search),
        MCPToolset("http://search:8000/mcp", id="search"),  # server name required
    ],
    skills=Skills(["skills/"]),
    session_manager=SingleTurnSessionManager(),
)
```

- Multi-server MCP tools are auto-prefixed (`{server}_{tool}`) so identical names never collide.
- If an MCP server is down, it degrades gracefully — other servers keep working
  (customize via `toolset_failure`).

## Extensions

`Extension` is a `Protocol` with a single hook: `on_agent_runner_event(event, data)`.
Subscribe to any of the 15 lifecycle events:

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

Selected events:

| Event | When | Can modify? |
|---|---|---|
| `BEFORE_AGENT_RUN` | before the model sees messages | ✅ replace `messages` (injected messages are persisted) |
| `TOOL_CALL` | before a tool runs | ✅ `{"block": true}` or rewrite `args` |
| `TOOL_RESULT` | after a tool runs | ✅ rewrite `content` |
| `TOKEN_STREAM` | each token chunk (`run_stream`) | ❌ |
| `SESSION_SAVE` | before persistence | ✅ replace `delta_messages` |
| `COMPACTION_TRIGGER` | compaction flagged | ✅ `{"cancel": true}` to veto |

## Public API

```python
from py_agent import AgentRunner, RunResult, ContextConfig, SummarizerConfig
from py_agent.session import SessionManager, SingleTurnSessionManager, LocalSessionManager, PostgresSessionManager
from py_agent.types import Extension, AgentRunnerEvent, MessageRole, ToolsetFailureHandler
```

`AgentRunner` also offers `run_stream()` — an async iterator yielding lifecycle
events and token chunks as they happen, ending with `run_end`.

## Development

Contributors: see [CONTRIBUTING.md](CONTRIBUTING.md) for setup and the
`make` workflow (`make bootstrap` → `make check`).

## Contributing

This is a **personal project** and not yet ready for external contributors.
Bug reports and feature discussions via
[issues](https://github.com/sjwan01/py-agent-framework/issues) are welcome. If you would
still like to submit a change, see [CONTRIBUTING.md](CONTRIBUTING.md) — PRs
are required and the project's standards are enforced.

## License

[MIT](LICENSE) © Shunji Wan
