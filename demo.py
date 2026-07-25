"""Minimal TUI for multi-turn conversation testing."""
from __future__ import annotations

import asyncio
import tempfile

from pydantic_ai.models.test import TestModel

from agent_framework.runner import AgentRunner
from agent_framework.models import AgentConfig
from agent_framework.settings import Settings
from agent_framework.session import LocalSessionManager


async def main():
    # -- model: use test model by default, real model if .env is configured --
    try:
        settings = Settings()
        model = None  # AgentRunner builds from settings
        using = f"real ({settings.llm_model_id})"
    except Exception:
        model = TestModel(custom_output_text="This is a test response.")
        settings = Settings(
            llm_model_id="test", llm_base_url="http://localhost", llm_api_key="test",
        )
        using = "test (offline)"

    # -- session: in-memory SQLite --
    db_path = tempfile.mktemp(suffix=".db")
    session_manager = LocalSessionManager(db_path=db_path)

    config = AgentConfig(instructions="Be helpful and concise.")

    runner = AgentRunner(
        settings=settings,
        config=config,
        model=model,
        session_manager=session_manager,
    )

    print(f"Model: {using}")
    print("Type /new for a fresh session, /quit to exit.\n")

    session_id = None
    while True:
        try:
            prompt = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not prompt:
            continue
        if prompt == "/quit":
            break
        if prompt == "/new":
            session_id = None
            print("[new session]\n")
            continue

        result = await runner.run(prompt, session_id=session_id)
        session_id = result.session_id
        print(f"\n{result.output}\n")


if __name__ == "__main__":
    asyncio.run(main())
