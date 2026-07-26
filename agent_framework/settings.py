"""Application settings loaded from environment variables and .env file."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Infrastructure configuration from .env.

    Separated from AgentConfig — infrastructure changes
    (model URL, credentials) don't touch agent behaviour.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── LLM ──
    llm_model_id: str
    llm_base_url: str
    llm_api_key: str

    # Thinking (main agent).  Maps to pydantic-ai ModelSettings.thinking.
    thinking_enabled: bool = True
    thinking_level: str | None = None

    # Session persistence
    postgres_url: str | None = None
    pg_pool_size: int = 5
    pg_max_overflow: int = 10
    sqlite_path: str | None = None
    session_idle_timeout_seconds: int | None = None
    context_window: int = 128_000

    # Context truncation watermarks
    low_watermark_ratio: float = 0.6
    high_watermark_ratio: float = 0.75
    protect_turns: int = 5
    truncate_tool_result_chars: int = 1_000

    # Tool execution guardrails
    max_tool_calls_per_turn: int = 5
    parallel_tool_calls: bool = False

    # Compaction summarizer
    compaction_model_id: str | None = None
    compaction_max_output_tokens: int | None = None

    # Compaction model — independent base_url / api_key (fall back to main LLM).
    compaction_base_url: str | None = None
    compaction_api_key: str | None = None

    # Compaction thinking (separate from main agent).
    compaction_thinking_enabled: bool = False
    compaction_thinking_level: str | None = None

    # Custom summary prompt.  When None, the built-in four-section template is used.
    compaction_summary_prompt: str | None = None
