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
    # TODO: 缺少 thinking 相关配置（是否开启思考、思考等级）。
    #       Pydantic AI 支持 thinking / reasoning，但 V2 目前完全没透出这个 knob。
    llm_model_id: str
    llm_base_url: str
    llm_api_key: str

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
    # TODO: compaction 模型缺少独立 provider 配置。
    #       虽然有 compaction_model_id + max_output_tokens，但 base_url /
    #       api_key 复用主模型的，prompts 硬编码在 HarnessSummarizer 里，
    #       thinking 等级不可配置。
    compaction_model_id: str | None = None
    compaction_max_output_tokens: int | None = None
