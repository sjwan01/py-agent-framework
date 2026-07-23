"""Application settings loaded from environment variables and .env file."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Infrastructure configuration from .env.

    Separated from AgentConfig — infrastructure changes
    (model URL, credentials) don't touch agent behaviour.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_model_id: str
    llm_base_url: str
    llm_api_key: str
