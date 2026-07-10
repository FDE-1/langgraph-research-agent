from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):  # type: ignore[explicit-any]
    """Application settings read from the environment and .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "langgraph-research-agent"
    debug: bool = False
    log_json: bool = False
    openai_api_key: str | None = None
    tavily_api_key: str | None = None
    client_path: str = "workspace/chroma"
    collection_name: str = "agent_memory"
    workspace: Path = Path("workspace")
    checkpoint_path: Path = Path("checkpoint.sqlite")

    @field_validator("workspace")
    @classmethod
    def _resolve_workspace(cls, value: Path) -> Path:
        """save_file compares against this with is_relative_to; both sides must be absolute."""
        return value.resolve()


@lru_cache
def get_settings() -> Settings:
    """Return the settings"""
    return Settings()
