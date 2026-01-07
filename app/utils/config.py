from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central configuration for the AgenticRAG-StudyCoach app.

    Loads values from environment variables and `.env` (project root).
    Keep secrets OUT of code; store them in `.env` locally and in your cloud secret store later.
    """

    # ---- OpenAI ----
    openai_api_key: str = ""

    # ---- Models ----
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model_name: str = "gpt-4o-mini"  # cost-effective; can swap later

    # ---- Paths ----
    raw_pdfs_dir: str = "data/raw_pdfs"
    chroma_dir: str = "data/chroma"

    # ---- Chunking ----
    chunk_size: int = 900
    chunk_overlap: int = 150

    # ---- Retrieval ----
    top_k: int = 6

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


def get_settings() -> Settings:
    """
    Single entry point to load settings.
    Use this everywhere instead of reading env vars directly.
    """
    s = Settings()

    # Guardrail: OpenAI key must be present before we call the LLM.
    # We don't hard-fail here because ingestion doesn't need it.
    return s
