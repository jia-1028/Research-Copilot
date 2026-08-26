from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Validated runtime configuration. Secret values are never rendered."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    dashscope_api_key: SecretStr = Field(alias="DASHSCOPE_API_KEY")
    dashscope_base_url: str = Field(alias="DASHSCOPE_BASE_URL")
    deepseek_api_key: SecretStr | None = Field(default=None, alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com", alias="DEEPSEEK_BASE_URL"
    )
    mineru_api_token: SecretStr | None = Field(default=None, alias="MINERU_API_TOKEN")
    chat_provider: Literal["dashscope", "deepseek"] = Field(
        default="dashscope", alias="CHAT_PROVIDER"
    )
    chat_model: str = Field(default="deepseek-v4-flash-0731", alias="CHAT_MODEL")
    embedding_model: str = Field(default="qwen3.7-text-embedding", alias="EMBEDDING_MODEL")
    mineru_model: str = Field(default="pipeline", alias="MINERU_MODEL")
    mineru_language: str = Field(default="en", alias="MINERU_LANGUAGE")
    mineru_enabled: bool = Field(default=False, alias="MINERU_ENABLED")
    multimodal_enabled: bool = Field(default=True, alias="MULTIMODAL_ENABLED")
    page_image_dpi: int = Field(default=120, alias="PAGE_IMAGE_DPI", ge=72, le=240)
    page_image_jpeg_quality: int = Field(
        default=85, alias="PAGE_IMAGE_JPEG_QUALITY", ge=50, le=100
    )
    max_visual_pages: int = Field(default=3, alias="MAX_VISUAL_PAGES", ge=1, le=6)
    project_data_dir: Path = Field(default=Path("./data"), alias="PROJECT_DATA_DIR")
    chroma_data_dir: Path | None = Field(default=None, alias="CHROMA_DATA_DIR")
    chunk_size: int = Field(default=800, alias="CHUNK_SIZE", ge=200, le=5000)
    chunk_overlap: int = Field(default=120, alias="CHUNK_OVERLAP", ge=0, le=1000)
    retrieval_candidate_k: int = Field(default=8, alias="RETRIEVAL_CANDIDATE_K", ge=1, le=30)
    retrieval_context_k: int = Field(default=6, alias="RETRIEVAL_CONTEXT_K", ge=1, le=20)
    summary_batch_chunks: int = Field(default=40, alias="SUMMARY_BATCH_CHUNKS", ge=8, le=80)
    model_timeout_seconds: int = Field(default=15, alias="MODEL_TIMEOUT_SECONDS", ge=5, le=600)
    model_max_output_tokens: int = Field(
        default=1200, alias="MODEL_MAX_OUTPUT_TOKENS", ge=256, le=8192
    )
    disable_model_thinking: bool = Field(default=True, alias="DISABLE_MODEL_THINKING")
    fallback_chat_model: str | None = Field(default="qwen-plus", alias="FALLBACK_CHAT_MODEL")
    fallback_chat_provider: Literal["dashscope", "deepseek"] = Field(
        default="dashscope", alias="FALLBACK_CHAT_PROVIDER"
    )
    fallback_model_timeout_seconds: int = Field(
        default=45, alias="FALLBACK_MODEL_TIMEOUT_SECONDS", ge=10, le=600
    )
    model_circuit_breaker_seconds: int = Field(
        default=300, alias="MODEL_CIRCUIT_BREAKER_SECONDS", ge=0, le=3600
    )
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    max_pdf_mb: int = 200
    collection_name: str = "paper_chunks_v1"

    @field_validator("project_data_dir", mode="after")
    @classmethod
    def resolve_data_dir(cls, value: Path) -> Path:
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    @field_validator("chroma_data_dir", mode="after")
    @classmethod
    def resolve_chroma_data_dir(cls, value: Path | None) -> Path | None:
        if value is None:
            return None
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    @model_validator(mode="after")
    def validate_chunking(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.retrieval_context_k > self.retrieval_candidate_k:
            raise ValueError("RETRIEVAL_CONTEXT_K cannot exceed RETRIEVAL_CANDIDATE_K")
        if os.name == "nt" and not str(self.chroma_dir).isascii():
            raise ValueError(
                "CHROMA_DATA_DIR 在 Windows 上必须是全英文路径。"
                "请设置例如 CHROMA_DATA_DIR=E:\\research-copilot-data\\chroma，"
                "避免 Chroma HNSW 不能在重启后读取索引。"
            )
        return self

    @property
    def papers_dir(self) -> Path:
        return self.project_data_dir / "papers"

    @property
    def parsed_dir(self) -> Path:
        return self.project_data_dir / "parsed"

    @property
    def uploads_dir(self) -> Path:
        return self.project_data_dir / "uploads"

    @property
    def chroma_dir(self) -> Path:
        return self.chroma_data_dir or self.project_data_dir / "chroma"

    @property
    def sqlite_path(self) -> Path:
        return self.project_data_dir / "app.db"

    @property
    def checkpoint_path(self) -> Path:
        return self.project_data_dir / "checkpoints.db"

    def ensure_directories(self) -> None:
        for path in (
            self.project_data_dir,
            self.papers_dir,
            self.parsed_dir,
            self.uploads_dir,
            self.chroma_dir,
            self.project_data_dir / "reports",
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
