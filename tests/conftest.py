from __future__ import annotations

from pathlib import Path

import pymupdf as fitz
import pytest

from research_copilot.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    value = Settings(
        DASHSCOPE_API_KEY="test-key",
        DASHSCOPE_BASE_URL="https://example.test/v1",
        MINERU_API_TOKEN="test-mineru-token",
        MINERU_ENABLED=False,
        PROJECT_DATA_DIR=tmp_path / "data",
    )
    value.ensure_directories()
    return value


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "sample.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Paper title\nMethod uses attention.\nResult DSC is 0.91.")
    page = document.new_page()
    page.insert_text((72, 72), "Experiments use Dataset A and Dice metric.")
    document.save(path)
    document.close()
    return path
