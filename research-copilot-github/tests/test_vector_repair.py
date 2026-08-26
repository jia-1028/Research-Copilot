from __future__ import annotations

from pathlib import Path

import pytest

from research_copilot.errors import ResearchCopilotError
from research_copilot.models import IngestionStatus, Paper, PaperVersion, SourceType
from research_copilot.storage import SQLiteRepository
from research_copilot.vector_index import ChromaVectorIndex
from research_copilot.vector_repair import repair_chroma_index
from tests.fakes import DeterministicEmbeddings


def add_ready_version(repository, sample_pdf: Path, *, managed_path: Path | None = None) -> None:
    repository.upsert_paper(
        Paper(
            paper_id="paper-a",
            title="Paper A",
            source_type=SourceType.LOCAL,
            source_uri=str(sample_pdf),
            active_version=1,
            status=IngestionStatus.READY,
        )
    )
    repository.add_version(
        PaperVersion(
            paper_id="paper-a",
            version=1,
            sha256="repair-sha",
            original_path=str(sample_pdf),
            managed_copy_path=str(managed_path or sample_pdf),
            parser_name="pymupdf",
            parser_version="test",
            parsed_dir=str(sample_pdf.parent / "parsed"),
            status=IngestionStatus.READY,
        )
    )


def test_repair_builds_valid_staging_then_preserves_corrupt_backup(
    settings, sample_pdf
) -> None:
    repository = SQLiteRepository(settings.sqlite_path)
    add_ready_version(repository, sample_pdf)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    (settings.chroma_dir / "corrupt.marker").write_text("broken", encoding="utf-8")

    result = repair_chroma_index(
        settings, embedding_model=DeterministicEmbeddings()
    )

    assert result["status"] == "repaired"
    assert result["paper_count"] == 1
    assert result["chunk_count"] > 0
    backup_dir = Path(result["backup_dir"])
    assert (backup_dir / "corrupt.marker").read_text(encoding="utf-8") == "broken"
    index = ChromaVectorIndex(settings, DeterministicEmbeddings())
    assert index.count() == result["chunk_count"]
    assert index.similarity_search("attention method", {"paper-a": 1}, 1)
    index.close()


def test_repair_preflight_does_not_touch_production_when_pdf_is_missing(
    settings, sample_pdf
) -> None:
    repository = SQLiteRepository(settings.sqlite_path)
    missing = sample_pdf.parent / "missing.pdf"
    add_ready_version(repository, sample_pdf, managed_path=missing)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    marker = settings.chroma_dir / "production.marker"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(ResearchCopilotError, match="受控 PDF 不存在"):
        repair_chroma_index(
            settings, embedding_model=DeterministicEmbeddings()
        )

    assert marker.read_text(encoding="utf-8") == "keep"
