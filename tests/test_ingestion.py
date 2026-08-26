from __future__ import annotations

from research_copilot.ingestion import PaperIngestionService
from research_copilot.models import IngestionStatus
from research_copilot.storage import SQLiteRepository
from tests.fakes import MemoryVectorIndex


def test_ingestion_and_sha_dedup(settings, sample_pdf) -> None:
    repository = SQLiteRepository(settings.sqlite_path)
    vectors = MemoryVectorIndex()
    service = PaperIngestionService(settings, repository, vectors)
    first = service.ingest_local(sample_pdf, title="Sample", prefer_mineru=False)
    second = service.ingest_local(sample_pdf, title="Renamed", prefer_mineru=False)
    assert first.status == IngestionStatus.READY
    assert first.page_count == 2
    assert first.chunk_count == vectors.count()
    assert second.duplicate is True
    assert second.paper_id == first.paper_id


def test_failed_vector_write_rolls_back(settings, sample_pdf) -> None:
    repository = SQLiteRepository(settings.sqlite_path)
    vectors = MemoryVectorIndex(fail_upsert=True)
    result = PaperIngestionService(settings, repository, vectors).ingest_local(
        sample_pdf, title="Failure", prefer_mineru=False
    )
    assert result.status == IngestionStatus.FAILED
    assert vectors.count() == 0
    assert repository.get_paper(result.paper_id)["status"] == "failed"


def test_same_sha_can_retry_after_failed_version(settings, sample_pdf) -> None:
    repository = SQLiteRepository(settings.sqlite_path)
    failed = PaperIngestionService(
        settings, repository, MemoryVectorIndex(fail_upsert=True)
    ).ingest_local(sample_pdf, title="Retry", prefer_mineru=False)
    vectors = MemoryVectorIndex()
    retried = PaperIngestionService(settings, repository, vectors).ingest_local(
        sample_pdf, title="Retry", prefer_mineru=False
    )
    assert failed.status == IngestionStatus.FAILED
    assert retried.status == IngestionStatus.READY
    assert retried.duplicate is False
    assert retried.version == 1
