from __future__ import annotations

import json
import time
from pathlib import Path

from research_copilot.background_tasks import BackgroundTaskService
from research_copilot.evaluation import benchmark_chunking, validate_eval_dataset
from research_copilot.exports import (
    comparison_markdown,
    conversation_json,
    conversation_markdown,
)
from research_copilot.models import (
    DocumentRole,
    IngestionResult,
    IngestionStatus,
    PaperComparison,
)
from research_copilot.storage import SQLiteRepository
from tests.fakes import DeterministicEmbeddings


class StubIngestion:
    def ingest_local(self, pdf_path: Path, **_kwargs) -> IngestionResult:
        return IngestionResult(
            job_id="ingestion-a",
            paper_id=pdf_path.stem,
            version=1,
            status=IngestionStatus.READY,
            page_count=2,
            chunk_count=3,
        )


class StubArxiv:
    def import_paper(self, arxiv_id: str, **_kwargs) -> IngestionResult:
        return IngestionResult(
            job_id="ingestion-arxiv",
            paper_id=f"arxiv-{arxiv_id}",
            version=1,
            status=IngestionStatus.READY,
        )


class StubRag:
    def compare(self, paper_ids: list[str]) -> PaperComparison:
        return PaperComparison(paper_ids=paper_ids, rows=[])


def _wait_for_terminal(repository: SQLiteRepository, task_id: str) -> dict:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        task = repository.get_background_task(task_id)
        if task and task["status"] in {"completed", "failed", "cancelled"}:
            return task
        time.sleep(0.02)
    raise AssertionError("background task did not finish")


def test_background_task_persists_result_and_cancel_state(settings, sample_pdf) -> None:
    repository = SQLiteRepository(settings.sqlite_path)
    service = BackgroundTaskService(
        repository,
        StubIngestion(),  # type: ignore[arg-type]
        StubArxiv(),  # type: ignore[arg-type]
        StubRag(),  # type: ignore[arg-type]
        max_workers=1,
    )
    try:
        task_id = service.submit_local_import(
            sample_pdf,
            title=None,
            document_role=DocumentRole.MAIN,
            parent_paper_id=None,
            prefer_mineru=False,
        )
        task = _wait_for_terminal(repository, task_id)
        assert task["status"] == "completed"
        assert task["result"]["paper_id"] == "sample"

        queued_id = "queued-cancel"
        repository.create_background_task(queued_id, "compare_papers", {"paper_ids": ["a", "b"]})
        assert repository.request_background_task_cancel(queued_id)
        assert repository.get_background_task(queued_id)["status"] == "cancelled"
    finally:
        service.close()


def test_exports_include_answers_and_pdf_evidence() -> None:
    conversation = {
        "title": "方法讨论",
        "paper_snapshots": [
            {
                "paper_title_snapshot": "Paper A",
                "paper_version_snapshot": 1,
            }
        ],
    }
    messages = [
        {"role": "user", "content": "方法是什么？", "payload": None},
        {
            "role": "assistant",
            "content": "使用注意力 [C1]",
            "payload": {
                "citations": [
                    {
                        "citation_id": "C1",
                        "paper_title": "Paper A",
                        "pdf_page": 3,
                        "chunk_id": "paper-a:v1:p0003:c000",
                        "evidence_text": "attention module",
                    }
                ]
            },
        },
    ]
    markdown = conversation_markdown(conversation, messages)
    assert "方法讨论" in markdown
    assert "PDF 第 3 页" in markdown
    assert json.loads(conversation_json(conversation, messages))["messages"][1]["role"] == "assistant"
    assert "多论文证据比较" in comparison_markdown(
        PaperComparison(paper_ids=["a", "b"], rows=[])
    )


def test_validate_eval_dataset_and_chunk_benchmark(settings, sample_pdf, tmp_path) -> None:
    repository = SQLiteRepository(settings.sqlite_path)
    now = "2026-01-01T00:00:00+00:00"
    with repository.connect() as conn:
        conn.execute(
            """
            INSERT INTO papers(
                paper_id,title,source_type,source_uri,document_role,parent_paper_id,
                active_version,status,authors_json,abstract,arxiv_id,page_count,chunk_count,
                created_at,updated_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                "paper-a",
                "Paper A",
                "local",
                str(sample_pdf),
                "main",
                None,
                1,
                "ready",
                "[]",
                None,
                None,
                2,
                0,
                now,
                now,
            ),
        )
        conn.execute(
            """
            INSERT INTO paper_versions(
                paper_id,version,sha256,original_path,managed_copy_path,parser_name,
                parser_version,parsed_dir,page_count,chunk_count,status,created_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                "paper-a",
                1,
                "sha-a",
                str(sample_pdf),
                str(sample_pdf),
                "pymupdf",
                "test",
                str(tmp_path / "parsed"),
                2,
                0,
                "ready",
                now,
            ),
        )
    dataset = tmp_path / "questions.jsonl"
    case = {
        "question": "What attention method is used?",
        "paper_ids": ["paper-a"],
        "expected_pages": [1],
        "should_refuse": False,
    }
    cases = [case for _ in range(27)] + [
        {**case, "expected_pages": [], "should_refuse": True}
        for _ in range(3)
    ]
    dataset.write_text(
        "\n".join(json.dumps(item) for item in cases), encoding="utf-8"
    )
    assert validate_eval_dataset(repository, dataset)["valid"]

    report = benchmark_chunking(
        settings,
        repository,
        DeterministicEmbeddings(),
        dataset,
        tmp_path / "benchmark.json",
        configurations=[(800, 120)],
        top_k=2,
    )
    assert report["variants"][0]["chunk_count"] == 2
    assert report["variants"][0]["metrics"]["hit_at_k"] == 1.0
