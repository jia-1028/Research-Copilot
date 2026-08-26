from __future__ import annotations

import re

from research_copilot.deep_analysis import DeepAnalysisService
from research_copilot.models import (
    DeepAnalysisPlan,
    DeepAnalysisTaskDraft,
    GroundedAnswerDraft,
    IngestionStatus,
    Paper,
    PaperChunk,
    PaperVersion,
    SourceType,
)
from research_copilot.rag import PaperRAGService
from research_copilot.storage import SQLiteRepository
from tests.fakes import MemoryVectorIndex


class DeepStructuredInvoker:
    def __init__(self, schema):
        self.schema = schema

    def invoke(self, messages):
        if self.schema is DeepAnalysisPlan:
            return DeepAnalysisPlan(
                tasks=[
                    DeepAnalysisTaskDraft(
                        focus="整体架构",
                        question="分析整体架构",
                        retrieval_query="architecture method",
                    ),
                    DeepAnalysisTaskDraft(
                        focus="训练与验证",
                        question="分析训练和实验验证",
                        retrieval_query="training experiments",
                    ),
                ]
            )
        if self.schema is GroundedAnswerDraft:
            content = str(messages[-1].content)
            citation_ids = list(dict.fromkeys(re.findall(r"T\d+-C\d+", content)))
            used = citation_ids[:2]
            return GroundedAnswerDraft(
                answer="Evidence-backed deep analysis "
                + "".join(f"[{item}]" for item in used),
                used_citation_ids=used,
            )
        raise AssertionError(f"unexpected schema: {self.schema}")


class FakeDeepModel:
    def with_structured_output(self, schema, **_kwargs):
        return DeepStructuredInvoker(schema)


def add_ready_paper(repository, vectors) -> None:
    repository.upsert_paper(
        Paper(
            paper_id="paper-a",
            title="Paper A",
            source_type=SourceType.LOCAL,
            source_uri="paper-a.pdf",
            status=IngestionStatus.PENDING,
        )
    )
    repository.add_version(
        PaperVersion(
            paper_id="paper-a",
            version=1,
            sha256="sha-paper-a",
            original_path="paper-a.pdf",
            managed_copy_path="managed-paper-a.pdf",
            parser_name="fake",
            parser_version="1",
            parsed_dir="parsed/paper-a",
        )
    )
    vectors.upsert_chunks(
        [
            PaperChunk(
                chunk_id=f"paper-a:v1:p000{page}:c000",
                paper_id="paper-a",
                paper_version=1,
                paper_title="Paper A",
                page_number=page,
                page_index=page - 1,
                chunk_index_on_page=0,
                text=text,
                text_hash=f"hash-{page}",
                source_uri="paper-a.pdf",
                embedding_model="fake",
            )
            for page, text in (
                (1, "The proposed architecture contains a specialist method."),
                (2, "Training and experiments validate the method."),
            )
        ]
    )
    repository.finish_version(
        "paper-a",
        1,
        parser_name="fake",
        parser_version="1",
        page_count=2,
        chunk_count=2,
        status=IngestionStatus.READY,
    )


def test_deep_analysis_fans_out_and_validates_citations(settings) -> None:
    repository = SQLiteRepository(settings.sqlite_path)
    vectors = MemoryVectorIndex()
    add_ready_paper(repository, vectors)
    model = FakeDeepModel()
    rag = PaperRAGService(settings, repository, vectors, model)
    service = DeepAnalysisService(rag, model)
    events: list[str] = []

    answer = service.analyze(
        "Give a detailed method overview",
        ["paper-a"],
        thread_id="deep-test-thread",
        progress_callback=events.append,
    )

    assert len(answer.facet_reports) == 2
    assert {item.focus for item in answer.facet_reports} == {"整体架构", "训练与验证"}
    assert answer.citations
    assert all(re.fullmatch(r"T\d+-C\d+", item.citation_id) for item in answer.citations)
    assert all(item.paper_id == "paper-a" for item in answer.citations)
    assert any("已拆分 2 个 specialist" in event for event in events)
    assert sum("specialist " in event for event in events) == 2
    trace = repository.get_retrieval_trace(answer.retrieval_trace_id)
    assert trace is not None
    assert trace["prompt_version"] == "paper-deep-analysis-v1"
