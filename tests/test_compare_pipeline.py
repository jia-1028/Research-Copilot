from __future__ import annotations

import re
from pathlib import Path

from langchain_core.messages import AIMessage

from research_copilot.models import (
    ComparisonNarrative,
    EvidenceValue,
    GroundedAnswerDraft,
    IngestionStatus,
    Paper,
    PaperChunk,
    PaperProfileDraft,
    PaperVersion,
    SourceType,
)
from research_copilot.rag import DIMENSIONS, PaperRAGService
from research_copilot.storage import SQLiteRepository
from tests.fakes import MemoryVectorIndex


class StructuredInvoker:
    def __init__(self, schema):
        self.schema = schema

    def invoke(self, _messages):
        if self.schema is PaperProfileDraft:
            value = EvidenceValue(value="evidence-backed value", citation_ids=["C1"])
            return PaperProfileDraft(**{dimension: value for dimension in DIMENSIONS})
        if self.schema is ComparisonNarrative:
            return ComparisonNarrative(
                similarities=["Both provide evidence-backed methods"],
                differences=["The methods differ"],
                non_comparable_items=["Datasets differ and scores are not directly comparable"],
            )
        if self.schema is GroundedAnswerDraft:
            return GroundedAnswerDraft(
                answer="Evidence-backed answer [C1]",
                used_citation_ids=["C1"],
            )
        raise AssertionError(f"unexpected schema: {self.schema}")


class FakeStructuredModel:
    def __init__(self):
        self.calls = 0
        self.plain_calls = 0
        self.structured_kwargs: list[dict] = []
        self.last_messages = None

    def with_structured_output(self, schema, **_kwargs):
        self.calls += 1
        self.structured_kwargs.append(_kwargs)
        return StructuredInvoker(schema)

    def invoke(self, messages):
        self.plain_calls += 1
        self.last_messages = messages
        content = str(messages[-1].content)
        citation_ids = list(dict.fromkeys(re.findall(r"\[([CF]\d+)\]", content)))
        text_ids = [item for item in citation_ids if item.startswith("C")]
        figure_ids = [item for item in citation_ids if item.startswith("F")]
        used = text_ids[:1] + figure_ids or ["C1"]
        return AIMessage(
            content=(
                "Evidence-backed answer "
                + " ".join(f"[{item}]" for item in used)
                + "\n\n"
                "<EVIDENCE_STATUS>\n"
                "insufficient_evidence: false\n"
                "limitations:\n"
                "</EVIDENCE_STATUS>"
            )
        )


class TimeoutAnswerModel(FakeStructuredModel):
    def __init__(self):
        super().__init__()
        self.timeout_calls = 0

    def invoke(self, _messages):
        self.timeout_calls += 1
        raise TimeoutError("primary model timed out")



def add_ready_paper(repository, vectors, paper_id: str) -> None:
    repository.upsert_paper(
        Paper(
            paper_id=paper_id,
            title=paper_id.upper(),
            source_type=SourceType.LOCAL,
            source_uri=f"{paper_id}.pdf",
            status=IngestionStatus.PENDING,
        )
    )
    repository.add_version(
        PaperVersion(
            paper_id=paper_id,
            version=1,
            sha256=f"sha-{paper_id}",
            original_path=f"{paper_id}.pdf",
            managed_copy_path=f"managed-{paper_id}.pdf",
            parser_name="fake",
            parser_version="1",
            parsed_dir=str(repository.path.parent / "parsed" / paper_id / "v1"),
        )
    )
    chunk = PaperChunk(
        chunk_id=f"{paper_id}:v1:p0001:c000",
        paper_id=paper_id,
        paper_version=1,
        paper_title=paper_id.upper(),
        page_number=1,
        page_index=0,
        chunk_index_on_page=0,
        text=f"Evidence for {paper_id}",
        text_hash=f"hash-{paper_id}",
        source_uri=f"{paper_id}.pdf",
        embedding_model="fake",
    )
    vectors.upsert_chunks([chunk])
    repository.finish_version(
        paper_id,
        1,
        parser_name="fake",
        parser_version="1",
        page_count=1,
        chunk_count=1,
        status=IngestionStatus.READY,
    )


def test_multi_paper_compare_has_nine_rows_and_unique_citations(settings) -> None:
    repository = SQLiteRepository(settings.sqlite_path)
    vectors = MemoryVectorIndex()
    add_ready_paper(repository, vectors, "paper-a")
    add_ready_paper(repository, vectors, "paper-b")
    model = FakeStructuredModel()
    service = PaperRAGService(settings, repository, vectors, model)
    comparison = service.compare(["paper-a", "paper-b"])
    assert len(comparison.rows) == 9
    assert comparison.paper_ids == ["paper-a", "paper-b"]
    ids = [citation.citation_id for citation in comparison.citations]
    assert ids == ["paper-a:C1", "paper-b:C1"]
    assert len(ids) == len(set(ids))
    assert comparison.similarities
    assert model.calls == 2

    cached_comparison = service.compare(["paper-a", "paper-b"])
    assert len(cached_comparison.rows) == 9
    assert model.calls == 2  # the complete comparison is returned from SQLite cache


def test_full_summary_uses_compact_maps_and_local_reduce(settings) -> None:
    repository = SQLiteRepository(settings.sqlite_path)
    vectors = MemoryVectorIndex()
    add_ready_paper(repository, vectors, "paper-a")
    model = FakeStructuredModel()
    service = PaperRAGService(settings, repository, vectors, model)

    profile = service.summarize("paper-a")

    assert profile.research_problem.value == "evidence-backed value"
    assert profile.research_problem.citation_ids == ["paper-a:C1"]
    assert model.calls == 1  # no long, second LLM Reduce response is generated


def test_full_summary_splits_a_length_limited_map_batch(settings) -> None:
    repository = SQLiteRepository(settings.sqlite_path)
    vectors = MemoryVectorIndex()
    add_ready_paper(repository, vectors, "paper-a")
    vectors.upsert_chunks(
        [
            PaperChunk(
                chunk_id=f"paper-a:v1:p{page:04d}:c000",
                paper_id="paper-a",
                paper_version=1,
                paper_title="PAPER-A",
                page_number=page,
                page_index=page - 1,
                chunk_index_on_page=0,
                text=f"Evidence on page {page}",
                text_hash=f"summary-hash-{page}",
                source_uri="paper-a.pdf",
                embedding_model="fake",
            )
            for page in range(2, 11)
        ]
    )

    class LengthLimitedInvoker:
        def __init__(self, model):
            self.model = model

        def invoke(self, messages):
            self.model.calls += 1
            citation_ids = re.findall(r"\[(C\d+)\]", str(messages[-1].content))
            if len(citation_ids) > 8:
                raise RuntimeError(
                    "Could not parse response content as the length limit was reached"
                )
            value = EvidenceValue(value="compact evidence", citation_ids=[citation_ids[0]])
            return PaperProfileDraft(**{dimension: value for dimension in DIMENSIONS})

    class LengthLimitedModel:
        def __init__(self):
            self.calls = 0

        def with_structured_output(self, schema, **_kwargs):
            assert schema is PaperProfileDraft
            return LengthLimitedInvoker(self)

    model = LengthLimitedModel()
    service = PaperRAGService(settings, repository, vectors, model)

    profile = service.summarize("paper-a")

    assert model.calls == 3  # 10 chunks fail once, then two 5-chunk maps succeed
    assert profile.method_architecture.value == "compact evidence"
    assert profile.method_architecture.citation_ids == ["paper-a:C1"]


def test_paper_question_reports_observable_progress(settings) -> None:
    repository = SQLiteRepository(settings.sqlite_path)
    vectors = MemoryVectorIndex()
    add_ready_paper(repository, vectors, "paper-a")
    service = PaperRAGService(settings, repository, vectors, FakeStructuredModel())
    events: list[str] = []

    answer = service.ask(
        "What method does this paper use?",
        ["paper-a"],
        progress_callback=events.append,
    )

    assert answer.answer == "Evidence-backed answer [C1]"
    assert any("论文状态" in event for event in events)
    assert any("向量检索" in event for event in events)
    assert any("证据块" in event for event in events)
    assert any("带引用回答" in event for event in events)
    assert any("引用校验完成" in event for event in events)
    assert service.chat_model.plain_calls == 1


def test_multi_paper_question_embeds_and_searches_only_once(settings) -> None:
    repository = SQLiteRepository(settings.sqlite_path)
    vectors = MemoryVectorIndex()
    add_ready_paper(repository, vectors, "paper-a")
    add_ready_paper(repository, vectors, "paper-b")
    service = PaperRAGService(settings, repository, vectors, FakeStructuredModel())

    answer = service.ask("Are these architectures CNNs?", ["paper-a", "paper-b"])

    assert answer.answer == "Evidence-backed answer [C1]"
    assert vectors.search_calls == 1
    assert {item.paper_id for item in answer.citations} == {"paper-a"}


def test_evidence_expansion_uses_two_retrieval_routes_and_links_traces(settings) -> None:
    repository = SQLiteRepository(settings.sqlite_path)
    vectors = MemoryVectorIndex()
    add_ready_paper(repository, vectors, "paper-a")
    service = PaperRAGService(settings, repository, vectors, FakeStructuredModel())
    initial = service.ask("What backbone is used?", ["paper-a"])
    events: list[str] = []

    expanded = service.ask(
        "What backbone is used?",
        ["paper-a"],
        evidence_hints=["The exact backbone variant is missing"],
        previous_trace_id=initial.retrieval_trace_id,
        evidence_expansion_attempt=1,
        progress_callback=events.append,
    )

    assert expanded.evidence_expansion_attempt == 1
    assert expanded.source_trace_id == initial.retrieval_trace_id
    assert vectors.search_calls == 3  # one initial route + two expansion routes
    trace = repository.get_retrieval_trace(expanded.retrieval_trace_id)
    assert trace is not None
    assert trace["prompt_version"] == "paper-qa-evidence-expansion-v1"
    assert any("两路逐论文检索" in event for event in events)
    assert any("不在上一轮候选" in event for event in events)


def test_paper_question_falls_back_after_primary_timeout(settings) -> None:
    repository = SQLiteRepository(settings.sqlite_path)
    vectors = MemoryVectorIndex()
    add_ready_paper(repository, vectors, "paper-a")
    fallback = FakeStructuredModel()
    primary = TimeoutAnswerModel()
    service = PaperRAGService(
        settings,
        repository,
        vectors,
        primary,
        fallback_chat_model=fallback,
    )
    events: list[str] = []

    answer = service.ask(
        "What method does this paper use?",
        ["paper-a"],
        progress_callback=events.append,
    )

    assert answer.answer == "Evidence-backed answer [C1]"
    assert answer.fallback_used is True
    assert answer.generation_model == "qwen-plus"
    assert fallback.plain_calls == 1
    assert any("正在切换" in event and "qwen-plus" in event for event in events)

    second_events: list[str] = []
    second = service.ask(
        "Explain the method again.",
        ["paper-a"],
        progress_callback=second_events.append,
    )
    assert second.fallback_used is True
    assert primary.timeout_calls == 1
    assert fallback.plain_calls == 2
    assert any("熔断保护期" in event for event in second_events)


def test_method_question_sends_retrieved_page_image_and_returns_figure_citation(
    settings,
) -> None:
    repository = SQLiteRepository(settings.sqlite_path)
    vectors = MemoryVectorIndex()
    add_ready_paper(repository, vectors, "paper-a")
    version = repository.get_version("paper-a", 1)
    assert version is not None
    image_dir = Path(version["parsed_dir"]) / "page_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / "page_0001.jpg").write_bytes(b"fake-jpeg")
    model = FakeStructuredModel()
    service = PaperRAGService(settings, repository, vectors, model)

    answer = service.ask("请结合架构图解释这篇论文的方法", ["paper-a"])

    assert {item.citation_id for item in answer.citations} == {"C1", "F1"}
    figure = next(item for item in answer.citations if item.citation_id == "F1")
    assert figure.evidence_type == "page_image"
    assert figure.pdf_page == 1
    content = model.last_messages[-1].content
    assert isinstance(content, list)
    assert any(part.get("type") == "image_url" for part in content)
    assert any("[F1]" in part.get("text", "") for part in content)
