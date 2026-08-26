from __future__ import annotations

import pytest

from research_copilot.errors import ResearchCopilotError
from research_copilot.models import Citation, GroundedAnswerDraft, PaperChunk, RetrievedChunk
from research_copilot.rag import PaperRAGService
from research_copilot.vector_index import ChromaVectorIndex
from tests.fakes import DeterministicEmbeddings


def make_chunk(paper_id: str, version: int, page: int, text: str) -> PaperChunk:
    return PaperChunk(
        chunk_id=f"{paper_id}:v{version}:p{page:04d}:c000",
        paper_id=paper_id,
        paper_version=version,
        paper_title=paper_id,
        page_number=page,
        page_index=page - 1,
        chunk_index_on_page=0,
        text=text,
        text_hash=f"hash-{paper_id}-{version}-{page}",
        source_uri=f"{paper_id}.pdf",
        embedding_model="fake",
    )


def test_chroma_filters_paper_and_active_version(settings) -> None:
    index = ChromaVectorIndex(settings, DeterministicEmbeddings())
    index.upsert_chunks(
        [
            make_chunk("a", 1, 1, "old attention method"),
            make_chunk("a", 2, 2, "new Dataset metric"),
            make_chunk("b", 1, 3, "other attention method"),
        ]
    )
    hits = index.similarity_search("Dataset metric", {"a": 2}, 8)
    assert [item.chunk.chunk_id for item in hits] == ["a:v2:p0002:c000"]


def test_invalid_citation_is_rejected() -> None:
    citation = Citation(
        citation_id="C1",
        paper_id="a",
        paper_title="A",
        paper_version=1,
        pdf_page=1,
        chunk_id="a:v1:p0001:c000",
        evidence_text="evidence",
    )
    draft = GroundedAnswerDraft(answer="claim [C9]", used_citation_ids=["C9"])
    with pytest.raises(ResearchCopilotError, match="非法引用"):
        PaperRAGService._validate_answer_draft(draft, [citation])


def test_answer_without_citation_becomes_refusal() -> None:
    citation = Citation(
        citation_id="C1",
        paper_id="a",
        paper_title="A",
        paper_version=1,
        pdf_page=1,
        chunk_id="a:v1:p0001:c000",
        evidence_text="evidence",
    )
    result = PaperRAGService._validate_answer_draft(
        GroundedAnswerDraft(answer="unsupported claim"), [citation]
    )
    assert result.insufficient_evidence is True
    assert result.used_citation_ids == []


def test_current_chunk_id_is_safely_normalized_to_citation_id() -> None:
    citation = Citation(
        citation_id="C1",
        paper_id="a",
        paper_title="A",
        paper_version=1,
        pdf_page=1,
        chunk_id="a:v1:p0001:c000",
        evidence_text="evidence",
    )
    result = PaperRAGService._validate_answer_draft(
        GroundedAnswerDraft(
            answer="claim [a:v1:p0001:c000]",
            used_citation_ids=["a:v1:p0001:c000"],
        ),
        [citation],
    )
    assert result.answer == "claim [C1]"
    assert result.used_citation_ids == ["C1"]


def test_decorated_short_citation_is_normalized_to_real_id() -> None:
    citation = Citation(
        citation_id="C3",
        paper_id="a",
        paper_title="A",
        paper_version=1,
        pdf_page=3,
        chunk_id="a:v1:p0003:c000",
        evidence_text="evidence",
    )
    result = PaperRAGService._validate_answer_draft(
        GroundedAnswerDraft(answer="claim [C3a]"), [citation]
    )
    assert result.answer == "claim [C3]"
    assert result.used_citation_ids == ["C3"]


def test_visual_citation_is_accepted_and_normalized() -> None:
    citation = Citation(
        citation_id="F1",
        paper_id="a",
        paper_title="A",
        paper_version=1,
        pdf_page=3,
        chunk_id="a:v1:p0003:page-image",
        evidence_text="page image",
        evidence_type="page_image",
        image_path="page_0003.jpg",
    )
    result = PaperRAGService._validate_answer_draft(
        GroundedAnswerDraft(answer="visual claim [F1a]"), [citation]
    )
    assert result.answer == "visual claim [F1]"
    assert result.used_citation_ids == ["F1"]


def test_evidence_context_does_not_expose_internal_chunk_id() -> None:
    citations, context = PaperRAGService._build_evidence(
        [RetrievedChunk(chunk=make_chunk("a", 1, 1, "method evidence"), score=0.8)]
    )
    assert citations[0].chunk_id == "a:v1:p0001:c000"
    assert "a:v1:p0001:c000" not in context
    assert "[C1]" in context


def test_experiment_question_gets_deterministic_query_expansion() -> None:
    expanded = PaperRAGService._expand_retrieval_query("这篇论文实验结果如何？")
    assert "评价指标" in expanded
    assert "对比基线" in expanded
    assert PaperRAGService._expand_retrieval_query("第一作者是谁？") == "第一作者是谁？"


def test_research_question_expands_and_reranks_abstract_over_references() -> None:
    query = PaperRAGService._expand_retrieval_query("这篇论文在研究什么")
    assert "Abstract" in query
    abstract = RetrievedChunk(
        chunk=make_chunk(
            "a",
            1,
            1,
            "Abstract. Existing methods are costly. To address this, we propose a compact model.",
        ),
        score=0.65,
    )
    references = RetrievedChunk(
        chunk=make_chunk(
            "a",
            1,
            9,
            "References Smith et al. 2021. Jones et al. 2022. Lee et al. 2023. Wu et al. 2024.",
        ),
        score=0.95,
    )
    reranked = PaperRAGService._rerank_retrieved(query, [references, abstract])
    assert reranked[0].chunk.page_number == 1


def test_method_question_gets_research_query_expansion() -> None:
    query = PaperRAGService._expand_retrieval_query("这篇论文研究的方法是什么？")
    assert "整体模型架构" in query
    assert "本文提出的方法" in query


def test_cnn_classification_question_gets_architecture_query_expansion() -> None:
    query = PaperRAGService._expand_retrieval_query("它们分别属于 CNN 吗？")
    assert "backbone" in query
    assert "卷积层" in query
    assert "混合架构" in query
