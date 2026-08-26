from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


class SourceType(StrEnum):
    LOCAL = "local"
    ARXIV = "arxiv"


class DocumentRole(StrEnum):
    MAIN = "main"
    SUPPLEMENTARY = "supplementary"


class IngestionStatus(StrEnum):
    PENDING = "pending"
    VALIDATING = "validating"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"


class ConversationScopeType(StrEnum):
    GENERAL = "general"
    PAPER_FAMILY = "paper_family"
    PAPER_SET = "paper_set"


class ConversationMode(StrEnum):
    QUICK = "quick"
    STANDARD_AGENT = "standard_agent"
    DEEP_ANALYSIS = "deep_analysis"
    EVIDENCE_EXPANSION = "evidence_expansion"


class ConversationMessageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"
    LEGACY_INCOMPLETE = "legacy_incomplete"


class Paper(BaseModel):
    paper_id: str
    title: str
    source_type: SourceType
    source_uri: str
    document_role: DocumentRole = DocumentRole.MAIN
    parent_paper_id: str | None = None
    active_version: int = 0
    status: IngestionStatus = IngestionStatus.PENDING
    authors: list[str] = Field(default_factory=list)
    abstract: str | None = None
    arxiv_id: str | None = None
    page_count: int = 0
    chunk_count: int = 0
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class PaperVersion(BaseModel):
    paper_id: str
    version: int
    sha256: str
    original_path: str
    managed_copy_path: str
    parser_name: str
    parser_version: str
    parsed_dir: str
    page_count: int = 0
    chunk_count: int = 0
    status: IngestionStatus = IngestionStatus.PENDING
    created_at: datetime = Field(default_factory=utc_now)


class ParsedPage(BaseModel):
    page_number: int = Field(ge=1)
    text: str


class ParsedPageImage(BaseModel):
    page_number: int = Field(ge=1)
    image_path: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    sha256: str


class ParsedPaper(BaseModel):
    parser_name: str
    parser_version: str
    pages: list[ParsedPage]
    markdown_path: str | None = None
    content_list_path: str | None = None
    page_images: list[ParsedPageImage] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    manifest: dict[str, Any] = Field(default_factory=dict)


class PaperChunk(BaseModel):
    chunk_id: str
    paper_id: str
    paper_version: int
    paper_title: str
    page_number: int = Field(ge=1)
    page_index: int = Field(ge=0)
    chunk_index_on_page: int = Field(ge=0)
    text: str
    text_hash: str
    section_hint: str | None = None
    source_uri: str
    embedding_model: str

    def metadata(self) -> dict[str, str | int]:
        return {
            "paper_id": self.paper_id,
            "paper_version": self.paper_version,
            "paper_title": self.paper_title,
            "page_number": self.page_number,
            "page_index": self.page_index,
            "chunk_index_on_page": self.chunk_index_on_page,
            "text_hash": self.text_hash,
            "section_hint": self.section_hint or "",
            "source_uri": self.source_uri,
            "embedding_model": self.embedding_model,
        }


class RetrievedChunk(BaseModel):
    chunk: PaperChunk
    score: float | None = None


class Citation(BaseModel):
    citation_id: str
    paper_id: str
    paper_title: str
    paper_version: int
    pdf_page: int
    chunk_id: str
    evidence_text: str
    retrieval_score: float | None = None
    evidence_type: Literal["text", "page_image"] = "text"
    image_path: str | None = None


class GroundedAnswerDraft(BaseModel):
    answer: str
    used_citation_ids: list[str] = Field(default_factory=list)
    insufficient_evidence: bool = False
    limitations: list[str] = Field(default_factory=list)


class GroundedAnswer(GroundedAnswerDraft):
    citations: list[Citation] = Field(default_factory=list)
    retrieval_trace_id: str
    evidence_expansion_attempt: int = Field(default=0, ge=0, le=2)
    source_trace_id: str | None = None
    generation_model: str | None = None
    fallback_used: bool = False


class DeepAnalysisTaskDraft(BaseModel):
    focus: str = Field(min_length=2, max_length=80)
    question: str = Field(min_length=4, max_length=500)
    retrieval_query: str = Field(min_length=4, max_length=500)


class DeepAnalysisPlan(BaseModel):
    tasks: list[DeepAnalysisTaskDraft] = Field(min_length=2, max_length=3)


class DeepFacetReport(GroundedAnswerDraft):
    task_id: str
    focus: str


class DeepAnalysisAnswer(GroundedAnswer):
    facet_reports: list[DeepFacetReport] = Field(default_factory=list)


class EvidenceValue(BaseModel):
    value: str
    citation_ids: list[str] = Field(default_factory=list)
    insufficient_evidence: bool = False


class PaperProfile(BaseModel):
    paper_id: str
    paper_title: str
    research_problem: EvidenceValue
    core_contributions: EvidenceValue
    method_architecture: EvidenceValue
    datasets: EvidenceValue
    experimental_setup: EvidenceValue
    metrics: EvidenceValue
    main_results: EvidenceValue
    efficiency: EvidenceValue
    limitations: EvidenceValue


class PaperProfileDraft(BaseModel):
    """LLM-facing profile without repository-owned paper identity fields."""

    research_problem: EvidenceValue
    core_contributions: EvidenceValue
    method_architecture: EvidenceValue
    datasets: EvidenceValue
    experimental_setup: EvidenceValue
    metrics: EvidenceValue
    main_results: EvidenceValue
    efficiency: EvidenceValue
    limitations: EvidenceValue


class ComparisonNarrative(BaseModel):
    similarities: list[str] = Field(default_factory=list)
    differences: list[str] = Field(default_factory=list)
    non_comparable_items: list[str] = Field(default_factory=list)


class ComparisonRow(BaseModel):
    dimension: str
    values: dict[str, EvidenceValue]


class PaperComparison(BaseModel):
    paper_ids: list[str]
    rows: list[ComparisonRow]
    similarities: list[str] = Field(default_factory=list)
    differences: list[str] = Field(default_factory=list)
    non_comparable_items: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)


class ArxivPaper(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published_at: datetime | None = None
    updated_at: datetime | None = None
    categories: list[str] = Field(default_factory=list)
    entry_url: str
    pdf_url: str
    already_imported: bool = False


class IngestionResult(BaseModel):
    job_id: str
    paper_id: str
    version: int
    status: IngestionStatus
    page_count: int = 0
    chunk_count: int = 0
    duplicate: bool = False
    parser_name: str | None = None
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


class ResearchAgentState(BaseModel):
    active_paper_ids: list[str] = Field(default_factory=list)
    last_arxiv_result_ids: list[str] = Field(default_factory=list)
    last_retrieval_trace_id: str | None = None
    pending_ingestion_job_id: str | None = None
    conversation_summary: str | None = None


class ConversationScope(BaseModel):
    scope_type: ConversationScopeType
    scope_key: str
    root_paper_ids: list[str] = Field(default_factory=list)
    effective_paper_ids: list[str] = Field(default_factory=list)
    paper_snapshots: list[dict[str, Any]] = Field(default_factory=list)


class ConversationMessage(BaseModel):
    message_id: str
    turn_id: str
    conversation_id: str
    sequence: int
    role: Literal["user", "assistant"]
    mode: ConversationMode
    status: ConversationMessageStatus
    content: str
    original_query: str | None = None
    standalone_query: str | None = None
    process: list[str] = Field(default_factory=list)
    payload: dict[str, Any] | None = None
    error: str | None = None
    retrieval_trace_id: str | None = None
    created_at: datetime
    completed_at: datetime | None = None


class StandaloneQuestion(BaseModel):
    needs_context: bool
    standalone_question: str = Field(min_length=1, max_length=2000)
    referenced_turn_ids: list[str] = Field(default_factory=list)


class ConversationSummary(BaseModel):
    user_goal: str = ""
    topics: list[str] = Field(default_factory=list)
    term_references: dict[str, str] = Field(default_factory=dict)
    preferences: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


ComparisonDimension = Literal[
    "research_problem",
    "core_contributions",
    "method_architecture",
    "datasets",
    "experimental_setup",
    "metrics",
    "main_results",
    "efficiency",
    "limitations",
]
