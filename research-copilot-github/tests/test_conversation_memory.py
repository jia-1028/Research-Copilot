from __future__ import annotations

import sqlite3
from contextlib import closing

import pytest

from research_copilot.conversation_memory import ConversationMemoryService
from research_copilot.errors import ResearchCopilotError
from research_copilot.models import (
    ConversationMode,
    ConversationSummary,
    DocumentRole,
    IngestionStatus,
    Paper,
    SourceType,
    StandaloneQuestion,
)
from research_copilot.storage import SCHEMA_VERSION, SQLiteRepository


class FakeCheckpointer:
    def __init__(self):
        self.deleted: list[str] = []

    def delete_thread(self, thread_id: str) -> None:
        self.deleted.append(thread_id)

    def get_tuple(self, _config):
        return None


class StructuredInvoker:
    def __init__(self, schema):
        self.schema = schema

    def invoke(self, _messages):
        if self.schema is StandaloneQuestion:
            return StandaloneQuestion(
                needs_context=True,
                standalone_question="示例主论文的方法创新有哪些？",
            )
        if self.schema is ConversationSummary:
            return ConversationSummary(user_goal="理解论文", topics=["方法"])
        raise AssertionError(self.schema)


class FakeMemoryModel:
    def __init__(self):
        self.calls: list[type] = []

    def with_structured_output(self, schema, **_kwargs):
        self.calls.append(schema)
        return StructuredInvoker(schema)


def add_papers(repository: SQLiteRepository) -> None:
    repository.upsert_paper(
        Paper(
            paper_id="example-main",
            title="Example Main Paper",
            source_type=SourceType.LOCAL,
            source_uri="main.pdf",
            active_version=1,
            status=IngestionStatus.READY,
        )
    )
    repository.upsert_paper(
        Paper(
            paper_id="example-main-supp",
            title="Example Main Paper Supplement",
            source_type=SourceType.LOCAL,
            source_uri="supp.pdf",
            document_role=DocumentRole.SUPPLEMENTARY,
            parent_paper_id="example-main",
            active_version=1,
            status=IngestionStatus.READY,
        )
    )
    repository.upsert_paper(
        Paper(
            paper_id="example-secondary",
            title="Example Secondary Paper",
            source_type=SourceType.LOCAL,
            source_uri="vmoe.pdf",
            active_version=1,
            status=IngestionStatus.READY,
        )
    )


def build_memory(settings):
    repository = SQLiteRepository(settings.sqlite_path)
    add_papers(repository)
    model = FakeMemoryModel()
    checkpointer = FakeCheckpointer()
    return repository, model, checkpointer, ConversationMemoryService(
        repository, model, checkpointer
    )


def test_scope_normalizes_order_and_supplement_family(settings) -> None:
    _repository, _model, _checkpointer, memory = build_memory(settings)
    main = memory.resolve_scope(["example-main"])
    supplement = memory.resolve_scope(["example-main-supp"])
    first = memory.resolve_scope(["example-secondary", "example-main-supp"])
    second = memory.resolve_scope(["example-main", "example-secondary"])

    assert main.scope_key == supplement.scope_key == "paper:example-main"
    assert main.effective_paper_ids == ["example-main", "example-main-supp"]
    assert first.scope_key == second.scope_key == "papers:example-main|example-secondary"


def test_turn_persists_payload_citations_and_failure(settings) -> None:
    repository, _model, _checkpointer, memory = build_memory(settings)
    conversation = memory.create_conversation(memory.resolve_scope(["example-main"]))
    turn = memory.begin_turn(
        conversation["thread_id"],
        mode=ConversationMode.QUICK,
        original_query="方法是什么？",
        standalone_query="示例主论文的方法是什么？",
    )
    payload = {
        "retrieval_trace_id": "trace-1",
        "citations": [
            {
                "citation_id": "C1",
                "paper_id": "example-main",
                "paper_title": "Example Main Paper",
                "paper_version": 1,
                "pdf_page": 3,
                "chunk_id": "example-main:v1:p0003:c000",
                "evidence_text": "method evidence",
                "retrieval_score": 0.9,
            }
        ],
    }
    memory.complete_turn(
        turn["assistant_message_id"],
        content="回答 [C1]",
        process=["检索完成"],
        payload=payload,
    )
    messages = memory.messages(conversation["thread_id"])
    assert [item["role"] for item in messages] == ["user", "assistant"]
    assert messages[-1]["payload"]["schema_version"] == 1
    assert messages[-1]["status"] == "completed"

    second = memory.begin_turn(
        conversation["thread_id"],
        mode=ConversationMode.DEEP_ANALYSIS,
        original_query="继续",
        standalone_query="继续分析方法",
    )
    memory.fail_turn(second["assistant_message_id"], error="timeout", process=[])
    assert memory.messages(conversation["thread_id"])[-1]["status"] == "failed"

    with repository.connect() as conn:
        citation = conn.execute("SELECT * FROM message_citations").fetchone()
    assert citation["pdf_page"] == 3


def test_only_referential_question_calls_context_model(settings) -> None:
    _repository, model, _checkpointer, memory = build_memory(settings)
    conversation = memory.create_conversation(memory.resolve_scope(["example-main"]))
    turn = memory.begin_turn(
        conversation["thread_id"],
        mode=ConversationMode.QUICK,
        original_query="示例主论文的方法是什么？",
        standalone_query="示例主论文的方法是什么？",
    )
    memory.complete_turn(
        turn["assistant_message_id"], content="它使用稀疏 token。", process=[]
    )

    assert memory.resolve_question(conversation["thread_id"], "实验结果如何？") == "实验结果如何？"
    assert model.calls == []
    assert memory.resolve_question(conversation["thread_id"], "它的创新呢？") == "示例主论文的方法创新有哪些？"
    assert model.calls == [StandaloneQuestion]


def test_deleted_paper_conversation_becomes_read_only(settings) -> None:
    repository, _model, _checkpointer, memory = build_memory(settings)
    conversation = memory.create_conversation(memory.resolve_scope(["example-main"]))
    assert repository.mark_paper_conversations_read_only("example-main", "Example Main Paper") == 1
    current = repository.get_conversation(conversation["thread_id"])
    assert current["archived_at"]
    assert current["read_only_reason"]
    with pytest.raises(ResearchCopilotError, match="论文已删除"):
        memory.begin_turn(
            conversation["thread_id"],
            mode=ConversationMode.QUICK,
            original_query="继续",
            standalone_query="继续",
        )


def test_permanent_delete_removes_checkpoint(settings) -> None:
    repository, _model, checkpointer, memory = build_memory(settings)
    conversation = memory.create_conversation(memory.resolve_scope([]))
    memory.delete_permanently(conversation["thread_id"])
    assert repository.get_conversation(conversation["thread_id"]) is None
    assert checkpointer.deleted == [conversation["thread_id"]]


def test_legacy_schema_is_backed_up_and_migrated(settings) -> None:
    path = settings.sqlite_path
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(
            """
            CREATE TABLE conversations(
                thread_id TEXT PRIMARY KEY,title TEXT NOT NULL,
                active_paper_ids_json TEXT NOT NULL DEFAULT '[]',
                last_arxiv_result_ids_json TEXT NOT NULL DEFAULT '[]',
                last_retrieval_trace_id TEXT,pending_ingestion_job_id TEXT,
                created_at TEXT NOT NULL,updated_at TEXT NOT NULL
            )
            """
        )
    repository = SQLiteRepository(path)
    with repository.connect() as conn:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        columns = {row[1] for row in conn.execute("PRAGMA table_info(conversations)")}
    assert version == SCHEMA_VERSION
    assert {"scope_key", "summary_json", "pending_turn_id"} <= columns
    assert list((path.parent / "backups").glob("app-*.sqlite.bak"))
