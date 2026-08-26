from __future__ import annotations

from datetime import UTC, datetime
from io import BytesIO
from types import SimpleNamespace
from urllib.error import HTTPError

import httpx
import pytest
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from research_copilot.arxiv_service import ArxivService, normalize_arxiv_id
from research_copilot.errors import ArxivTemporarilyUnavailableError, ResearchCopilotError
from research_copilot.middleware import (
    ArxivImportRoutingMiddleware,
    _arxiv_import_plan,
    is_transient_model_error,
)
from research_copilot.models import ArxivPaper, GroundedAnswer, IngestionResult, IngestionStatus
from research_copilot.tools import (
    AskPapersInput,
    ComparePapersInput,
    FindArxivImportCandidateInput,
    SearchArxivInput,
    build_tools,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2401.01234", "2401.01234"),
        ("https://arxiv.org/abs/2401.01234v2", "2401.01234"),
        ("https://arxiv.org/pdf/2401.01234.pdf", "2401.01234"),
        ("hep-th/9901001", "hep-th/9901001"),
    ],
)
def test_normalize_arxiv_id(value: str, expected: str) -> None:
    assert normalize_arxiv_id(value) == expected


def test_invalid_arxiv_id() -> None:
    with pytest.raises(ResearchCopilotError):
        normalize_arxiv_id("not-an-id")


def _arxiv_service(settings) -> ArxivService:
    return ArxivService(
        settings,
        SimpleNamespace(arxiv_id_exists=lambda _arxiv_id: False),
        None,  # type: ignore[arg-type]
    )


def test_arxiv_search_uses_one_sdk_attempt_and_caches_success(settings) -> None:
    service = _arxiv_service(settings)
    assert service.client.num_retries == 0
    assert service.client.delay_seconds == 3.0
    result = SimpleNamespace(
        entry_id="https://arxiv.org/abs/2401.01234v2",
        title=" Mamba  Medical Segmentation ",
        authors=[SimpleNamespace(name="Ada")],
        summary=" An abstract ",
        published=datetime(2024, 1, 1, tzinfo=UTC),
        updated=datetime(2024, 1, 2, tzinfo=UTC),
        categories=["cs.CV"],
        pdf_url="https://arxiv.org/pdf/2401.01234",
    )

    class FakeClient:
        num_retries = 0
        calls = 0

        def results(self, _search):
            self.calls += 1
            return iter([result])

    service.client = FakeClient()
    first = service.search("Mamba medical segmentation")
    second = service.search("  mamba   medical segmentation  ")

    assert service.client.calls == 1
    assert first[0].arxiv_id == "2401.01234"
    assert second[0].title == "Mamba Medical Segmentation"


def test_arxiv_rate_limit_starts_cooldown_without_second_request(settings) -> None:
    service = _arxiv_service(settings)

    class RateLimitedClient:
        calls = 0

        def results(self, _search):
            self.calls += 1
            raise HTTPError("https://export.arxiv.org/api/query", 429, "Too Many Requests", None, None)

    service.client = RateLimitedClient()
    with pytest.raises(ArxivTemporarilyUnavailableError, match="HTTP 429"):
        service.search("Mamba medical segmentation")
    with pytest.raises(ArxivTemporarilyUnavailableError, match="本轮重复搜索"):
        service.search("another mamba query")

    assert service.client.calls == 1


def test_find_import_candidate_skips_existing_papers_and_enforces_year(settings) -> None:
    service = _arxiv_service(settings)
    existing = ArxivPaper(
        arxiv_id="2601.00001",
        title="Lightweight Mamba for Medical Image Segmentation",
        authors=["Ada"],
        abstract="An efficient Mamba medical image segmentation model.",
        published_at=datetime(2026, 1, 2, tzinfo=UTC),
        updated_at=datetime(2026, 1, 2, tzinfo=UTC),
        categories=["cs.CV"],
        entry_url="https://arxiv.org/abs/2601.00001",
        pdf_url="https://arxiv.org/pdf/2601.00001",
        already_imported=True,
    )
    fresh = existing.model_copy(
        update={
            "arxiv_id": "2602.00002",
            "title": "Compact Mamba for Medical Image Segmentation",
            "entry_url": "https://arxiv.org/abs/2602.00002",
            "pdf_url": "https://arxiv.org/pdf/2602.00002",
            "already_imported": False,
        }
    )
    wrong_year = fresh.model_copy(
        update={
            "arxiv_id": "2501.00003",
            "published_at": datetime(2025, 1, 1, tzinfo=UTC),
        }
    )
    service.search = lambda *_args, **_kwargs: [existing, wrong_year, fresh]  # type: ignore[method-assign]

    candidate, already_imported = service.find_import_candidate(
        "all:Mamba", target_year=2026, require_lightweight=True
    )

    assert candidate is not None
    assert candidate.arxiv_id == "2602.00002"
    assert [paper.arxiv_id for paper in already_imported] == ["2601.00001"]


def test_find_import_route_supports_chinese_year_and_lightweight_intent() -> None:
    plan = _arxiv_import_plan("帮我找一篇二六年的轻量化Mamba医学图像分割论文并添加到论文库中")

    assert plan is not None
    assert plan["target_year"] == 2026
    assert plan["require_lightweight"] is True
    assert "submittedDate:[202601010000 TO 202612312359]" in str(plan["query"])


def test_arxiv_import_downloads_result_pdf_url_without_sdk_download_helper(settings, monkeypatch) -> None:
    imported: dict[str, object] = {}

    class StubIngestion:
        def ingest_local(self, pdf_path, **kwargs):
            imported["pdf_bytes"] = pdf_path.read_bytes()
            imported["pdf_path"] = pdf_path
            imported["kwargs"] = kwargs
            return IngestionResult(
                job_id="job-a",
                paper_id="arxiv-2403.05246",
                version=1,
                status=IngestionStatus.READY,
            )

    service = ArxivService(
        settings,
        SimpleNamespace(arxiv_id_exists=lambda _arxiv_id: False),
        StubIngestion(),
    )
    result = SimpleNamespace(
        entry_id="https://arxiv.org/abs/2403.05246v1",
        title="LightM UNet",
        authors=[SimpleNamespace(name="Ada")],
        summary="A lightweight segmentation method.",
        pdf_url="https://arxiv.org/pdf/2403.05246",
    )

    class FakeClient:
        def results(self, _search):
            return iter([result])

    class FakeResponse:
        def __init__(self):
            self.buffer = BytesIO(b"%PDF-1.7\nexample paper")

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, size: int) -> bytes:
            return self.buffer.read(size)

    captured: dict[str, object] = {}

    def fake_urlopen(request, *, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["user_agent"] = request.get_header("User-agent")
        return FakeResponse()

    service.client = FakeClient()
    monkeypatch.setattr("research_copilot.arxiv_service.urlopen", fake_urlopen)

    outcome = service.import_paper("2403.05246")

    assert outcome.status is IngestionStatus.READY
    assert captured["url"] == result.pdf_url
    assert captured["user_agent"] == "Research-Copilot/0.2 (+local-paper-rag)"
    assert captured["timeout"] == 60
    assert imported["pdf_bytes"] == b"%PDF-1.7\nexample paper"
    assert imported["pdf_path"].suffix == ".pdf"
    assert imported["kwargs"]["arxiv_id"] == "2403.05246"


def test_tool_input_limits() -> None:
    assert SearchArxivInput(query="agents", max_results=20).max_results == 20
    assert FindArxivImportCandidateInput(query="all:Mamba", target_year=2026).target_year == 2026
    assert len(AskPapersInput(question="q", paper_ids=["a"]).paper_ids) == 1
    with pytest.raises(ValueError):
        ComparePapersInput(paper_ids=["a"])


def test_retry_policy_is_narrow() -> None:
    assert is_transient_model_error(httpx.TimeoutException("timeout"))
    assert not is_transient_model_error(ValueError("bad schema"))


def test_settings_repr_does_not_leak_secret(settings) -> None:
    rendered = repr(settings)
    assert "test-key" not in rendered
    assert "test-mineru-token" not in rendered


def test_read_tool_limit_blocks_duplicates_without_aborting_run() -> None:
    limiter = ToolCallLimitMiddleware(
        tool_name="ask_papers",
        run_limit=1,
        exit_behavior="continue",
    )
    calls = [
        {
            "name": "ask_papers",
            "args": {"question": f"detail {index}", "paper_ids": ["paper-a"]},
            "id": f"call-{index}",
            "type": "tool_call",
        }
        for index in range(4)
    ]

    update = limiter.after_model(
        {"messages": [AIMessage(content="", tool_calls=calls)]},
        runtime=None,  # type: ignore[arg-type]
    )

    assert update is not None
    blocked = [message for message in update["messages"] if isinstance(message, ToolMessage)]
    assert len(blocked) == 3
    assert all(message.status == "error" for message in blocked)
    assert update["run_tool_call_count"]["ask_papers"] == 4


def test_search_tool_limit_blocks_duplicate_calls_without_aborting_run() -> None:
    limiter = ToolCallLimitMiddleware(
        tool_name="search_arxiv",
        run_limit=1,
        exit_behavior="continue",
    )
    calls = [
        {
            "name": "search_arxiv",
            "args": {"query": "Mamba medical segmentation"},
            "id": f"call-{index}",
            "type": "tool_call",
        }
        for index in range(2)
    ]

    update = limiter.after_model(
        {"messages": [AIMessage(content="", tool_calls=calls)]},
        runtime=None,  # type: ignore[arg-type]
    )

    assert update is not None
    blocked = [message for message in update["messages"] if isinstance(message, ToolMessage)]
    assert len(blocked) == 1
    assert blocked[0].status == "error"


def test_ask_papers_returns_direct_artifact_without_exposing_runtime() -> None:
    ask_tool = next(
        item
        for item in build_tools(SimpleNamespace())  # type: ignore[arg-type]
        if item.name == "ask_papers"
    )

    assert ask_tool.return_direct is True
    assert ask_tool.response_format == "content_and_artifact"
    assert "runtime" not in ask_tool.args


def test_ask_papers_tool_runtime_returns_answer_and_full_artifact() -> None:
    class StubRag:
        def ask(self, question, paper_ids, *, thread_id=None, progress_callback=None):
            assert question == "Is it a CNN?"
            assert paper_ids == ["paper-a"]
            assert thread_id == "thread-a"
            progress_callback("retrieval complete")
            return GroundedAnswer(
                answer="Hybrid CNN architecture [C1]",
                used_citation_ids=["C1"],
                citations=[],
                retrieval_trace_id="trace-a",
            )

    ask_tool = next(
        item
        for item in build_tools(SimpleNamespace(rag=StubRag()))  # type: ignore[arg-type]
        if item.name == "ask_papers"
    )
    builder = StateGraph(MessagesState)
    builder.add_node("tools", ToolNode([ask_tool]))
    builder.add_edge(START, "tools")
    graph = builder.compile()
    result = graph.invoke(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "ask_papers",
                            "args": {
                                "question": "Is it a CNN?",
                                "paper_ids": ["paper-a"],
                                "thread_id": "thread-a",
                            },
                            "id": "call-a",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        }
    )
    message = result["messages"][-1]

    assert message.content == "Hybrid CNN architecture [C1]"
    assert message.artifact["retrieval_trace_id"] == "trace-a"


def test_find_import_candidate_tool_does_not_queue_until_human_approves() -> None:
    candidate = ArxivPaper(
        arxiv_id="2602.00002",
        title="Compact Mamba for Medical Image Segmentation",
        authors=["Ada"],
        abstract="An efficient medical image segmentation model.",
        published_at=datetime(2026, 2, 1, tzinfo=UTC),
        updated_at=datetime(2026, 2, 1, tzinfo=UTC),
        categories=["cs.CV"],
        entry_url="https://arxiv.org/abs/2602.00002",
        pdf_url="https://arxiv.org/pdf/2602.00002",
    )
    class StubArxiv:
        def find_import_candidate(self, *_args, **_kwargs):
            return candidate, []

    tool_to_run = next(
        item
        for item in build_tools(
            SimpleNamespace(arxiv=StubArxiv())
        )
        if item.name == "find_arxiv_import_candidate"
    )
    builder = StateGraph(MessagesState)
    builder.add_node("tools", ToolNode([tool_to_run]))
    builder.add_edge(START, "tools")
    graph = builder.compile()
    result = graph.invoke(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "find_arxiv_import_candidate",
                            "args": {
                                "query": "all:Mamba",
                                "target_year": 2026,
                                "require_lightweight": True,
                            },
                            "id": "call-a",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        }
    )
    message = result["messages"][-1]

    assert tool_to_run.return_direct is False
    assert "Compact Mamba" in message.content
    assert "candidate_found" in message.content


def test_candidate_result_routes_to_human_approved_import_call() -> None:
    middleware = ArxivImportRoutingMiddleware()
    request = SimpleNamespace(
        messages=[
            ToolMessage(
                name="find_arxiv_import_candidate",
                tool_call_id="find-1",
                content=(
                    '{"status":"candidate_found","selected_paper":'
                    '{"arxiv_id":"2602.00002","title":"Compact Mamba"}}'
                ),
            )
        ]
    )

    response = middleware.wrap_model_call(
        request,  # type: ignore[arg-type]
        lambda _request: pytest.fail("model should not be called after candidate selection"),
    )

    call = response.result[0].tool_calls[0]
    assert call["name"] == "import_arxiv_paper"
    assert call["args"] == {
        "arxiv_id_or_url": "2602.00002",
        "expected_title": "Compact Mamba",
    }
