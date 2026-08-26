from __future__ import annotations

from datetime import UTC, datetime
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
from research_copilot.middleware import is_transient_model_error
from research_copilot.models import GroundedAnswer
from research_copilot.tools import AskPapersInput, ComparePapersInput, SearchArxivInput, build_tools


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


def test_tool_input_limits() -> None:
    assert SearchArxivInput(query="agents", max_results=20).max_results == 20
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
