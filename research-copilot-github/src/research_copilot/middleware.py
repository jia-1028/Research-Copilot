from __future__ import annotations

import json
import logging
import re
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from research_copilot.errors import ResearchCopilotError
from research_copilot.services import ServiceContainer

LOGGER = logging.getLogger("research_copilot.trace")

_FIND_WORDS = ("找", "搜索", "检索", "查找", "find", "search", "look for")
_IMPORT_WORDS = ("导入", "添加", "加入", "import", "add")


def _arxiv_import_plan(text: str) -> dict[str, object] | None:
    """Return a deterministic plan for a high-confidence find-and-import request."""

    lowered = text.casefold()
    wants_find = any(word in lowered for word in _FIND_WORDS)
    wants_import = any(word in lowered for word in _IMPORT_WORDS)
    is_mamba = "mamba" in lowered
    is_medical_segmentation = (
        ("医学" in text or "medical" in lowered)
        and ("分割" in text or "segmentation" in lowered)
    )
    if not (wants_find and wants_import and is_mamba and is_medical_segmentation):
        return None

    target_year = None
    if "二六年" in text or re.search(r"(?<!\d)2026(?:年|\b)", lowered):
        target_year = 2026
    else:
        year_match = re.search(r"(?<!\d)(20\d{2})(?:年|\b)", lowered)
        if year_match:
            target_year = int(year_match.group(1))
    require_lightweight = any(
        word in lowered for word in ("轻量", "高效", "lightweight", "efficient", "compact")
    )
    query = "all:Mamba AND all:medical AND all:segmentation"
    if require_lightweight:
        query += " AND (all:lightweight OR all:efficient OR all:compact)"
    if target_year is not None:
        query += f" AND submittedDate:[{target_year}01010000 TO {target_year}12312359]"
    return {
        "query": query,
        "target_year": target_year,
        "require_lightweight": require_lightweight,
    }


def is_transient_model_error(exc: Exception) -> bool:
    """Retry only network/timeout/429/5xx failures, never validation/policy errors."""
    if isinstance(exc, (TimeoutError, ConnectionError, httpx.TimeoutException, httpx.NetworkError)):
        return True
    status_code = getattr(exc, "status_code", None)
    if status_code == 429 or (isinstance(status_code, int) and status_code >= 500):
        return True
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status == 429 or (
        isinstance(response_status, int) and response_status >= 500
    )


class ArxivImportRoutingMiddleware(AgentMiddleware):
    """Route explicit Mamba-paper discovery/import requests without LLM replanning.

    This is intentionally narrow. General arXiv conversations still use the
    Agent model, while this frequent compound action has a fixed, safe workflow:
    one search, duplicate-aware candidate selection, then human-approved import.
    """

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        messages = request.messages
        if not messages:
            return handler(request)
        last_message = messages[-1]
        if isinstance(last_message, HumanMessage):
            return self._route_user_request(last_message, handler, request)
        if isinstance(last_message, ToolMessage) and last_message.name == "find_arxiv_import_candidate":
            return self._route_candidate_result(last_message, handler, request)
        return handler(request)

    @staticmethod
    def _route_user_request(
        message: HumanMessage,
        handler: Callable[[ModelRequest], ModelResponse],
        request: ModelRequest,
    ) -> ModelResponse:
        content = message.content
        if not isinstance(content, str):
            return handler(request)
        plan = _arxiv_import_plan(content)
        if plan is None:
            return handler(request)
        tool_call_id = f"find-candidate-{uuid.uuid4().hex}"
        LOGGER.info(
            "routed explicit arxiv import intent query=%s target_year=%s",
            plan["query"],
            plan["target_year"],
        )
        return ModelResponse(
            result=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "find_arxiv_import_candidate",
                            "args": plan,
                            "id": tool_call_id,
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        )

    @staticmethod
    def _route_candidate_result(
        message: ToolMessage,
        handler: Callable[[ModelRequest], ModelResponse],
        request: ModelRequest,
    ) -> ModelResponse:
        payload = ArxivImportRoutingMiddleware._tool_payload(message)
        if payload is None:
            return handler(request)
        if payload.get("status") != "candidate_found":
            return ModelResponse(
                result=[AIMessage(content=str(payload.get("message") or "未创建导入任务。"))]
            )
        selected = payload.get("selected_paper")
        if not isinstance(selected, dict):
            return handler(request)
        arxiv_id = selected.get("arxiv_id")
        title = selected.get("title")
        if not isinstance(arxiv_id, str) or not arxiv_id:
            return handler(request)
        LOGGER.info("selected fresh arxiv candidate id=%s for human-approved import", arxiv_id)
        return ModelResponse(
            result=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "import_arxiv_paper",
                            "args": {
                                "arxiv_id_or_url": arxiv_id,
                                "expected_title": title if isinstance(title, str) else None,
                            },
                            "id": f"import-arxiv-{uuid.uuid4().hex}",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        )

    @staticmethod
    def _tool_payload(message: ToolMessage) -> dict[str, object] | None:
        if isinstance(message.content, dict):
            return message.content
        if not isinstance(message.content, str):
            return None
        try:
            payload = json.loads(message.content)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None


class ResearchTraceMiddleware(AgentMiddleware):
    """Record timings without serializing prompts, tool args, outputs, or secrets."""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        trace_id = str(uuid.uuid4())
        started = time.perf_counter()
        try:
            response = handler(request)
        except Exception:
            LOGGER.exception(
                "model_call trace_id=%s status=failed duration_ms=%d model=%s",
                trace_id,
                int((time.perf_counter() - started) * 1000),
                type(request.model).__name__,
            )
            raise
        LOGGER.info(
            "model_call trace_id=%s status=ok duration_ms=%d model=%s",
            trace_id,
            int((time.perf_counter() - started) * 1000),
            type(request.model).__name__,
        )
        return response

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        trace_id = str(uuid.uuid4())
        started = time.perf_counter()
        tool_name = str(request.tool_call.get("name", "unknown"))
        try:
            result = handler(request)
        except Exception:
            LOGGER.exception(
                "tool_call trace_id=%s tool=%s status=failed duration_ms=%d",
                trace_id,
                tool_name,
                int((time.perf_counter() - started) * 1000),
            )
            raise
        LOGGER.info(
            "tool_call trace_id=%s tool=%s status=ok duration_ms=%d",
            trace_id,
            tool_name,
            int((time.perf_counter() - started) * 1000),
        )
        return result


class PaperToolPolicyMiddleware(AgentMiddleware):
    def __init__(self, services: ServiceContainer):
        self.services = services

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        name = str(request.tool_call.get("name", ""))
        args = request.tool_call.get("args", {}) or {}
        if name in {"ask_papers", "compare_papers"}:
            paper_ids = list(dict.fromkeys(args.get("paper_ids") or []))
            limit = 8 if name == "ask_papers" else 5
            if not paper_ids or len(paper_ids) > limit:
                raise ResearchCopilotError(f"{name} 的论文数量必须在 1 到 {limit} 之间")
            self.services.rag.resolve_ready_papers(paper_ids)
        elif name == "summarize_paper":
            self.services.rag.resolve_ready_papers([str(args.get("paper_id", ""))])
        elif name == "import_local_paper":
            upload_id = str(args.get("upload_id", ""))
            if not upload_id or upload_id != Path(upload_id).name:
                raise ResearchCopilotError("非法 upload_id")
        return handler(request)
