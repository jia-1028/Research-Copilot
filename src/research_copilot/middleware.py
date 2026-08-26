from __future__ import annotations

import logging
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
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from research_copilot.errors import ResearchCopilotError
from research_copilot.services import ServiceContainer

LOGGER = logging.getLogger("research_copilot.trace")


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
