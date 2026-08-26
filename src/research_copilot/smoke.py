from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel

from research_copilot.config import Settings
from research_copilot.model_factory import (
    create_chat_model,
    create_embedding_model,
    create_fallback_chat_model,
)
from research_copilot.parsers import MinerUParser


class SmokeStructuredOutput(BaseModel):
    status: str
    number: int


def run_online_smoke_tests(
    settings: Settings, *, mineru_pdf: Path | None = None
) -> dict[str, object]:
    report: dict[str, object] = {}
    model = create_chat_model(settings)
    response = model.invoke("只回复 OK")
    report["chat"] = {"ok": bool(response.content), "response_type": type(response).__name__}

    structured_model = create_fallback_chat_model(settings) or model
    structured = structured_model.with_structured_output(SmokeStructuredOutput).invoke(
        "返回 status='ok' 和 number=7"
    )
    report["structured_output"] = {
        "ok": structured.status.lower() == "ok" and structured.number == 7,
        "model": settings.fallback_chat_model or settings.chat_model,
    }

    @tool
    def multiply(a: int, b: int) -> int:
        """Multiply two integers."""
        return a * b

    tool_response = model.bind_tools([multiply], tool_choice="multiply").invoke(
        "请调用工具计算 6×7"
    )
    report["tool_calling"] = {
        "ok": bool(tool_response.tool_calls),
        "tool": tool_response.tool_calls[0]["name"] if tool_response.tool_calls else None,
    }

    embeddings = create_embedding_model(settings)
    vector = embeddings.embed_query("Research Copilot embedding smoke test")
    report["embedding"] = {"ok": len(vector) > 0, "dimension": len(vector)}

    report["mineru_token"] = {
        "ok": settings.mineru_api_token is not None,
        "network_tested": mineru_pdf is not None,
    }
    if mineru_pdf is not None:
        parsed = MinerUParser(settings).parse(
            mineru_pdf, settings.project_data_dir / "reports" / "mineru-smoke"
        )
        report["mineru"] = {
            "ok": bool(parsed.pages),
            "page_count": len(parsed.pages),
            "parser_version": parsed.parser_version,
        }
    return report
