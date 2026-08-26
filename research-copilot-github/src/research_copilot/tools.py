from __future__ import annotations

from pathlib import Path
from typing import Literal

from langchain_core.tools import BaseTool, tool
from langgraph.config import get_stream_writer
from pydantic import BaseModel, Field

from research_copilot.errors import ResearchCopilotError
from research_copilot.models import DocumentRole
from research_copilot.services import ServiceContainer


class ListPapersInput(BaseModel):
    keyword: str | None = Field(default=None, description="标题或 paper_id 关键词")
    include_non_ready: bool = Field(default=False, description="是否包含导入中/失败论文")


class ImportLocalPaperInput(BaseModel):
    upload_id: str = Field(description="Streamlit 已保存到受控 uploads 目录的文件名")
    title: str | None = None
    document_role: Literal["main", "supplementary"] = "main"
    parent_paper_id: str | None = None


class SearchArxivInput(BaseModel):
    query: str = Field(min_length=2)
    max_results: int = Field(default=5, ge=1, le=20)


class ImportArxivInput(BaseModel):
    arxiv_id_or_url: str


class IngestionStatusInput(BaseModel):
    job_id: str | None = None
    paper_id: str | None = None


class AskPapersInput(BaseModel):
    question: str = Field(min_length=1)
    paper_ids: list[str] = Field(min_length=1, max_length=8)
    thread_id: str | None = None


class SummarizePaperInput(BaseModel):
    paper_id: str


class ComparePapersInput(BaseModel):
    paper_ids: list[str] = Field(min_length=2, max_length=5)


def _resolve_upload(uploads_dir: Path, upload_id: str) -> Path:
    if Path(upload_id).name != upload_id:
        raise ResearchCopilotError("upload_id 只能是受控上传目录中的文件名")
    root = uploads_dir.resolve()
    candidate = (root / upload_id).resolve()
    if candidate.parent != root or not candidate.is_file():
        raise ResearchCopilotError(f"上传文件不存在：{upload_id}")
    return candidate


def build_tools(services: ServiceContainer) -> list[BaseTool]:
    @tool(args_schema=ListPapersInput)
    def list_papers(keyword: str | None = None, include_non_ready: bool = False) -> list[dict]:
        """列出论文库中的论文、paper_id、状态、页数与当前版本。"""
        return services.repository.list_papers(
            keyword=keyword, status=None if include_non_ready else "ready"
        )

    @tool(args_schema=ImportLocalPaperInput)
    def import_local_paper(
        upload_id: str,
        title: str | None = None,
        document_role: str = "main",
        parent_paper_id: str | None = None,
    ) -> dict:
        """导入用户已经上传的本地 PDF；该操作需要人工确认。"""
        if document_role == "supplementary" and not parent_paper_id:
            raise ResearchCopilotError("补充材料必须提供 parent_paper_id")
        task_id = services.tasks.submit_local_import(
            _resolve_upload(services.settings.uploads_dir, upload_id),
            title=title,
            document_role=DocumentRole(document_role),
            parent_paper_id=parent_paper_id,
            prefer_mineru=services.settings.mineru_enabled,
        )
        return {"task_id": task_id, "status": "queued", "message": "论文已加入后台导入队列"}

    @tool(args_schema=SearchArxivInput)
    def search_arxiv(query: str, max_results: int = 5) -> list[dict]:
        """按主题检索 arXiv 元数据，不会下载或导入论文。"""
        return [
            item.model_dump(mode="json")
            for item in services.arxiv.search(query, max_results=max_results)
        ]

    @tool(args_schema=ImportArxivInput)
    def import_arxiv_paper(arxiv_id_or_url: str) -> dict:
        """下载并导入指定 arXiv 论文；该操作需要人工确认。"""
        task_id = services.tasks.submit_arxiv_import(
            arxiv_id_or_url, prefer_mineru=services.settings.mineru_enabled
        )
        return {
            "task_id": task_id,
            "status": "queued",
            "message": "arXiv 论文已加入后台下载与导入队列",
        }

    @tool(args_schema=IngestionStatusInput)
    def get_ingestion_status(
        job_id: str | None = None, paper_id: str | None = None
    ) -> dict:
        """根据 job_id 或 paper_id 查询导入状态。"""
        if bool(job_id) == bool(paper_id):
            raise ResearchCopilotError("job_id 和 paper_id 必须且只能提供一个")
        background = services.repository.get_background_task(job_id) if job_id else None
        result = background or (
            services.repository.get_job(job_id)
            if job_id
            else services.repository.get_job_for_paper(paper_id or "")
        )
        if result is None:
            raise ResearchCopilotError("未找到导入任务")
        if background:
            return {
                key: result[key]
                for key in (
                    "task_id",
                    "task_type",
                    "status",
                    "progress",
                    "current_step",
                    "error",
                    "result",
                )
            }
        return result

    @tool(
        args_schema=AskPapersInput,
        response_format="content_and_artifact",
        return_direct=True,
    )
    def ask_papers(
        question: str,
        paper_ids: list[str],
        thread_id: str | None = None,
    ) -> tuple[str, dict]:
        """一次性回答本轮全部论文细节问题并返回 PDF 证据；同一轮不得拆成多次调用。"""
        stream_writer = get_stream_writer()

        def report_progress(message: str) -> None:
            stream_writer(
                {"type": "tool_progress", "tool": "ask_papers", "message": message}
            )

        answer = services.rag.ask(
            question,
            paper_ids,
            thread_id=thread_id,
            progress_callback=report_progress,
        )
        payload = answer.model_dump(mode="json")
        # The complete evidence snapshot remains available to the UI as an
        # artifact, but is not sent back through another expensive model call.
        return answer.answer, payload

    @tool(args_schema=SummarizePaperInput)
    def summarize_paper(paper_id: str) -> dict:
        """仅在用户明确要求整篇论文完整摘要时，使用全文 Map-Reduce 生成论文画像。"""
        return services.rag.summarize(paper_id).model_dump(mode="json")

    @tool(args_schema=ComparePapersInput)
    def compare_papers(paper_ids: list[str]) -> dict:
        """逐篇生成论文画像，再按统一维度进行可追溯比较。"""
        return services.rag.compare(paper_ids).model_dump(mode="json")

    return [
        list_papers,
        import_local_paper,
        search_arxiv,
        import_arxiv_paper,
        get_ingestion_status,
        ask_papers,
        summarize_paper,
        compare_papers,
    ]
