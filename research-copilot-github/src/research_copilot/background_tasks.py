from __future__ import annotations

import logging
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any

from research_copilot.arxiv_service import ArxivService
from research_copilot.ingestion import PaperIngestionService
from research_copilot.models import DocumentRole
from research_copilot.rag import PaperRAGService
from research_copilot.storage import SQLiteRepository

LOGGER = logging.getLogger(__name__)

TASK_IMPORT_LOCAL = "import_local"
TASK_IMPORT_ARXIV = "import_arxiv"
TASK_SUMMARIZE = "summarize_paper"
TASK_COMPARE = "compare_papers"
TASK_TYPES = {TASK_IMPORT_LOCAL, TASK_IMPORT_ARXIV, TASK_SUMMARIZE, TASK_COMPARE}


class BackgroundTaskService:
    """Small persistent executor for the single-user Streamlit deployment.

    SQLite is the source of truth. A process restart changes interrupted tasks
    back to ``queued`` and resubmits them; completed results remain available
    after every Streamlit rerun.
    """

    def __init__(
        self,
        repository: SQLiteRepository,
        ingestion: PaperIngestionService,
        arxiv: ArxivService,
        rag: PaperRAGService,
        *,
        max_workers: int = 2,
    ):
        self.repository = repository
        self.ingestion = ingestion
        self.arxiv = arxiv
        self.rag = rag
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="research-copilot-task"
        )
        self._futures: dict[str, Future[None]] = {}
        self._lock = Lock()
        for task_id in self.repository.recover_background_tasks():
            self._schedule(task_id)

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)

    def submit(self, task_type: str, request: dict[str, Any]) -> str:
        if task_type not in TASK_TYPES:
            raise ValueError(f"不支持的后台任务类型：{task_type}")
        task_id = str(uuid.uuid4())
        self.repository.create_background_task(task_id, task_type, request)
        self._schedule(task_id)
        return task_id

    def submit_local_import(
        self,
        pdf_path: Path,
        *,
        title: str | None,
        document_role: DocumentRole,
        parent_paper_id: str | None,
        prefer_mineru: bool,
    ) -> str:
        return self.submit(
            TASK_IMPORT_LOCAL,
            {
                "pdf_path": str(pdf_path.resolve()),
                "title": title,
                "document_role": document_role.value,
                "parent_paper_id": parent_paper_id,
                "prefer_mineru": prefer_mineru,
            },
        )

    def submit_arxiv_import(self, arxiv_id: str, *, prefer_mineru: bool = False) -> str:
        return self.submit(
            TASK_IMPORT_ARXIV,
            {"arxiv_id": arxiv_id, "prefer_mineru": prefer_mineru},
        )

    def submit_summary(self, paper_id: str) -> str:
        return self.submit(TASK_SUMMARIZE, {"paper_id": paper_id})

    def submit_comparison(self, paper_ids: list[str]) -> str:
        return self.submit(TASK_COMPARE, {"paper_ids": sorted(set(paper_ids))})

    def cancel(self, task_id: str) -> bool:
        changed = self.repository.request_background_task_cancel(task_id)
        with self._lock:
            future = self._futures.get(task_id)
        if future is not None:
            future.cancel()
        return changed

    def _schedule(self, task_id: str) -> None:
        with self._lock:
            existing = self._futures.get(task_id)
            if existing is not None and not existing.done():
                return
            self._futures[task_id] = self.executor.submit(self._run, task_id)

    def _run(self, task_id: str) -> None:
        if not self.repository.claim_background_task(task_id):
            return
        task = self.repository.get_background_task(task_id)
        if task is None:
            return
        try:
            self.repository.update_background_task(
                task_id, progress=0.1, current_step="正在准备任务输入"
            )
            if self.repository.background_task_cancel_requested(task_id):
                self.repository.finish_background_task(task_id, status="cancelled")
                return
            result = self._execute(task_id, task["task_type"], task["request"])
            if self.repository.background_task_cancel_requested(task_id):
                self.repository.finish_background_task(task_id, status="cancelled")
            else:
                self.repository.finish_background_task(
                    task_id, status="completed", result=result
                )
        except Exception as exc:
            LOGGER.exception("Background task failed task_id=%s", task_id)
            self.repository.finish_background_task(
                task_id, status="failed", error=f"{type(exc).__name__}: {exc}"
            )

    def _execute(
        self, task_id: str, task_type: str, request: dict[str, Any]
    ) -> dict[str, Any]:
        if task_type == TASK_IMPORT_LOCAL:
            self.repository.update_background_task(
                task_id, progress=0.2, current_step="校验、解析并建立论文索引"
            )
            result = self.ingestion.ingest_local(
                Path(request["pdf_path"]),
                title=request.get("title"),
                document_role=DocumentRole(request.get("document_role", "main")),
                parent_paper_id=request.get("parent_paper_id"),
                prefer_mineru=bool(request.get("prefer_mineru", False)),
            )
            if result.error:
                raise RuntimeError(result.error)
            return result.model_dump(mode="json")
        if task_type == TASK_IMPORT_ARXIV:
            self.repository.update_background_task(
                task_id, progress=0.2, current_step="下载 arXiv PDF 并建立索引"
            )
            result = self.arxiv.import_paper(
                request["arxiv_id"],
                prefer_mineru=bool(request.get("prefer_mineru", False)),
            )
            if result.error:
                raise RuntimeError(result.error)
            return result.model_dump(mode="json")
        if task_type == TASK_SUMMARIZE:
            self.repository.update_background_task(
                task_id, progress=0.25, current_step="正在执行全文 Map-Reduce 摘要"
            )
            return self.rag.summarize(request["paper_id"]).model_dump(mode="json")
        if task_type == TASK_COMPARE:
            self.repository.update_background_task(
                task_id, progress=0.2, current_step="正在逐篇生成九维证据画像"
            )
            return self.rag.compare(request["paper_ids"]).model_dump(mode="json")
        raise ValueError(f"不支持的后台任务类型：{task_type}")
