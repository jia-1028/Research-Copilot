from __future__ import annotations

import hashlib
import logging
import re
import shutil
import uuid
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from research_copilot.config import Settings
from research_copilot.errors import ParserError, ResearchCopilotError
from research_copilot.models import (
    DocumentRole,
    IngestionResult,
    IngestionStatus,
    Paper,
    PaperChunk,
    PaperVersion,
    ParsedPage,
    SourceType,
)
from research_copilot.parsers import (
    MinerUParser,
    PyMuPDFParser,
    infer_pdf_title,
    validate_pdf,
)
from research_copilot.storage import SQLiteRepository
from research_copilot.vector_index import VectorIndex

LOGGER = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_slug(value: str) -> str:
    value = value.strip().lower().replace("_", "-")
    value = re.sub(r"[^a-z0-9\-]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value or "paper"


class PaperChunker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", "。", "; ", "；", " ", ""],
        )

    def split(
        self,
        pages: list[ParsedPage],
        *,
        paper_id: str,
        paper_version: int,
        paper_title: str,
        source_uri: str,
    ) -> list[PaperChunk]:
        chunks: list[PaperChunk] = []
        for page in pages:
            if not page.text.strip():
                continue
            parts = [part.strip() for part in self.splitter.split_text(page.text) if part.strip()]
            for chunk_index, text in enumerate(parts):
                text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                chunk_id = (
                    f"{paper_id}:v{paper_version}:p{page.page_number:04d}:c{chunk_index:03d}"
                )
                first_line = text.splitlines()[0].strip() if text.splitlines() else ""
                section_hint = first_line if 0 < len(first_line) <= 100 else None
                chunks.append(
                    PaperChunk(
                        chunk_id=chunk_id,
                        paper_id=paper_id,
                        paper_version=paper_version,
                        paper_title=paper_title,
                        page_number=page.page_number,
                        page_index=page.page_number - 1,
                        chunk_index_on_page=chunk_index,
                        text=text,
                        text_hash=text_hash,
                        section_hint=section_hint,
                        source_uri=source_uri,
                        embedding_model=self.settings.embedding_model,
                    )
                )
        return chunks


class PaperIngestionService:
    def __init__(
        self,
        settings: Settings,
        repository: SQLiteRepository,
        vector_index: VectorIndex,
    ):
        self.settings = settings
        self.repository = repository
        self.vector_index = vector_index
        self.chunker = PaperChunker(settings)

    def ingest_local(
        self,
        pdf_path: Path,
        *,
        title: str | None = None,
        paper_id: str | None = None,
        source_type: SourceType = SourceType.LOCAL,
        source_uri: str | None = None,
        document_role: DocumentRole = DocumentRole.MAIN,
        parent_paper_id: str | None = None,
        authors: list[str] | None = None,
        abstract: str | None = None,
        arxiv_id: str | None = None,
        prefer_mineru: bool | None = None,
    ) -> IngestionResult:
        pdf_path = pdf_path.resolve()
        job_id = str(uuid.uuid4())
        warnings: list[str] = []
        self.repository.update_job(
            job_id, IngestionStatus.PENDING, 0.0, "等待校验"
        )
        resolved_paper_id: str | None = None
        version = 0
        try:
            self.repository.update_job(
                job_id, IngestionStatus.VALIDATING, 0.05, "校验 PDF"
            )
            validate_pdf(pdf_path, self.settings.max_pdf_mb)
            digest = sha256_file(pdf_path)
            duplicate = self.repository.find_by_sha256(digest)
            if duplicate:
                if duplicate["status"] == IngestionStatus.FAILED.value:
                    self.repository.delete_failed_version(
                        duplicate["paper_id"], int(duplicate["version"])
                    )
                    warnings.append("检测到同一 SHA 的失败版本，已清理后重试")
                else:
                    return IngestionResult(
                        job_id=job_id,
                        paper_id=duplicate["paper_id"],
                        version=int(duplicate["version"]),
                        status=IngestionStatus(duplicate["status"]),
                        page_count=int(duplicate["page_count"]),
                        chunk_count=int(duplicate["chunk_count"]),
                        duplicate=True,
                        parser_name=duplicate["parser_name"],
                    )

            if title:
                title = title.strip()
            else:
                title = infer_pdf_title(pdf_path) or pdf_path.stem
                warnings.append(f"自动识别论文标题：{title}")
            base_id = paper_id or safe_slug(title)
            resolved_paper_id = base_id
            existing = self.repository.get_paper(resolved_paper_id)
            if existing and existing["title"] != title:
                resolved_paper_id = f"{base_id}-{digest[:8]}"
            version = self.repository.next_version(resolved_paper_id)
            paper = Paper(
                paper_id=resolved_paper_id,
                title=title,
                source_type=source_type,
                source_uri=source_uri or str(pdf_path),
                document_role=document_role,
                parent_paper_id=parent_paper_id,
                status=IngestionStatus.PARSING,
                authors=authors or [],
                abstract=abstract,
                arxiv_id=arxiv_id,
            )
            self.repository.upsert_paper(paper)

            paper_dir = self.settings.papers_dir / resolved_paper_id
            paper_dir.mkdir(parents=True, exist_ok=True)
            managed_path = paper_dir / f"{digest[:12]}.pdf"
            shutil.copy2(pdf_path, managed_path)
            parsed_dir = self.settings.parsed_dir / resolved_paper_id / f"v{version}"
            version_record = PaperVersion(
                paper_id=resolved_paper_id,
                version=version,
                sha256=digest,
                original_path=str(pdf_path),
                managed_copy_path=str(managed_path),
                parser_name="pending",
                parser_version="pending",
                parsed_dir=str(parsed_dir),
                status=IngestionStatus.PARSING,
            )
            self.repository.add_version(version_record)
            self.repository.update_job(
                job_id,
                IngestionStatus.PARSING,
                0.15,
                "解析论文版面",
                paper_id=resolved_paper_id,
                version=version,
            )

            use_mineru = self.settings.mineru_enabled if prefer_mineru is None else prefer_mineru
            if use_mineru:
                try:
                    parsed = MinerUParser(self.settings).parse(managed_path, parsed_dir)
                except ParserError as exc:
                    warnings.append(f"MinerU 失败，已降级 PyMuPDF：{exc}")
                    parsed = PyMuPDFParser(self.settings).parse(managed_path, parsed_dir)
            else:
                parsed = PyMuPDFParser(self.settings).parse(managed_path, parsed_dir)
            warnings.extend(parsed.warnings)

            self.repository.update_job(
                job_id,
                IngestionStatus.CHUNKING,
                0.45,
                "按页清洗并分块",
                paper_id=resolved_paper_id,
                version=version,
            )
            chunks = self.chunker.split(
                parsed.pages,
                paper_id=resolved_paper_id,
                paper_version=version,
                paper_title=title,
                source_uri=source_uri or str(pdf_path),
            )
            if not chunks:
                raise ResearchCopilotError("解析结果没有可索引文本")

            self.repository.update_job(
                job_id,
                IngestionStatus.EMBEDDING,
                0.6,
                f"生成 {len(chunks)} 个文本块的向量",
                paper_id=resolved_paper_id,
                version=version,
            )
            self.repository.update_job(
                job_id,
                IngestionStatus.INDEXING,
                0.8,
                "写入 Chroma",
                paper_id=resolved_paper_id,
                version=version,
            )
            self.vector_index.upsert_chunks(chunks)
            self.repository.finish_version(
                resolved_paper_id,
                version,
                parser_name=parsed.parser_name,
                parser_version=parsed.parser_version,
                page_count=len(parsed.pages),
                chunk_count=len(chunks),
                status=IngestionStatus.READY,
            )
            self.repository.update_job(
                job_id,
                IngestionStatus.READY,
                1.0,
                "导入完成",
                paper_id=resolved_paper_id,
                version=version,
            )
            return IngestionResult(
                job_id=job_id,
                paper_id=resolved_paper_id,
                version=version,
                status=IngestionStatus.READY,
                page_count=len(parsed.pages),
                chunk_count=len(chunks),
                parser_name=parsed.parser_name,
                warnings=warnings,
            )
        except Exception as exc:  # noqa: BLE001 - transaction boundary records failure
            if resolved_paper_id and version:
                try:
                    self.vector_index.delete_paper(resolved_paper_id, version)
                    self.repository.finish_version(
                        resolved_paper_id,
                        version,
                        parser_name="failed",
                        parser_version="failed",
                        page_count=0,
                        chunk_count=0,
                        status=IngestionStatus.FAILED,
                    )
                except Exception:
                    LOGGER.exception(
                        "Failed to clean partial ingestion paper_id=%s version=%s",
                        resolved_paper_id,
                        version,
                    )
            self.repository.update_job(
                job_id,
                IngestionStatus.FAILED,
                1.0,
                "导入失败",
                paper_id=resolved_paper_id,
                version=version or None,
                error=str(exc),
            )
            return IngestionResult(
                job_id=job_id,
                paper_id=resolved_paper_id or "unknown",
                version=version,
                status=IngestionStatus.FAILED,
                warnings=warnings,
                error=str(exc),
            )
