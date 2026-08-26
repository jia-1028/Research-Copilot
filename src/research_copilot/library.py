from __future__ import annotations

import json
import shutil
from pathlib import Path

from research_copilot.errors import ResearchCopilotError
from research_copilot.ingestion import PaperChunker
from research_copilot.models import IngestionStatus
from research_copilot.parsers import (
    MinerUParser,
    PyMuPDFParser,
    infer_pdf_title,
    render_pdf_page_images,
)
from research_copilot.services_types import CoreServices


class PaperLibraryService:
    """Explicit destructive/rebuild operations; deliberately not exposed as Agent tools."""

    def __init__(self, core: CoreServices):
        self.core = core

    def delete_paper(self, paper_id: str) -> None:
        paper = self.core.repository.get_paper(paper_id)
        if paper is None:
            raise ResearchCopilotError(f"论文不存在：{paper_id}")
        children = self.core.repository.list_paper_children(paper_id)
        if children:
            child_ids = "、".join(item["paper_id"] for item in children)
            raise ResearchCopilotError(
                f"主论文仍有关联 supplementary：{child_ids}。请先明确删除补充材料。"
            )
        self.core.vector_index.delete_paper(paper_id)
        self.core.repository.mark_paper_conversations_read_only(
            paper_id, paper["title"]
        )
        self.core.repository.delete_paper_metadata(paper_id)
        for root in (self.core.settings.papers_dir, self.core.settings.parsed_dir):
            target = (root / paper_id).resolve()
            if target.parent == root.resolve() and target.exists():
                shutil.rmtree(target)

    def deletion_impact(self, paper_id: str) -> dict:
        paper = self.core.repository.get_paper(paper_id)
        if paper is None:
            raise ResearchCopilotError(f"论文不存在：{paper_id}")
        children = self.core.repository.list_paper_children(paper_id)
        conversations = self.core.repository.affected_conversations_for_paper(paper_id)
        return {
            "paper_id": paper_id,
            "title": paper["title"],
            "supplementary_ids": [item["paper_id"] for item in children],
            "affected_conversation_count": len(conversations),
            "affected_conversation_titles": [item["title"] for item in conversations[:10]],
        }

    def rebuild_active_version(self, paper_id: str, *, prefer_mineru: bool = False) -> int:
        paper = self.core.repository.get_paper(paper_id)
        if paper is None or paper["status"] != "ready":
            raise ResearchCopilotError(f"只能重建 ready 论文：{paper_id}")
        version_number = int(paper["active_version"])
        version = self.core.repository.get_version(paper_id, version_number)
        if version is None:
            raise ResearchCopilotError("找不到当前论文版本记录")
        source = Path(version["managed_copy_path"])
        parsed_dir = Path(version["parsed_dir"])
        parser = (
            MinerUParser(self.core.settings)
            if prefer_mineru
            else PyMuPDFParser(self.core.settings)
        )
        parsed = parser.parse(source, parsed_dir)
        chunks = PaperChunker(self.core.settings).split(
            parsed.pages,
            paper_id=paper_id,
            paper_version=version_number,
            paper_title=paper["title"],
            source_uri=paper["source_uri"],
        )
        self.core.vector_index.delete_paper(paper_id, version_number)
        try:
            self.core.vector_index.upsert_chunks(chunks)
        except Exception:
            self.core.repository.finish_version(
                paper_id,
                version_number,
                parser_name=parsed.parser_name,
                parser_version=parsed.parser_version,
                page_count=len(parsed.pages),
                chunk_count=0,
                status=IngestionStatus.FAILED,
            )
            raise
        self.core.repository.finish_version(
            paper_id,
            version_number,
            parser_name=parsed.parser_name,
            parser_version=parsed.parser_version,
            page_count=len(parsed.pages),
            chunk_count=len(chunks),
            status=IngestionStatus.READY,
        )
        self.core.repository.invalidate_profile_cache(paper_id, version_number)
        return len(chunks)

    def page_image_status(self, paper_id: str) -> dict:
        paper = self.core.repository.get_paper(paper_id)
        if paper is None or int(paper["active_version"]) < 1:
            return {"count": 0, "page_count": 0, "complete": False}
        version = self.core.repository.get_version(
            paper_id, int(paper["active_version"])
        )
        if version is None:
            return {"count": 0, "page_count": 0, "complete": False}
        image_dir = Path(version["parsed_dir"]) / "page_images"
        count = len(list(image_dir.glob("page_*.jpg"))) if image_dir.exists() else 0
        page_count = int(version["page_count"] or 0)
        return {
            "count": count,
            "page_count": page_count,
            "complete": page_count > 0 and count == page_count,
        }

    def render_page_images(self, paper_id: str) -> int:
        paper = self.core.repository.get_paper(paper_id)
        if paper is None or paper["status"] != "ready":
            raise ResearchCopilotError(f"只能为 ready 论文生成页面图像：{paper_id}")
        version_number = int(paper["active_version"])
        version = self.core.repository.get_version(paper_id, version_number)
        if version is None:
            raise ResearchCopilotError("找不到当前论文版本记录")
        parsed_dir = Path(version["parsed_dir"])
        images = render_pdf_page_images(
            Path(version["managed_copy_path"]),
            parsed_dir,
            dpi=self.core.settings.page_image_dpi,
            jpeg_quality=self.core.settings.page_image_jpeg_quality,
        )
        manifest_path = parsed_dir / "manifest.json"
        manifest = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["page_images"] = [item.model_dump() for item in images]
        manifest["page_image_dpi"] = self.core.settings.page_image_dpi
        manifest["page_image_jpeg_quality"] = (
            self.core.settings.page_image_jpeg_quality
        )
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return len(images)

    def refresh_title_from_pdf(self, paper_id: str) -> str:
        paper = self.core.repository.get_paper(paper_id)
        if paper is None or int(paper["active_version"]) < 1:
            raise ResearchCopilotError(f"论文没有可用版本：{paper_id}")
        version_number = int(paper["active_version"])
        version = self.core.repository.get_version(paper_id, version_number)
        if version is None:
            raise ResearchCopilotError("找不到当前论文版本记录")
        title = infer_pdf_title(Path(version["managed_copy_path"]))
        if not title:
            raise ResearchCopilotError("无法从 PDF metadata 或首页版式识别标题")
        self.core.repository.update_paper_title(paper_id, title)
        self.core.vector_index.update_paper_title(paper_id, version_number, title)
        return title
