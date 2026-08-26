from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pymupdf as fitz
from mineru import MinerU

from research_copilot.config import Settings
from research_copilot.errors import InvalidPdfError, ParserError
from research_copilot.models import ParsedPage, ParsedPageImage, ParsedPaper


@dataclass(frozen=True)
class _TitleLine:
    text: str
    font_size: float
    y0: float
    y1: float


_NON_TITLE_PATTERNS = (
    r"^abstract$",
    r"^anonymous submission$",
    r"manuscript draft",
    r"^manuscript number",
    r"^article type",
    r"^section/category",
    r"^keywords?:",
    r"^corresponding author",
    r"^first author",
    r"^order of authors",
)


class PaperParser(Protocol):
    def parse(self, pdf_path: Path, output_dir: Path) -> ParsedPaper: ...


def infer_pdf_title(pdf_path: Path) -> str | None:
    """Infer the canonical paper title from metadata or first-page typography."""
    try:
        with fitz.open(pdf_path) as document:
            metadata_title = _clean_title(document.metadata.get("title") or "")
            if _is_meaningful_title(metadata_title, pdf_path.stem):
                return metadata_title
            if document.page_count < 1:
                return None
            page = document[0]
            page_dict = page.get_text("dict")
            lines: list[_TitleLine] = []
            for block in page_dict.get("blocks", []):
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    text = _clean_title("".join(str(span.get("text", "")) for span in spans))
                    if not text or not spans:
                        continue
                    bbox = line.get("bbox", (0, 0, 0, 0))
                    lines.append(
                        _TitleLine(
                            text=text,
                            font_size=max(float(span.get("size", 0)) for span in spans),
                            y0=float(bbox[1]),
                            y1=float(bbox[3]),
                        )
                    )
            return _best_title_from_lines(lines, page.rect.height)
    except Exception:  # noqa: BLE001 - title inference must safely fall back to filename
        return None


def _best_title_from_lines(lines: list[_TitleLine], page_height: float) -> str | None:
    if not lines:
        return None
    top_lines = [line for line in lines if line.y0 <= page_height * 0.36]
    if not top_lines:
        return None
    max_font = max(line.font_size for line in top_lines)
    eligible = [line for line in top_lines if line.font_size >= max_font * 0.72]
    eligible.sort(key=lambda line: line.y0)
    groups: list[list[_TitleLine]] = []
    for line in eligible:
        if (
            groups
            and abs(groups[-1][-1].font_size - line.font_size) <= 0.6
            and line.y0 - groups[-1][-1].y1 <= max(8.0, line.font_size * 0.8)
        ):
            groups[-1].append(line)
        else:
            groups.append([line])

    candidates: list[tuple[float, str]] = []
    for group in groups:
        text = _clean_title(" ".join(line.text for line in group))
        lowered = text.lower()
        if not _is_meaningful_title(text, ""):
            continue
        if any(re.search(pattern, lowered) for pattern in _NON_TITLE_PATTERNS):
            continue
        average_font = sum(line.font_size for line in group) / len(group)
        score = average_font * 2.0 + min(len(text), 160) * 0.28
        if ":" in text or "：" in text:
            score += 8.0
        if len(text) < 24:
            score -= 10.0
        if len(group) > 1:
            score += 4.0
        candidates.append((score, text))
    return max(candidates, default=(0.0, None), key=lambda item: item[0])[1]


def _clean_title(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" \t\r\n-_")


def _is_meaningful_title(value: str, file_stem: str) -> bool:
    if len(value) < 8 or len(value) > 400:
        return False
    lowered = value.casefold()
    if lowered in {file_stem.casefold(), "untitled", "document", "paper", "manuscript"}:
        return False
    return not lowered.startswith(("microsoft word", "latex", "acrobat distiller"))


def validate_pdf(pdf_path: Path, max_pdf_mb: int = 200) -> int:
    if not pdf_path.exists() or not pdf_path.is_file():
        raise InvalidPdfError(f"PDF 不存在：{pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise InvalidPdfError("只支持 PDF 文件")
    if pdf_path.stat().st_size > max_pdf_mb * 1024 * 1024:
        raise InvalidPdfError(f"PDF 超过 {max_pdf_mb} MB 限制")
    with pdf_path.open("rb") as handle:
        if handle.read(5) != b"%PDF-":
            raise InvalidPdfError("文件扩展名为 PDF，但内容不是有效 PDF")
    try:
        with fitz.open(pdf_path) as document:
            if document.needs_pass:
                raise InvalidPdfError("暂不支持加密 PDF")
            if document.page_count < 1:
                raise InvalidPdfError("PDF 没有可读取页面")
            return document.page_count
    except InvalidPdfError:
        raise
    except Exception as exc:
        raise InvalidPdfError(f"PDF 无法打开：{exc}") from exc


def clean_pages(pages: list[ParsedPage]) -> list[ParsedPage]:
    """Remove obvious repeated headers/footers while preserving page boundaries."""
    if not pages:
        return pages
    edge_counts: Counter[str] = Counter()
    split_pages: list[list[str]] = []
    for page in pages:
        lines = [re.sub(r"\s+", " ", line).strip() for line in page.text.splitlines()]
        lines = [line for line in lines if line]
        split_pages.append(lines)
        for line in lines[:2] + lines[-2:]:
            if len(line) <= 120:
                edge_counts[line] += 1
    threshold = max(3, int(len(pages) * 0.6))
    repeated = {line for line, count in edge_counts.items() if count >= threshold}
    cleaned: list[ParsedPage] = []
    for page, lines in zip(pages, split_pages, strict=True):
        kept = [line for line in lines if line not in repeated and not re.fullmatch(r"\d+", line)]
        cleaned.append(ParsedPage(page_number=page.page_number, text="\n".join(kept).strip()))
    return cleaned


def render_pdf_page_images(
    pdf_path: Path,
    output_dir: Path,
    *,
    dpi: int = 120,
    jpeg_quality: int = 85,
) -> list[ParsedPageImage]:
    """Render complete PDF pages so vector figures and captions stay together."""
    image_dir = (output_dir / "page_images").resolve()
    output_root = output_dir.resolve()
    if image_dir.parent != output_root:
        raise ParserError("非法页面图像输出目录")
    image_dir.mkdir(parents=True, exist_ok=True)
    for stale in image_dir.glob("page_*.jpg"):
        stale.unlink()
    rendered: list[ParsedPageImage] = []
    scale = dpi / 72.0
    try:
        with fitz.open(pdf_path) as document:
            for index, page in enumerate(document, start=1):
                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(scale, scale),
                    colorspace=fitz.csRGB,
                    alpha=False,
                )
                image_path = image_dir / f"page_{index:04d}.jpg"
                pixmap.save(image_path, jpg_quality=jpeg_quality)
                rendered.append(
                    ParsedPageImage(
                        page_number=index,
                        image_path=str(image_path),
                        width=pixmap.width,
                        height=pixmap.height,
                        sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
                    )
                )
    except Exception as exc:
        raise ParserError(f"PDF 页面图像渲染失败：{exc}") from exc
    return rendered


class PyMuPDFParser:
    name = "pymupdf"
    version = fitz.VersionBind

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        render_page_images: bool | None = None,
    ):
        self.settings = settings
        self.render_page_images = (
            settings.multimodal_enabled
            if render_page_images is None and settings is not None
            else (True if render_page_images is None else render_page_images)
        )

    def parse(self, pdf_path: Path, output_dir: Path) -> ParsedPaper:
        output_dir.mkdir(parents=True, exist_ok=True)
        pages: list[ParsedPage] = []
        try:
            with fitz.open(pdf_path) as document:
                for index, page in enumerate(document):
                    pages.append(
                        ParsedPage(page_number=index + 1, text=page.get_text("text", sort=True))
                    )
        except Exception as exc:
            raise ParserError(f"PyMuPDF 解析失败：{exc}") from exc
        pages = clean_pages(pages)
        page_images = (
            render_pdf_page_images(
                pdf_path,
                output_dir,
                dpi=self.settings.page_image_dpi if self.settings else 120,
                jpeg_quality=(
                    self.settings.page_image_jpeg_quality if self.settings else 85
                ),
            )
            if self.render_page_images
            else []
        )
        markdown_path = output_dir / "document.md"
        markdown = "\n\n".join(
            f"<!-- PDF_PAGE:{page.page_number} -->\n\n{page.text}" for page in pages
        )
        markdown_path.write_text(markdown, encoding="utf-8")
        content_path = output_dir / "content_list.json"
        content_path.write_text(
            json.dumps([page.model_dump() for page in pages], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        warnings = []
        empty_ratio = sum(not page.text.strip() for page in pages) / max(len(pages), 1)
        if empty_ratio > 0.2:
            warnings.append("超过 20% 页面没有可提取文本，PDF 可能包含扫描页")
        manifest = {
            "parser": self.name,
            "parser_version": self.version,
            "page_count": len(pages),
            "page_images": [item.model_dump() for item in page_images],
            "warnings": warnings,
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return ParsedPaper(
            parser_name=self.name,
            parser_version=self.version,
            pages=pages,
            markdown_path=str(markdown_path),
            content_list_path=str(content_path),
            page_images=page_images,
            warnings=warnings,
            manifest=manifest,
        )


class MinerUParser:
    name = "mineru-online"
    version = "v4-sdk-0.2.5"

    def __init__(self, settings: Settings):
        if not settings.mineru_api_token:
            raise ParserError("MINERU_API_TOKEN 未配置")
        self.settings = settings

    def parse(self, pdf_path: Path, output_dir: Path) -> ParsedPaper:
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            with MinerU(self.settings.mineru_api_token.get_secret_value()) as client:
                result = client.extract(
                    str(pdf_path),
                    model=self.settings.mineru_model,
                    ocr=False,
                    formula=True,
                    table=True,
                    language=self.settings.mineru_language,
                    timeout=900,
                )
                if result.state != "done" or not result.markdown:
                    raise ParserError(result.error or f"MinerU 返回状态：{result.state}")
                result.save_all(str(output_dir))
        except ParserError:
            raise
        except Exception as exc:
            raise ParserError(f"MinerU 在线解析失败：{exc}") from exc

        markdown_path = output_dir / "document.md"
        if not markdown_path.exists():
            result.save_markdown(str(markdown_path), with_images=True)
        content_path = output_dir / "content_list.json"
        content_list = result.content_list or []
        if not content_path.exists():
            content_path.write_text(
                json.dumps(content_list, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        pages = self._pages_from_content_list(content_list)
        warnings: list[str] = []
        if not pages:
            warnings.append("MinerU 结果缺少可靠页码映射；使用 PyMuPDF 恢复逐页文本")
            pages = PyMuPDFParser(render_page_images=False).parse(
                pdf_path, output_dir / "page_map_fallback"
            ).pages
        pages = clean_pages(pages)
        page_images = (
            render_pdf_page_images(
                pdf_path,
                output_dir,
                dpi=self.settings.page_image_dpi,
                jpeg_quality=self.settings.page_image_jpeg_quality,
            )
            if self.settings.multimodal_enabled
            else []
        )
        manifest = {
            "parser": self.name,
            "parser_version": self.version,
            "task_id": result.task_id,
            "model": self.settings.mineru_model,
            "language": self.settings.mineru_language,
            "formula": True,
            "table": True,
            "page_count": len(pages),
            "page_images": [item.model_dump() for item in page_images],
            "warnings": warnings,
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return ParsedPaper(
            parser_name=self.name,
            parser_version=self.version,
            pages=pages,
            markdown_path=str(markdown_path),
            content_list_path=str(content_path),
            page_images=page_images,
            warnings=warnings,
            manifest=manifest,
        )

    @staticmethod
    def _pages_from_content_list(content_list: list[dict]) -> list[ParsedPage]:
        grouped: dict[int, list[str]] = defaultdict(list)
        for item in content_list:
            page_index = item.get("page_idx", item.get("page_index"))
            if page_index is None:
                continue
            text = item.get("text") or item.get("content") or item.get("table_body") or ""
            if isinstance(text, list):
                text = "\n".join(str(part) for part in text)
            if str(text).strip():
                grouped[int(page_index) + 1].append(str(text).strip())
        return [
            ParsedPage(page_number=page_number, text="\n\n".join(parts))
            for page_number, parts in sorted(grouped.items())
        ]
