from __future__ import annotations

from pathlib import Path

import pytest

from research_copilot.errors import InvalidPdfError
from research_copilot.ingestion import PaperChunker
from research_copilot.models import ParsedPage
from research_copilot.parsers import PyMuPDFParser, infer_pdf_title, validate_pdf


def test_validate_pdf_checks_magic_and_pages(sample_pdf: Path, tmp_path: Path) -> None:
    assert validate_pdf(sample_pdf) == 2
    fake = tmp_path / "fake.pdf"
    fake.write_text("not pdf", encoding="utf-8")
    with pytest.raises(InvalidPdfError, match="不是有效 PDF"):
        validate_pdf(fake)


def test_validate_pdf_rejects_non_pdf_extension(tmp_path: Path) -> None:
    file = tmp_path / "paper.txt"
    file.write_bytes(b"%PDF-1.7")
    with pytest.raises(InvalidPdfError, match="只支持 PDF"):
        validate_pdf(file)


def test_pymupdf_parser_preserves_page_numbers(sample_pdf: Path, tmp_path: Path) -> None:
    parsed = PyMuPDFParser().parse(sample_pdf, tmp_path / "parsed")
    assert [item.page_number for item in parsed.pages] == [1, 2]
    assert Path(parsed.markdown_path).exists()
    assert [item.page_number for item in parsed.page_images] == [1, 2]
    assert all(Path(item.image_path).is_file() for item in parsed.page_images)


def test_stable_chunk_id_is_page_scoped(settings) -> None:
    chunks = PaperChunker(settings).split(
        [ParsedPage(page_number=3, text="A sufficiently useful paragraph about a method.")],
        paper_id="paper-x",
        paper_version=2,
        paper_title="Paper X",
        source_uri="local.pdf",
    )
    assert chunks[0].chunk_id == "paper-x:v2:p0003:c000"
    assert chunks[0].page_number == 3


def test_title_inference_prefers_multiline_paper_title_over_header(tmp_path: Path) -> None:
    path = tmp_path / "opaque-upload-name.pdf"
    document = __import__("pymupdf").open()
    page = document.new_page()
    page.insert_text((72, 50), "Journal Header", fontsize=16)
    page.insert_text((72, 80), "Research Copilot: Evidence-Grounded Paper Question", fontsize=14)
    page.insert_text((72, 96), "Answering with Reliable PDF Citations", fontsize=14)
    page.insert_text((72, 140), "Anonymous submission", fontsize=12)
    document.save(path)
    document.close()
    assert infer_pdf_title(path) == (
        "Research Copilot: Evidence-Grounded Paper Question "
        "Answering with Reliable PDF Citations"
    )


def test_title_inference_uses_meaningful_metadata(tmp_path: Path) -> None:
    path = tmp_path / "random.pdf"
    document = __import__("pymupdf").open()
    document.set_metadata({"title": "Canonical Metadata Paper Title"})
    document.new_page().insert_text((72, 72), "First page")
    document.save(path)
    document.close()
    assert infer_pdf_title(path) == "Canonical Metadata Paper Title"
