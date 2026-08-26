from __future__ import annotations

import json
import math
import time
from pathlib import Path
from statistics import mean
from typing import Any

import pymupdf as fitz
from langchain_core.embeddings import Embeddings

from research_copilot.config import Settings
from research_copilot.ingestion import PaperChunker
from research_copilot.models import ParsedPage
from research_copilot.parsers import clean_pages
from research_copilot.rag import PaperRAGService
from research_copilot.storage import SQLiteRepository


def load_eval_cases(dataset: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_eval_dataset(
    repository: SQLiteRepository, dataset: Path, *, minimum_cases: int = 30
) -> dict[str, Any]:
    cases = load_eval_cases(dataset)
    errors: list[str] = []
    refusal_count = 0
    for index, case in enumerate(cases, 1):
        if not str(case.get("question") or "").strip():
            errors.append(f"第 {index} 条缺少 question")
        paper_ids = case.get("paper_ids") or []
        if not paper_ids:
            errors.append(f"第 {index} 条缺少 paper_ids")
        for paper_id in paper_ids:
            paper = repository.get_paper(paper_id)
            if not paper or paper["status"] != "ready":
                errors.append(f"第 {index} 条引用不存在或未 ready 的论文：{paper_id}")
                continue
            for page in case.get("expected_pages") or []:
                if not isinstance(page, int) or page < 1 or page > int(paper["page_count"]):
                    errors.append(f"第 {index} 条页码越界：{paper_id} p.{page}")
        if case.get("should_refuse"):
            refusal_count += 1
    if len(cases) < minimum_cases:
        errors.append(f"评测集只有 {len(cases)} 条，至少需要 {minimum_cases} 条")
    if refusal_count < max(3, len(cases) // 10):
        errors.append("拒答样例不足，至少需要 3 条且不低于总数的 10%")
    return {
        "valid": not errors,
        "case_count": len(cases),
        "refusal_count": refusal_count,
        "errors": errors,
    }


def evaluate_jsonl(rag: PaperRAGService, dataset: Path, output: Path) -> dict:
    cases = load_eval_cases(dataset)
    results = []
    for case in cases:
        started = time.perf_counter()
        answer = rag.ask(case["question"], case["paper_ids"])
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        cited_pages = {citation.pdf_page for citation in answer.citations}
        expected_pages = set(case.get("expected_pages") or [])
        valid_scope = all(citation.paper_id in case["paper_ids"] for citation in answer.citations)
        citation_valid = all(
            citation.chunk_id.startswith(f"{citation.paper_id}:v{citation.paper_version}:")
            for citation in answer.citations
        )
        expected_refusal = bool(case.get("should_refuse", False))
        results.append(
            {
                "question": case["question"],
                "hit_at_k": bool(cited_pages & expected_pages) if expected_pages else None,
                "paper_scope_accurate": valid_scope,
                "citations_valid": citation_valid,
                "refusal_accurate": answer.insufficient_evidence == expected_refusal,
                "latency_ms": latency_ms,
                "trace_id": answer.retrieval_trace_id,
            }
        )

    def avg_bool(key: str) -> float | None:
        values = [float(item[key]) for item in results if item[key] is not None]
        return round(mean(values), 4) if values else None

    report = {
        "case_count": len(results),
        "metrics": {
            "hit_at_k": avg_bool("hit_at_k"),
            "paper_scope_accuracy": avg_bool("paper_scope_accurate"),
            "citation_validity": avg_bool("citations_valid"),
            "refusal_accuracy": avg_bool("refusal_accurate"),
            "mean_latency_ms": round(mean(item["latency_ms"] for item in results), 2)
            if results
            else None,
        },
        "cases": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def evaluate_retrieval_jsonl(
    rag: PaperRAGService, dataset: Path, output: Path, *, top_k: int = 8
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for case in load_eval_cases(dataset):
        started = time.perf_counter()
        hits = rag.retrieve(
            case["question"],
            case["paper_ids"],
            per_paper_k=top_k,
            context_per_paper=top_k,
        )
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        retrieved_pages = [hit.chunk.page_number for hit in hits]
        expected_pages = set(case.get("expected_pages") or [])
        first_relevant_rank = next(
            (rank for rank, page in enumerate(retrieved_pages, 1) if page in expected_pages),
            None,
        )
        results.append(
            {
                "question": case["question"],
                "paper_ids": case["paper_ids"],
                "expected_pages": sorted(expected_pages),
                "retrieved_pages": retrieved_pages,
                "hit_at_k": bool(expected_pages.intersection(retrieved_pages))
                if expected_pages
                else None,
                "reciprocal_rank": round(1 / first_relevant_rank, 4)
                if first_relevant_rank
                else (0.0 if expected_pages else None),
                "latency_ms": latency_ms,
            }
        )
    report = _retrieval_report(results, top_k=top_k)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def benchmark_chunking(
    settings: Settings,
    repository: SQLiteRepository,
    embedding_model: Embeddings,
    dataset: Path,
    output: Path,
    *,
    configurations: list[tuple[int, int]] | None = None,
    top_k: int = 8,
) -> dict[str, Any]:
    """Re-chunk managed PDFs and compare dense retrieval without mutating Chroma."""

    cases = load_eval_cases(dataset)
    configurations = configurations or [(800, 120), (1200, 180), (1500, 200)]
    paper_ids = sorted({paper_id for case in cases for paper_id in case["paper_ids"]})
    query_vectors = [embedding_model.embed_query(case["question"]) for case in cases]
    reports = []
    for chunk_size, chunk_overlap in configurations:
        variant_settings = settings.model_copy(
            update={"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
        )
        chunks = []
        for paper_id in paper_ids:
            paper = repository.get_paper(paper_id)
            if not paper or paper["status"] != "ready":
                raise ValueError(f"评测论文不存在或未 ready：{paper_id}")
            version = repository.get_version(paper_id, int(paper["active_version"]))
            if not version:
                raise ValueError(f"论文版本不存在：{paper_id}")
            with fitz.open(version["managed_copy_path"]) as document:
                pages = clean_pages(
                    [
                        ParsedPage(page_number=index + 1, text=page.get_text("text", sort=True))
                        for index, page in enumerate(document)
                    ]
                )
            chunks.extend(
                PaperChunker(variant_settings).split(
                    pages,
                    paper_id=paper_id,
                    paper_version=int(paper["active_version"]),
                    paper_title=paper["title"],
                    source_uri=paper["source_uri"],
                )
            )
        vectors: list[list[float]] = []
        for start in range(0, len(chunks), 20):
            vectors.extend(
                embedding_model.embed_documents(
                    [chunk.text for chunk in chunks[start : start + 20]]
                )
            )
        norms = [_norm(vector) for vector in vectors]
        results = []
        for case, query_vector in zip(cases, query_vectors, strict=True):
            started = time.perf_counter()
            query_norm = _norm(query_vector)
            allowed = set(case["paper_ids"])
            ranked = sorted(
                (
                    (_cosine(query_vector, query_norm, vector, norm), chunk)
                    for chunk, vector, norm in zip(chunks, vectors, norms, strict=True)
                    if chunk.paper_id in allowed
                ),
                key=lambda item: item[0],
                reverse=True,
            )[:top_k]
            retrieved_pages = [chunk.page_number for _, chunk in ranked]
            expected_pages = set(case.get("expected_pages") or [])
            first_rank = next(
                (rank for rank, page in enumerate(retrieved_pages, 1) if page in expected_pages),
                None,
            )
            results.append(
                {
                    "question": case["question"],
                    "expected_pages": sorted(expected_pages),
                    "retrieved_pages": retrieved_pages,
                    "hit_at_k": bool(expected_pages.intersection(retrieved_pages))
                    if expected_pages
                    else None,
                    "reciprocal_rank": round(1 / first_rank, 4)
                    if first_rank
                    else (0.0 if expected_pages else None),
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                }
            )
        variant = _retrieval_report(results, top_k=top_k)
        variant.update(
            {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "chunk_count": len(chunks),
            }
        )
        reports.append(variant)
    report = {"dataset": str(dataset), "top_k": top_k, "variants": reports}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _retrieval_report(results: list[dict[str, Any]], *, top_k: int) -> dict[str, Any]:
    hit_values = [float(item["hit_at_k"]) for item in results if item["hit_at_k"] is not None]
    reciprocal_ranks = [
        float(item["reciprocal_rank"])
        for item in results
        if item["reciprocal_rank"] is not None
    ]
    return {
        "case_count": len(results),
        "top_k": top_k,
        "metrics": {
            "hit_at_k": round(mean(hit_values), 4) if hit_values else None,
            "mrr": round(mean(reciprocal_ranks), 4) if reciprocal_ranks else None,
            "mean_latency_ms": round(mean(item["latency_ms"] for item in results), 2)
            if results
            else None,
        },
        "cases": results,
    }


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector)) or 1.0


def _cosine(
    query: list[float], query_norm: float, vector: list[float], vector_norm: float
) -> float:
    return sum(left * right for left, right in zip(query, vector, strict=True)) / (
        query_norm * vector_norm
    )
