from __future__ import annotations

import shutil
import subprocess
import sys
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings

from research_copilot.config import Settings
from research_copilot.errors import ResearchCopilotError
from research_copilot.ingestion import PaperChunker
from research_copilot.model_factory import create_embedding_model
from research_copilot.parsers import PyMuPDFParser
from research_copilot.storage import SQLiteRepository
from research_copilot.vector_index import ChromaVectorIndex


def _validate_reopened_index(
    persist_directory: Path, collection_name: str, expected_total: int
) -> None:
    """Verify a fresh Python process can open and query the replacement index.

    Chroma can keep an in-memory HNSW reader alive in the process that built an
    index.  A same-process count is therefore insufficient proof that the files
    on disk are usable after Streamlit restarts.
    """

    script = """
import sys
import chromadb

persist_directory, collection_name, expected_total = sys.argv[1:]
client = chromadb.PersistentClient(path=persist_directory)
try:
    collection = client.get_collection(collection_name)
    actual_total = collection.count()
    if actual_total != int(expected_total):
        raise RuntimeError(f"count mismatch: expected={expected_total}, actual={actual_total}")
    sample = collection.get(limit=1, include=["embeddings"])
    embeddings = sample.get("embeddings")
    if embeddings is None or len(embeddings) == 0:
        raise RuntimeError("collection contains no stored embedding")
    result = collection.query(
        query_embeddings=[[0.0] * len(embeddings[0])], n_results=1
    )
    if not result.get("ids") or not result["ids"][0]:
        raise RuntimeError("query returned no candidates")
    print(f"ok count={actual_total}")
finally:
    client.close()
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(persist_directory),
            collection_name,
            str(expected_total),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout or "unknown error").strip()
        raise ResearchCopilotError(
            "staging 重启校验失败，未替换生产索引：" + details[-1200:]
        )


def repair_chroma_index(
    settings: Settings,
    *,
    embedding_model: Embeddings | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Build and validate a replacement collection before swapping directories."""

    def progress(message: str) -> None:
        if progress_callback:
            progress_callback(message)

    repository = SQLiteRepository(
        settings.sqlite_path, checkpoint_path=settings.checkpoint_path
    )
    papers = repository.list_papers(status="ready")
    if not papers:
        raise ResearchCopilotError("没有 ready 论文，无法重建向量索引")

    run_id = uuid.uuid4().hex[:12]
    # The vector directory may be intentionally separated from the rest of
    # application data (for example, to use an ASCII-only Windows path).
    staging_dir = settings.chroma_dir.parent / f".chroma-rebuild-{run_id}"
    parse_dir = settings.project_data_dir / f".chroma-repair-parse-{run_id}"
    production_dir = settings.chroma_dir.resolve()
    backup_root = (settings.project_data_dir / "backups").resolve()
    backup_root.mkdir(parents=True, exist_ok=True)
    backup_dir = backup_root / (
        "chroma-corrupt-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    for path in (staging_dir, parse_dir):
        if path.exists():
            raise ResearchCopilotError(f"修复临时目录已存在：{path}")

    plans: list[tuple[dict[str, Any], dict[str, Any], Path]] = []
    for paper in papers:
        version_number = int(paper["active_version"])
        version = repository.get_version(paper["paper_id"], version_number)
        if version is None:
            raise ResearchCopilotError(
                f"缺少 active version 记录：{paper['paper_id']} v{version_number}"
            )
        source = Path(version["managed_copy_path"]).resolve()
        if not source.is_file():
            raise ResearchCopilotError(f"受控 PDF 不存在：{source}")
        plans.append((paper, version, source))
    progress(f"预检完成：{len(plans)} 份 ready 文档均有受控 PDF。")

    embedding_model = embedding_model or create_embedding_model(settings)
    index: ChromaVectorIndex | None = None
    expected_total = 0
    per_paper: dict[str, int] = {}
    versions: dict[str, int] = {}
    swapped = False
    try:
        index = ChromaVectorIndex(
            settings, embedding_model, persist_directory=staging_dir
        )
        parser = PyMuPDFParser(settings)
        chunker = PaperChunker(settings)
        for position, (paper, _version, source) in enumerate(plans, 1):
            paper_id = paper["paper_id"]
            version_number = int(paper["active_version"])
            progress(f"[{position}/{len(plans)}] 解析并重建：{paper['title']}")
            parsed = parser.parse(
                source, parse_dir / paper_id / f"v{version_number}"
            )
            chunks = chunker.split(
                parsed.pages,
                paper_id=paper_id,
                paper_version=version_number,
                paper_title=paper["title"],
                source_uri=paper["source_uri"],
            )
            if not chunks:
                raise ResearchCopilotError(f"论文未生成任何 chunk：{paper_id}")
            index.upsert_chunks(chunks)
            per_paper[paper_id] = len(chunks)
            versions[paper_id] = version_number
            expected_total += len(chunks)
            progress(f"已写入 {len(chunks)} 个 chunks：{paper_id} v{version_number}")

        actual_total = index.count()
        if actual_total != expected_total:
            raise ResearchCopilotError(
                f"staging 数量校验失败：expected={expected_total}, actual={actual_total}"
            )
        for paper_id, version_number in versions.items():
            actual = len(index.get_paper_chunks(paper_id, version_number))
            if actual != per_paper[paper_id]:
                raise ResearchCopilotError(
                    f"论文过滤校验失败：{paper_id} expected={per_paper[paper_id]}, actual={actual}"
                )
        hits = index.similarity_search(
            "method architecture experiments results", versions, 1
        )
        hit_papers = {item.chunk.paper_id for item in hits}
        missing = sorted(set(versions) - hit_papers)
        if missing:
            raise ResearchCopilotError("HNSW 查询校验缺少论文：" + "、".join(missing))
        progress(f"staging 校验通过：{actual_total} chunks，{len(hit_papers)} 篇过滤查询正常。")

        index.close()
        index = None
        _validate_reopened_index(
            staging_dir, settings.collection_name, actual_total
        )
        progress("staging 重启校验通过：新 Python 进程可重新打开并查询 HNSW 索引。")
        if backup_dir.exists():
            raise ResearchCopilotError(f"备份目标已存在：{backup_dir}")
        if production_dir.exists():
            shutil.move(str(production_dir), str(backup_dir))
        try:
            shutil.move(str(staging_dir), str(production_dir))
        except Exception:
            if backup_dir.exists() and not production_dir.exists():
                shutil.move(str(backup_dir), str(production_dir))
            raise
        swapped = True
        progress(f"已切换生产索引；损坏索引保存在：{backup_dir}")
        return {
            "status": "repaired",
            "collection": settings.collection_name,
            "paper_count": len(plans),
            "chunk_count": actual_total,
            "per_paper_chunks": per_paper,
            "backup_dir": str(backup_dir),
            "production_dir": str(production_dir),
        }
    finally:
        if index is not None:
            index.close()
        if parse_dir.exists():
            shutil.rmtree(parse_dir)
        if staging_dir.exists() and not swapped:
            shutil.rmtree(staging_dir)
