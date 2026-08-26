from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import chromadb
from langchain_core.embeddings import Embeddings

from research_copilot.config import Settings
from research_copilot.errors import ResearchCopilotError
from research_copilot.models import PaperChunk, RetrievedChunk


class VectorIndex(Protocol):
    def upsert_chunks(self, chunks: Sequence[PaperChunk]) -> None: ...

    def similarity_search(
        self, query: str, paper_versions: dict[str, int], top_k: int
    ) -> list[RetrievedChunk]: ...

    def delete_paper(self, paper_id: str, version: int | None = None) -> None: ...

    def get_chunks(self, chunk_ids: Sequence[str]) -> list[PaperChunk]: ...

    def get_paper_chunks(self, paper_id: str, version: int) -> list[PaperChunk]: ...

    def update_paper_title(self, paper_id: str, version: int, title: str) -> None: ...

    def count(self) -> int: ...


class ChromaVectorIndex:
    def __init__(
        self,
        settings: Settings,
        embedding_model: Embeddings,
        *,
        persist_directory: Path | None = None,
    ):
        self.settings = settings
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory or settings.chroma_dir
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            metadata={
                "schema_version": "1",
                "embedding_model": settings.embedding_model,
                "embedding_dimension": 1024,
                "hnsw:space": "cosine",
            },
        )

    def upsert_chunks(self, chunks: Sequence[PaperChunk]) -> None:
        if not chunks:
            return
        # DashScope embedding endpoint rejects input.contents batches larger than 20.
        batch_size = 20
        for start in range(0, len(chunks), batch_size):
            batch = list(chunks[start : start + batch_size])
            embeddings = self.embedding_model.embed_documents([chunk.text for chunk in batch])
            self._call(
                self.collection.upsert,
                ids=[chunk.chunk_id for chunk in batch],
                documents=[chunk.text for chunk in batch],
                metadatas=[chunk.metadata() for chunk in batch],
                embeddings=embeddings,
            )

    def similarity_search(
        self, query: str, paper_versions: dict[str, int], top_k: int
    ) -> list[RetrievedChunk]:
        if not paper_versions:
            return []
        query_embedding = self.embedding_model.embed_query(query)
        all_results: list[RetrievedChunk] = []
        for paper_id, version in paper_versions.items():
            where = {
                "$and": [
                    {"paper_id": {"$eq": paper_id}},
                    {"paper_version": {"$eq": version}},
                ]
            }
            result = self._call(
                self.collection.query,
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
            ids = result.get("ids", [[]])[0]
            docs = result.get("documents", [[]])[0]
            metas = result.get("metadatas", [[]])[0]
            distances = result.get("distances", [[]])[0]
            for chunk_id, text, meta, distance in zip(ids, docs, metas, distances, strict=True):
                chunk = self._from_record(chunk_id, text, meta)
                all_results.append(RetrievedChunk(chunk=chunk, score=1.0 - float(distance)))
        all_results.sort(key=lambda item: item.score if item.score is not None else -1, reverse=True)
        return all_results

    def delete_paper(self, paper_id: str, version: int | None = None) -> None:
        if version is None:
            where = {"paper_id": {"$eq": paper_id}}
        else:
            where = {
                "$and": [
                    {"paper_id": {"$eq": paper_id}},
                    {"paper_version": {"$eq": version}},
                ]
            }
        self._call(self.collection.delete, where=where)

    def get_chunks(self, chunk_ids: Sequence[str]) -> list[PaperChunk]:
        if not chunk_ids:
            return []
        result = self._call(
            self.collection.get,
            ids=list(chunk_ids),
            include=["documents", "metadatas"],
        )
        by_id: dict[str, PaperChunk] = {}
        for chunk_id, text, meta in zip(
            result.get("ids", []),
            result.get("documents", []),
            result.get("metadatas", []),
            strict=True,
        ):
            by_id[chunk_id] = self._from_record(chunk_id, text, meta)
        return [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id]

    def get_paper_chunks(self, paper_id: str, version: int) -> list[PaperChunk]:
        where = {
            "$and": [
                {"paper_id": {"$eq": paper_id}},
                {"paper_version": {"$eq": version}},
            ]
        }
        result = self._call(
            self.collection.get, where=where, include=["documents", "metadatas"]
        )
        chunks = [
            self._from_record(chunk_id, text, meta)
            for chunk_id, text, meta in zip(
                result.get("ids", []),
                result.get("documents", []),
                result.get("metadatas", []),
                strict=True,
            )
        ]
        return sorted(chunks, key=lambda item: (item.page_number, item.chunk_index_on_page))

    def update_paper_title(self, paper_id: str, version: int, title: str) -> None:
        where = {
            "$and": [
                {"paper_id": {"$eq": paper_id}},
                {"paper_version": {"$eq": version}},
            ]
        }
        result = self._call(self.collection.get, where=where, include=["metadatas"])
        ids = result.get("ids", [])
        metadatas = result.get("metadatas", [])
        if ids:
            self._call(
                self.collection.update,
                ids=ids,
                metadatas=[{**metadata, "paper_title": title} for metadata in metadatas],
            )

    def count(self) -> int:
        return int(self._call(self.collection.count))

    def health_check(self) -> dict[str, str | int]:
        return {"status": "ok", "collection": self.collection.name, "count": self.count()}

    def close(self) -> None:
        self.client.close()

    @staticmethod
    def _call(operation, **kwargs) -> Any:
        try:
            return operation(**kwargs)
        except Exception as exc:
            message = str(exc).lower()
            if "hnsw" in message or "backfill request to compactor" in message:
                raise ResearchCopilotError(
                    "Chroma HNSW 持久化索引无法读取。请停止 Streamlit 后运行 "
                    "`research-copilot repair-chroma`；该命令会先在 staging 重建并验证，"
                    "再备份和切换生产索引。"
                ) from exc
            raise

    @staticmethod
    def _from_record(chunk_id: str, text: str, meta: dict) -> PaperChunk:
        return PaperChunk(
            chunk_id=chunk_id,
            paper_id=str(meta["paper_id"]),
            paper_version=int(meta["paper_version"]),
            paper_title=str(meta["paper_title"]),
            page_number=int(meta["page_number"]),
            page_index=int(meta["page_index"]),
            chunk_index_on_page=int(meta["chunk_index_on_page"]),
            text=text,
            text_hash=str(meta["text_hash"]),
            section_hint=str(meta.get("section_hint") or "") or None,
            source_uri=str(meta["source_uri"]),
            embedding_model=str(meta["embedding_model"]),
        )
