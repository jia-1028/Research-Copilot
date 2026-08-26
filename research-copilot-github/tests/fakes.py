from __future__ import annotations

from collections.abc import Sequence

from langchain_core.embeddings import Embeddings

from research_copilot.models import PaperChunk, RetrievedChunk


class DeterministicEmbeddings(Embeddings):
    def _vector(self, text: str) -> list[float]:
        lowered = text.lower()
        return [
            float(lowered.count("attention") + lowered.count("method")),
            float(lowered.count("dataset") + lowered.count("metric")),
            1.0,
        ]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class MemoryVectorIndex:
    def __init__(self, fail_upsert: bool = False):
        self.items: dict[str, PaperChunk] = {}
        self.fail_upsert = fail_upsert
        self.search_calls = 0

    def upsert_chunks(self, chunks: Sequence[PaperChunk]) -> None:
        if self.fail_upsert:
            raise RuntimeError("simulated vector write failure")
        self.items.update({item.chunk_id: item for item in chunks})

    def similarity_search(
        self, query: str, paper_versions: dict[str, int], top_k: int
    ) -> list[RetrievedChunk]:
        self.search_calls += 1
        matches = [
            item
            for item in self.items.values()
            if paper_versions.get(item.paper_id) == item.paper_version
        ]
        return [RetrievedChunk(chunk=item, score=1.0) for item in matches[:top_k]]

    def delete_paper(self, paper_id: str, version: int | None = None) -> None:
        self.items = {
            key: value
            for key, value in self.items.items()
            if not (
                value.paper_id == paper_id
                and (version is None or value.paper_version == version)
            )
        }

    def get_chunks(self, chunk_ids: Sequence[str]) -> list[PaperChunk]:
        return [self.items[item] for item in chunk_ids if item in self.items]

    def get_paper_chunks(self, paper_id: str, version: int) -> list[PaperChunk]:
        return sorted(
            [
                item
                for item in self.items.values()
                if item.paper_id == paper_id and item.paper_version == version
            ],
            key=lambda item: (item.page_number, item.chunk_index_on_page),
        )

    def update_paper_title(self, paper_id: str, version: int, title: str) -> None:
        for chunk_id, item in list(self.items.items()):
            if item.paper_id == paper_id and item.paper_version == version:
                self.items[chunk_id] = item.model_copy(update={"paper_title": title})

    def count(self) -> int:
        return len(self.items)
