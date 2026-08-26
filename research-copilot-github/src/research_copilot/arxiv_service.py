from __future__ import annotations

import re
import time
from math import ceil
from pathlib import Path
from threading import Lock
from urllib.error import HTTPError

import arxiv

from research_copilot.config import Settings
from research_copilot.errors import ArxivTemporarilyUnavailableError, ResearchCopilotError
from research_copilot.ingestion import PaperIngestionService, safe_slug
from research_copilot.models import ArxivPaper, IngestionResult, SourceType
from research_copilot.storage import SQLiteRepository

ARXIV_ID_PATTERN = re.compile(
    r"^(?:https?://arxiv\.org/(?:abs|pdf)/)?(?P<id>(?:[a-z-]+(?:\.[A-Z]{2})?/\d{7}|\d{4}\.\d{4,5}))(?:v\d+)?(?:\.pdf)?$",
    re.IGNORECASE,
)
SEARCH_CACHE_TTL_SECONDS = 10 * 60
RATE_LIMIT_COOLDOWN_SECONDS = 60
SERVICE_COOLDOWN_SECONDS = 30


def normalize_arxiv_id(value: str) -> str:
    match = ARXIV_ID_PATTERN.match(value.strip())
    if not match:
        raise ResearchCopilotError(f"无效的 arXiv ID 或 URL：{value}")
    return match.group("id")


class ArxivService:
    def __init__(
        self,
        settings: Settings,
        repository: SQLiteRepository,
        ingestion: PaperIngestionService,
    ):
        self.settings = settings
        self.repository = repository
        self.ingestion = ingestion
        # arxiv.Client would otherwise silently retry HTTP 429/503 twice.  A
        # failed Agent tool call can then trigger another high-level call, which
        # compounds rate limiting. We expose one failure and apply a shared
        # process cooldown instead.
        self.client = arxiv.Client(page_size=20, delay_seconds=3.0, num_retries=0)
        self._search_cache: dict[tuple[str, int], tuple[float, list[ArxivPaper]]] = {}
        self._cooldown_until = 0.0
        self._cooldown_status: int | None = None
        self._lock = Lock()

    def search(self, query: str, max_results: int = 5) -> list[ArxivPaper]:
        if not query.strip():
            raise ResearchCopilotError("arXiv 检索词不能为空")
        limit = max(1, min(max_results, 20))
        cache_key = (" ".join(query.casefold().split()), limit)
        cached = self._get_cached_search(cache_key)
        if cached is not None:
            return cached
        self._ensure_not_cooling_down()
        search = arxiv.Search(
            query=query.strip(),
            max_results=limit,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        papers: list[ArxivPaper] = []
        try:
            for result in self.client.results(search):
                arxiv_id = normalize_arxiv_id(result.entry_id)
                papers.append(
                    ArxivPaper(
                        arxiv_id=arxiv_id,
                        title=" ".join(result.title.split()),
                        authors=[author.name for author in result.authors],
                        abstract=" ".join(result.summary.split()),
                        published_at=result.published,
                        updated_at=result.updated,
                        categories=list(result.categories),
                        entry_url=result.entry_id,
                        pdf_url=result.pdf_url,
                        already_imported=self.repository.arxiv_id_exists(arxiv_id),
                    )
                )
        except HTTPError as exc:
            self._raise_http_error(exc)
        self._save_cached_search(cache_key, papers)
        return [item.model_copy(deep=True) for item in papers]

    def import_paper(
        self, arxiv_id_or_url: str, *, prefer_mineru: bool | None = None
    ) -> IngestionResult:
        self._ensure_not_cooling_down()
        arxiv_id = normalize_arxiv_id(arxiv_id_or_url)
        try:
            result = next(
                iter(self.client.results(arxiv.Search(id_list=[arxiv_id], max_results=1))),
                None,
            )
        except HTTPError as exc:
            self._raise_http_error(exc)
        if result is None:
            raise ResearchCopilotError(f"arXiv 未找到论文：{arxiv_id}")
        title = " ".join(result.title.split())
        download_dir = self.settings.uploads_dir / "arxiv"
        download_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{safe_slug(title)}-{arxiv_id.replace('/', '-')}.pdf"
        pdf_path = Path(result.download_pdf(dirpath=str(download_dir), filename=filename))
        return self.ingestion.ingest_local(
            pdf_path,
            title=title,
            paper_id=f"arxiv-{arxiv_id.replace('/', '-')}",
            source_type=SourceType.ARXIV,
            source_uri=result.entry_id,
            authors=[author.name for author in result.authors],
            abstract=" ".join(result.summary.split()),
            arxiv_id=arxiv_id,
            prefer_mineru=prefer_mineru,
        )

    def _get_cached_search(self, cache_key: tuple[str, int]) -> list[ArxivPaper] | None:
        with self._lock:
            cached = self._search_cache.get(cache_key)
            if cached is None:
                return None
            saved_at, papers = cached
            if time.monotonic() - saved_at > SEARCH_CACHE_TTL_SECONDS:
                del self._search_cache[cache_key]
                return None
            return [item.model_copy(deep=True) for item in papers]

    def _save_cached_search(self, cache_key: tuple[str, int], papers: list[ArxivPaper]) -> None:
        with self._lock:
            self._search_cache[cache_key] = (
                time.monotonic(),
                [item.model_copy(deep=True) for item in papers],
            )

    def _ensure_not_cooling_down(self) -> None:
        with self._lock:
            remaining = self._cooldown_until - time.monotonic()
            status_code = self._cooldown_status
        if remaining > 0 and status_code is not None:
            raise ArxivTemporarilyUnavailableError(status_code, ceil(remaining))

    def _raise_http_error(self, exc: HTTPError) -> None:
        status_code = int(exc.code)
        if status_code == 429:
            wait_seconds = self._retry_after_seconds(exc, RATE_LIMIT_COOLDOWN_SECONDS)
        elif status_code >= 500:
            wait_seconds = self._retry_after_seconds(exc, SERVICE_COOLDOWN_SECONDS)
        else:
            raise ResearchCopilotError(f"arXiv 请求失败（HTTP {status_code}）：{exc.reason}") from exc
        with self._lock:
            self._cooldown_until = max(self._cooldown_until, time.monotonic() + wait_seconds)
            self._cooldown_status = status_code
        raise ArxivTemporarilyUnavailableError(status_code, wait_seconds) from exc

    @staticmethod
    def _retry_after_seconds(exc: HTTPError, fallback: int) -> int:
        value = (exc.headers or {}).get("Retry-After")
        try:
            return max(1, min(int(value), 15 * 60))
        except (TypeError, ValueError):
            return fallback
