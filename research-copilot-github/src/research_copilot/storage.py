from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from research_copilot.models import IngestionStatus, Paper, PaperVersion


def _now() -> str:
    return datetime.now(UTC).isoformat()


SCHEMA_VERSION = 2


class SQLiteRepository:
    """Application metadata repository; Chroma remains the chunk store."""

    def __init__(self, path: Path, *, checkpoint_path: Path | None = None):
        self.path = path
        self.checkpoint_path = checkpoint_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize(self) -> None:
        needs_backup = False
        if self.path.exists() and self.path.stat().st_size:
            with closing(sqlite3.connect(self.path)) as probe:
                version = int(probe.execute("PRAGMA user_version").fetchone()[0])
                has_tables = bool(
                    probe.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' LIMIT 1"
                    ).fetchone()
                )
                needs_backup = has_tables and version < SCHEMA_VERSION
        if needs_backup:
            self._backup_databases()
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS papers (
                    paper_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_uri TEXT NOT NULL,
                    document_role TEXT NOT NULL DEFAULT 'main',
                    parent_paper_id TEXT,
                    active_version INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'pending',
                    authors_json TEXT NOT NULL DEFAULT '[]',
                    abstract TEXT,
                    arxiv_id TEXT,
                    page_count INTEGER NOT NULL DEFAULT 0,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(parent_paper_id) REFERENCES papers(paper_id)
                );

                CREATE TABLE IF NOT EXISTS paper_versions (
                    paper_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    original_path TEXT NOT NULL,
                    managed_copy_path TEXT NOT NULL,
                    parser_name TEXT NOT NULL,
                    parser_version TEXT NOT NULL,
                    parsed_dir TEXT NOT NULL,
                    page_count INTEGER NOT NULL DEFAULT 0,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(paper_id, version),
                    UNIQUE(sha256),
                    FOREIGN KEY(paper_id) REFERENCES papers(paper_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS ingestion_jobs (
                    job_id TEXT PRIMARY KEY,
                    paper_id TEXT,
                    version INTEGER,
                    status TEXT NOT NULL,
                    progress REAL NOT NULL DEFAULT 0,
                    current_step TEXT NOT NULL DEFAULT '',
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS retrieval_traces (
                    trace_id TEXT PRIMARY KEY,
                    thread_id TEXT,
                    question TEXT NOT NULL,
                    standalone_query TEXT NOT NULL,
                    paper_ids_json TEXT NOT NULL,
                    paper_versions_json TEXT NOT NULL,
                    retrieved_chunk_ids_json TEXT NOT NULL,
                    used_chunk_ids_json TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS conversations (
                    thread_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    active_paper_ids_json TEXT NOT NULL DEFAULT '[]',
                    last_arxiv_result_ids_json TEXT NOT NULL DEFAULT '[]',
                    last_retrieval_trace_id TEXT,
                    pending_ingestion_job_id TEXT,
                    scope_type TEXT NOT NULL DEFAULT 'general',
                    scope_key TEXT NOT NULL DEFAULT 'general',
                    summary_json TEXT,
                    summary_through_sequence INTEGER NOT NULL DEFAULT 0,
                    archived_at TEXT,
                    read_only_reason TEXT,
                    last_message_at TEXT,
                    pending_turn_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS conversation_papers (
                    conversation_id TEXT NOT NULL,
                    paper_id TEXT NOT NULL,
                    paper_title_snapshot TEXT NOT NULL,
                    paper_version_snapshot INTEGER NOT NULL DEFAULT 0,
                    position INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(conversation_id, paper_id),
                    FOREIGN KEY(conversation_id) REFERENCES conversations(thread_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS conversation_messages (
                    message_id TEXT PRIMARY KEY,
                    turn_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user','assistant')),
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    content TEXT NOT NULL DEFAULT '',
                    original_query TEXT,
                    standalone_query TEXT,
                    process_json TEXT NOT NULL DEFAULT '[]',
                    payload_json TEXT,
                    error TEXT,
                    retrieval_trace_id TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    UNIQUE(conversation_id, sequence),
                    FOREIGN KEY(conversation_id) REFERENCES conversations(thread_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS message_citations (
                    message_id TEXT NOT NULL,
                    citation_id TEXT NOT NULL,
                    paper_id TEXT NOT NULL,
                    paper_title_snapshot TEXT NOT NULL,
                    paper_version INTEGER NOT NULL,
                    pdf_page INTEGER NOT NULL,
                    chunk_id TEXT NOT NULL,
                    evidence_text TEXT NOT NULL,
                    retrieval_score REAL,
                    PRIMARY KEY(message_id, citation_id),
                    FOREIGN KEY(message_id) REFERENCES conversation_messages(message_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id TEXT PRIMARY KEY,
                    preferred_language TEXT NOT NULL DEFAULT 'zh-CN',
                    answer_detail_level TEXT NOT NULL DEFAULT 'standard',
                    citation_style TEXT NOT NULL DEFAULT 'inline',
                    show_evidence INTEGER NOT NULL DEFAULT 1,
                    default_comparison_dimensions_json TEXT NOT NULL DEFAULT '[]',
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS paper_profiles (
                    paper_id TEXT NOT NULL,
                    paper_version INTEGER NOT NULL,
                    profile_mode TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    profile_json TEXT NOT NULL,
                    citations_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(paper_id, paper_version, profile_mode, prompt_version),
                    FOREIGN KEY(paper_id) REFERENCES papers(paper_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS paper_comparisons (
                    cache_key TEXT PRIMARY KEY,
                    prompt_version TEXT NOT NULL,
                    comparison_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS background_tasks (
                    task_id TEXT PRIMARY KEY,
                    task_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    result_json TEXT,
                    progress REAL NOT NULL DEFAULT 0,
                    current_step TEXT NOT NULL DEFAULT '',
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    updated_at TEXT NOT NULL
                );
                """
            )
            self._ensure_conversation_columns(conn)
            conn.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_conversations_scope
                    ON conversations(scope_key, archived_at, last_message_at);
                CREATE INDEX IF NOT EXISTS idx_conversation_messages_thread
                    ON conversation_messages(conversation_id, sequence);
                CREATE INDEX IF NOT EXISTS idx_conversation_messages_turn
                    ON conversation_messages(conversation_id, turn_id);
                CREATE INDEX IF NOT EXISTS idx_message_citations_paper
                    ON message_citations(paper_id);
                CREATE INDEX IF NOT EXISTS idx_background_tasks_status
                    ON background_tasks(status, updated_at);
                """
            )
            self._migrate_legacy_conversation_scopes(conn)
            conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    def _backup_databases(self) -> None:
        backup_dir = self.path.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        for source in (self.path, self.checkpoint_path):
            if source is None or not source.exists() or not source.stat().st_size:
                continue
            target = backup_dir / f"{source.stem}-{stamp}.sqlite.bak"
            with (
                closing(sqlite3.connect(source)) as source_conn,
                closing(sqlite3.connect(target)) as target_conn,
            ):
                source_conn.backup(target_conn)

    @staticmethod
    def _ensure_conversation_columns(conn: sqlite3.Connection) -> None:
        existing = {row[1] for row in conn.execute("PRAGMA table_info(conversations)")}
        additions = {
            "scope_type": "TEXT NOT NULL DEFAULT 'general'",
            "scope_key": "TEXT NOT NULL DEFAULT 'general'",
            "summary_json": "TEXT",
            "summary_through_sequence": "INTEGER NOT NULL DEFAULT 0",
            "archived_at": "TEXT",
            "read_only_reason": "TEXT",
            "last_message_at": "TEXT",
            "pending_turn_id": "TEXT",
        }
        for name, declaration in additions.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE conversations ADD COLUMN {name} {declaration}")

    @staticmethod
    def _migrate_legacy_conversation_scopes(conn: sqlite3.Connection) -> None:
        papers = {
            row["paper_id"]: dict(row)
            for row in conn.execute(
                "SELECT paper_id,title,parent_paper_id,active_version FROM papers"
            ).fetchall()
        }
        rows = conn.execute(
            "SELECT thread_id,active_paper_ids_json,scope_key FROM conversations"
        ).fetchall()
        for row in rows:
            if row["scope_key"] != "general":
                continue
            try:
                selected = json.loads(row["active_paper_ids_json"] or "[]")
            except json.JSONDecodeError:
                selected = []
            roots = sorted(
                {
                    papers.get(paper_id, {}).get("parent_paper_id") or paper_id
                    for paper_id in selected
                }
            )
            if not roots:
                scope_type, scope_key = "general", "general"
            elif len(roots) == 1:
                scope_type, scope_key = "paper_family", f"paper:{roots[0]}"
            else:
                scope_type, scope_key = "paper_set", "papers:" + "|".join(roots)
            conn.execute(
                "UPDATE conversations SET scope_type=?,scope_key=? WHERE thread_id=?",
                (scope_type, scope_key, row["thread_id"]),
            )
            for position, paper_id in enumerate(roots):
                paper = papers.get(paper_id, {})
                conn.execute(
                    """
                    INSERT OR IGNORE INTO conversation_papers(
                        conversation_id,paper_id,paper_title_snapshot,
                        paper_version_snapshot,position,created_at
                    ) VALUES(?,?,?,?,?,?)
                    """,
                    (
                        row["thread_id"],
                        paper_id,
                        paper.get("title", paper_id),
                        int(paper.get("active_version", 0)),
                        position,
                        _now(),
                    ),
                )

    def upsert_paper(self, paper: Paper) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO papers (
                    paper_id,title,source_type,source_uri,document_role,parent_paper_id,
                    active_version,status,authors_json,abstract,arxiv_id,page_count,chunk_count,
                    created_at,updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(paper_id) DO UPDATE SET
                    title=excluded.title, source_type=excluded.source_type,
                    source_uri=excluded.source_uri, document_role=excluded.document_role,
                    parent_paper_id=excluded.parent_paper_id, status=excluded.status,
                    authors_json=excluded.authors_json, abstract=excluded.abstract,
                    arxiv_id=excluded.arxiv_id, updated_at=excluded.updated_at
                """,
                (
                    paper.paper_id,
                    paper.title,
                    paper.source_type.value,
                    paper.source_uri,
                    paper.document_role.value,
                    paper.parent_paper_id,
                    paper.active_version,
                    paper.status.value,
                    json.dumps(paper.authors, ensure_ascii=False),
                    paper.abstract,
                    paper.arxiv_id,
                    paper.page_count,
                    paper.chunk_count,
                    paper.created_at.isoformat(),
                    paper.updated_at.isoformat(),
                ),
            )

    def add_version(self, version: PaperVersion) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO paper_versions (
                    paper_id,version,sha256,original_path,managed_copy_path,parser_name,
                    parser_version,parsed_dir,page_count,chunk_count,status,created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    version.paper_id,
                    version.version,
                    version.sha256,
                    version.original_path,
                    version.managed_copy_path,
                    version.parser_name,
                    version.parser_version,
                    version.parsed_dir,
                    version.page_count,
                    version.chunk_count,
                    version.status.value,
                    version.created_at.isoformat(),
                ),
            )

    def next_version(self, paper_id: str) -> int:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(version), 0) + 1 AS value FROM paper_versions WHERE paper_id=?",
                (paper_id,),
            ).fetchone()
        return int(row["value"])

    def find_by_sha256(self, sha256: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT v.*, p.title, p.status AS paper_status
                FROM paper_versions v JOIN papers p ON p.paper_id=v.paper_id
                WHERE v.sha256=?
                """,
                (sha256,),
            ).fetchone()
        return dict(row) if row else None

    def delete_failed_version(self, paper_id: str, version: int) -> None:
        """Allow an identical source to be retried after a fully rolled-back failure."""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT status FROM paper_versions WHERE paper_id=? AND version=?",
                (paper_id, version),
            ).fetchone()
            if row and row["status"] == IngestionStatus.FAILED.value:
                conn.execute(
                    "DELETE FROM paper_versions WHERE paper_id=? AND version=?",
                    (paper_id, version),
                )

    def update_job(
        self,
        job_id: str,
        status: IngestionStatus,
        progress: float,
        current_step: str,
        *,
        paper_id: str | None = None,
        version: int | None = None,
        error: str | None = None,
    ) -> None:
        now = _now()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_jobs (
                    job_id,paper_id,version,status,progress,current_step,error,created_at,updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(job_id) DO UPDATE SET
                    paper_id=COALESCE(excluded.paper_id, ingestion_jobs.paper_id),
                    version=COALESCE(excluded.version, ingestion_jobs.version),
                    status=excluded.status, progress=excluded.progress,
                    current_step=excluded.current_step, error=excluded.error,
                    updated_at=excluded.updated_at
                """,
                (job_id, paper_id, version, status.value, progress, current_step, error, now, now),
            )

    def finish_version(
        self,
        paper_id: str,
        version: int,
        *,
        parser_name: str,
        parser_version: str,
        page_count: int,
        chunk_count: int,
        status: IngestionStatus,
    ) -> None:
        now = _now()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE paper_versions SET parser_name=?, parser_version=?, page_count=?,
                    chunk_count=?, status=? WHERE paper_id=? AND version=?
                """,
                (
                    parser_name,
                    parser_version,
                    page_count,
                    chunk_count,
                    status.value,
                    paper_id,
                    version,
                ),
            )
            if status == IngestionStatus.READY:
                conn.execute(
                    """
                    UPDATE papers SET active_version=?, status=?, page_count=?, chunk_count=?,
                        updated_at=? WHERE paper_id=?
                    """,
                    (version, status.value, page_count, chunk_count, now, paper_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE papers SET
                        status=CASE WHEN active_version > 0 THEN 'ready' ELSE ? END,
                        updated_at=? WHERE paper_id=?
                    """,
                    (status.value, now, paper_id),
                )

    def get_paper(self, paper_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM papers WHERE paper_id=?", (paper_id,)).fetchone()
        return self._decode_paper(row) if row else None

    def get_version(self, paper_id: str, version: int) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM paper_versions WHERE paper_id=? AND version=?",
                (paper_id, version),
            ).fetchone()
        return dict(row) if row else None

    def update_paper_title(self, paper_id: str, title: str) -> None:
        with self.connect() as conn:
            cursor = conn.execute(
                "UPDATE papers SET title=?, updated_at=? WHERE paper_id=?",
                (title, _now(), paper_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(paper_id)
            conn.execute("DELETE FROM paper_profiles WHERE paper_id=?", (paper_id,))
            conn.execute("DELETE FROM paper_comparisons")

    def invalidate_profile_cache(self, paper_id: str, version: int | None = None) -> None:
        with self.connect() as conn:
            if version is None:
                conn.execute("DELETE FROM paper_profiles WHERE paper_id=?", (paper_id,))
            else:
                conn.execute(
                    "DELETE FROM paper_profiles WHERE paper_id=? AND paper_version=?",
                    (paper_id, version),
                )
            conn.execute("DELETE FROM paper_comparisons")

    def get_cached_comparison(self, cache_key: str, prompt_version: str) -> dict | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT comparison_json FROM paper_comparisons
                WHERE cache_key=? AND prompt_version=?
                """,
                (cache_key, prompt_version),
            ).fetchone()
        return json.loads(row["comparison_json"]) if row else None

    def save_cached_comparison(
        self, cache_key: str, prompt_version: str, comparison: dict[str, Any]
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO paper_comparisons (
                    cache_key,prompt_version,comparison_json,created_at
                ) VALUES (?,?,?,?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    prompt_version=excluded.prompt_version,
                    comparison_json=excluded.comparison_json,
                    created_at=excluded.created_at
                """,
                (
                    cache_key,
                    prompt_version,
                    json.dumps(comparison, ensure_ascii=False),
                    _now(),
                ),
            )

    def get_cached_profile(
        self,
        paper_id: str,
        paper_version: int,
        *,
        profile_mode: str,
        prompt_version: str,
    ) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT profile_json, citations_json, created_at
                FROM paper_profiles
                WHERE paper_id=? AND paper_version=? AND profile_mode=? AND prompt_version=?
                """,
                (paper_id, paper_version, profile_mode, prompt_version),
            ).fetchone()
        if not row:
            return None
        return {
            "profile": json.loads(row["profile_json"]),
            "citations": json.loads(row["citations_json"]),
            "created_at": row["created_at"],
        }

    def save_cached_profile(
        self,
        paper_id: str,
        paper_version: int,
        *,
        profile_mode: str,
        prompt_version: str,
        profile: dict[str, Any],
        citations: list[dict[str, Any]],
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO paper_profiles (
                    paper_id,paper_version,profile_mode,prompt_version,
                    profile_json,citations_json,created_at
                ) VALUES (?,?,?,?,?,?,?)
                ON CONFLICT(paper_id,paper_version,profile_mode,prompt_version)
                DO UPDATE SET profile_json=excluded.profile_json,
                    citations_json=excluded.citations_json,
                    created_at=excluded.created_at
                """,
                (
                    paper_id,
                    paper_version,
                    profile_mode,
                    prompt_version,
                    json.dumps(profile, ensure_ascii=False),
                    json.dumps(citations, ensure_ascii=False),
                    _now(),
                ),
            )

    def list_papers(self, keyword: str | None = None, status: str | None = "ready") -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if keyword:
            clauses.append("(title LIKE ? OR paper_id LIKE ?)")
            params.extend([f"%{keyword}%", f"%{keyword}%"])
        if status:
            clauses.append("status=?")
            params.append(status)
        sql = "SELECT * FROM papers"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY updated_at DESC"
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._decode_paper(row) for row in rows]

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM ingestion_jobs WHERE job_id=?", (job_id,)).fetchone()
        return dict(row) if row else None

    def get_job_for_paper(self, paper_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM ingestion_jobs WHERE paper_id=? ORDER BY updated_at DESC LIMIT 1",
                (paper_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_jobs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM ingestion_jobs ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def create_background_task(
        self, task_id: str, task_type: str, request: dict[str, Any]
    ) -> None:
        now = _now()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO background_tasks(
                    task_id,task_type,status,request_json,progress,current_step,
                    cancel_requested,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (
                    task_id,
                    task_type,
                    "queued",
                    json.dumps(request, ensure_ascii=False),
                    0.0,
                    "等待执行",
                    0,
                    now,
                    now,
                ),
            )

    def get_background_task(self, task_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM background_tasks WHERE task_id=?", (task_id,)
            ).fetchone()
        return self._decode_background_task(row) if row else None

    def list_background_tasks(
        self, *, task_types: list[str] | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        params: list[Any] = []
        sql = "SELECT * FROM background_tasks"
        if task_types:
            placeholders = ",".join("?" for _ in task_types)
            sql += f" WHERE task_type IN ({placeholders})"
            params.extend(task_types)
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._decode_background_task(row) for row in rows]

    def recover_background_tasks(self) -> list[str]:
        now = _now()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE background_tasks SET status='queued',progress=0,
                    current_step='应用重启后等待恢复',started_at=NULL,updated_at=?
                WHERE status='running' AND cancel_requested=0
                """,
                (now,),
            )
            conn.execute(
                """
                UPDATE background_tasks SET status='cancelled',progress=1,
                    current_step='已取消',completed_at=?,updated_at=?
                WHERE status IN ('queued','running') AND cancel_requested=1
                """,
                (now, now),
            )
            rows = conn.execute(
                "SELECT task_id FROM background_tasks WHERE status='queued' ORDER BY created_at"
            ).fetchall()
        return [str(row["task_id"]) for row in rows]

    def claim_background_task(self, task_id: str) -> bool:
        now = _now()
        with self.connect() as conn:
            cursor = conn.execute(
                """
                UPDATE background_tasks SET status='running',progress=0.05,
                    current_step='开始执行',started_at=COALESCE(started_at,?),updated_at=?
                WHERE task_id=? AND status='queued' AND cancel_requested=0
                """,
                (now, now, task_id),
            )
        return cursor.rowcount == 1

    def update_background_task(
        self,
        task_id: str,
        *,
        progress: float,
        current_step: str,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE background_tasks SET progress=?,current_step=?,updated_at=?
                WHERE task_id=? AND status='running'
                """,
                (max(0.0, min(progress, 1.0)), current_step, _now(), task_id),
            )

    def finish_background_task(
        self,
        task_id: str,
        *,
        status: str,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        if status not in {"completed", "failed", "cancelled"}:
            raise ValueError(f"无效后台任务终态：{status}")
        now = _now()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE background_tasks SET status=?,result_json=?,error=?,progress=1,
                    current_step=?,completed_at=?,updated_at=? WHERE task_id=?
                """,
                (
                    status,
                    json.dumps(result, ensure_ascii=False) if result is not None else None,
                    error,
                    {"completed": "已完成", "failed": "执行失败", "cancelled": "已取消"}[
                        status
                    ],
                    now,
                    now,
                    task_id,
                ),
            )

    def request_background_task_cancel(self, task_id: str) -> bool:
        now = _now()
        with self.connect() as conn:
            row = conn.execute(
                "SELECT status FROM background_tasks WHERE task_id=?", (task_id,)
            ).fetchone()
            if row is None or row["status"] not in {"queued", "running"}:
                return False
            if row["status"] == "queued":
                conn.execute(
                    """
                    UPDATE background_tasks SET cancel_requested=1,status='cancelled',
                        progress=1,current_step='已取消',completed_at=?,updated_at=?
                    WHERE task_id=?
                    """,
                    (now, now, task_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE background_tasks SET cancel_requested=1,
                        current_step='已请求取消，当前步骤结束后停止',updated_at=?
                    WHERE task_id=?
                    """,
                    (now, task_id),
                )
        return True

    def background_task_cancel_requested(self, task_id: str) -> bool:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT cancel_requested FROM background_tasks WHERE task_id=?", (task_id,)
            ).fetchone()
        return bool(row and row["cancel_requested"])

    @staticmethod
    def _decode_background_task(row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        value["request"] = json.loads(value.pop("request_json") or "{}")
        raw_result = value.pop("result_json")
        value["result"] = json.loads(raw_result) if raw_result else None
        value["cancel_requested"] = bool(value["cancel_requested"])
        return value

    def save_conversation_state(
        self,
        thread_id: str,
        *,
        title: str = "Research Copilot conversation",
        active_paper_ids: list[str] | None = None,
        last_arxiv_result_ids: list[str] | None = None,
        last_retrieval_trace_id: str | None = None,
        pending_ingestion_job_id: str | None = None,
    ) -> None:
        now = _now()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO conversations (
                    thread_id,title,active_paper_ids_json,last_arxiv_result_ids_json,
                    last_retrieval_trace_id,pending_ingestion_job_id,created_at,updated_at
                ) VALUES (?,?,?,?,?,?,?,?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    active_paper_ids_json=excluded.active_paper_ids_json,
                    last_arxiv_result_ids_json=excluded.last_arxiv_result_ids_json,
                    last_retrieval_trace_id=excluded.last_retrieval_trace_id,
                    pending_ingestion_job_id=excluded.pending_ingestion_job_id,
                    updated_at=excluded.updated_at
                """,
                (
                    thread_id,
                    title,
                    json.dumps(active_paper_ids or [], ensure_ascii=False),
                    json.dumps(last_arxiv_result_ids or [], ensure_ascii=False),
                    last_retrieval_trace_id,
                    pending_ingestion_job_id,
                    now,
                    now,
                ),
            )

    def get_conversation_state(self, thread_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE thread_id=?", (thread_id,)
            ).fetchone()
        if not row:
            return None
        result = dict(row)
        result["active_paper_ids"] = json.loads(result.pop("active_paper_ids_json"))
        result["last_arxiv_result_ids"] = json.loads(
            result.pop("last_arxiv_result_ids_json")
        )
        return result

    def create_conversation(
        self,
        thread_id: str,
        *,
        title: str,
        scope_type: str,
        scope_key: str,
        papers: list[dict[str, Any]],
    ) -> dict[str, Any]:
        now = _now()
        paper_ids = [item["paper_id"] for item in papers]
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO conversations(
                    thread_id,title,active_paper_ids_json,last_arxiv_result_ids_json,
                    scope_type,scope_key,created_at,updated_at,last_message_at
                ) VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (
                    thread_id,
                    title,
                    json.dumps(paper_ids, ensure_ascii=False),
                    "[]",
                    scope_type,
                    scope_key,
                    now,
                    now,
                    now,
                ),
            )
            for position, paper in enumerate(papers):
                conn.execute(
                    """
                    INSERT INTO conversation_papers(
                        conversation_id,paper_id,paper_title_snapshot,
                        paper_version_snapshot,position,created_at
                    ) VALUES(?,?,?,?,?,?)
                    """,
                    (
                        thread_id,
                        paper["paper_id"],
                        paper["title"],
                        int(paper.get("active_version", 0)),
                        position,
                        now,
                    ),
                )
        return self.get_conversation(thread_id) or {}

    def get_conversation(self, thread_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE thread_id=?", (thread_id,)
            ).fetchone()
            if not row:
                return None
            papers = conn.execute(
                """
                SELECT paper_id,paper_title_snapshot,paper_version_snapshot,position
                FROM conversation_papers WHERE conversation_id=? ORDER BY position
                """,
                (thread_id,),
            ).fetchall()
        result = dict(row)
        result["paper_ids"] = [item["paper_id"] for item in papers]
        result["paper_snapshots"] = [dict(item) for item in papers]
        result["summary"] = json.loads(result["summary_json"]) if result["summary_json"] else None
        return result

    def list_conversations(
        self,
        *,
        scope_key: str | None = None,
        include_archived: bool = False,
        require_messages: bool = True,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if scope_key is not None:
            clauses.append("c.scope_key=?")
            params.append(scope_key)
        if not include_archived:
            clauses.append("c.archived_at IS NULL")
        if require_messages:
            clauses.append("EXISTS(SELECT 1 FROM conversation_messages m WHERE m.conversation_id=c.thread_id)")
        sql = "SELECT c.* FROM conversations c"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY COALESCE(c.last_message_at,c.updated_at) DESC"
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def rename_conversation(self, thread_id: str, title: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE conversations SET title=?,updated_at=? WHERE thread_id=?",
                (title.strip(), _now(), thread_id),
            )

    def archive_conversation(self, thread_id: str, *, archived: bool = True) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE conversations SET archived_at=?,updated_at=? WHERE thread_id=?",
                (_now() if archived else None, _now(), thread_id),
            )

    def delete_conversation(self, thread_id: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM retrieval_traces WHERE thread_id=?", (thread_id,))
            conn.execute("DELETE FROM conversations WHERE thread_id=?", (thread_id,))

    def start_conversation_turn(
        self,
        thread_id: str,
        *,
        mode: str,
        original_query: str,
        standalone_query: str,
        turn_id: str | None = None,
    ) -> tuple[str, str, str]:
        turn_id = turn_id or str(uuid.uuid4())
        user_message_id = str(uuid.uuid4())
        assistant_message_id = str(uuid.uuid4())
        now = _now()
        with self.connect() as conn:
            conversation = conn.execute(
                "SELECT read_only_reason,pending_turn_id FROM conversations WHERE thread_id=?",
                (thread_id,),
            ).fetchone()
            if conversation is None:
                raise KeyError(thread_id)
            if conversation["read_only_reason"]:
                raise ValueError(conversation["read_only_reason"])
            if conversation["pending_turn_id"]:
                raise ValueError("该会话已有未完成请求，请先完成或重试上一轮")
            sequence = int(
                conn.execute(
                    "SELECT COALESCE(MAX(sequence),0)+1 FROM conversation_messages WHERE conversation_id=?",
                    (thread_id,),
                ).fetchone()[0]
            )
            conn.execute(
                """
                INSERT INTO conversation_messages(
                    message_id,turn_id,conversation_id,sequence,role,mode,status,content,
                    original_query,standalone_query,created_at,completed_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    user_message_id,
                    turn_id,
                    thread_id,
                    sequence,
                    "user",
                    mode,
                    "completed",
                    original_query,
                    original_query,
                    standalone_query,
                    now,
                    now,
                ),
            )
            conn.execute(
                """
                INSERT INTO conversation_messages(
                    message_id,turn_id,conversation_id,sequence,role,mode,status,content,
                    original_query,standalone_query,created_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    assistant_message_id,
                    turn_id,
                    thread_id,
                    sequence + 1,
                    "assistant",
                    mode,
                    "pending",
                    "",
                    original_query,
                    standalone_query,
                    now,
                ),
            )
            conn.execute(
                """
                UPDATE conversations SET pending_turn_id=?,last_message_at=?,updated_at=?
                WHERE thread_id=?
                """,
                (turn_id, now, now, thread_id),
            )
        return turn_id, user_message_id, assistant_message_id

    def update_turn_progress(
        self, assistant_message_id: str, process: list[str], *, status: str = "running"
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE conversation_messages SET process_json=?,status=? WHERE message_id=?",
                (json.dumps(process, ensure_ascii=False), status, assistant_message_id),
            )

    def finish_conversation_turn(
        self,
        assistant_message_id: str,
        *,
        content: str,
        process: list[str] | None = None,
        payload: dict[str, Any] | None = None,
        status: str = "completed",
        error: str | None = None,
        retrieval_trace_id: str | None = None,
    ) -> None:
        now = _now()
        process = process or []
        with self.connect() as conn:
            row = conn.execute(
                "SELECT conversation_id,turn_id FROM conversation_messages WHERE message_id=?",
                (assistant_message_id,),
            ).fetchone()
            if row is None:
                raise KeyError(assistant_message_id)
            conn.execute(
                """
                UPDATE conversation_messages SET content=?,process_json=?,payload_json=?,
                    status=?,error=?,retrieval_trace_id=?,completed_at=? WHERE message_id=?
                """,
                (
                    content,
                    json.dumps(process, ensure_ascii=False),
                    json.dumps(payload, ensure_ascii=False) if payload is not None else None,
                    status,
                    error,
                    retrieval_trace_id,
                    now,
                    assistant_message_id,
                ),
            )
            conn.execute(
                """
                UPDATE conversations SET pending_turn_id=NULL,last_message_at=?,updated_at=?
                WHERE thread_id=? AND pending_turn_id=?
                """,
                (now, now, row["conversation_id"], row["turn_id"]),
            )
            conn.execute("DELETE FROM message_citations WHERE message_id=?", (assistant_message_id,))
            for citation in (payload or {}).get("citations", []):
                conn.execute(
                    """
                    INSERT INTO message_citations(
                        message_id,citation_id,paper_id,paper_title_snapshot,paper_version,
                        pdf_page,chunk_id,evidence_text,retrieval_score
                    ) VALUES(?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        assistant_message_id,
                        citation["citation_id"],
                        citation["paper_id"],
                        citation["paper_title"],
                        int(citation["paper_version"]),
                        int(citation["pdf_page"]),
                        citation["chunk_id"],
                        citation["evidence_text"],
                        citation.get("retrieval_score"),
                    ),
                )

    def get_conversation_messages(self, thread_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM conversation_messages WHERE conversation_id=? ORDER BY sequence",
                (thread_id,),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["process"] = json.loads(item.pop("process_json") or "[]")
            payload_json = item.pop("payload_json")
            item["payload"] = json.loads(payload_json) if payload_json else None
            result.append(item)
        return result

    def get_conversation_messages_by_ids(
        self, message_ids: list[str]
    ) -> list[dict[str, Any]]:
        if not message_ids:
            return []
        placeholders = ",".join("?" for _ in message_ids)
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM conversation_messages WHERE message_id IN ({placeholders})",
                message_ids,
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["process"] = json.loads(item.pop("process_json") or "[]")
            payload_json = item.pop("payload_json")
            item["payload"] = json.loads(payload_json) if payload_json else None
            result.append(item)
        return result

    def import_conversation_message(
        self,
        thread_id: str,
        *,
        role: str,
        content: str,
        mode: str = "standard_agent",
        status: str = "completed",
        turn_id: str | None = None,
    ) -> None:
        now = _now()
        turn_id = turn_id or str(uuid.uuid4())
        with self.connect() as conn:
            sequence = int(
                conn.execute(
                    "SELECT COALESCE(MAX(sequence),0)+1 FROM conversation_messages WHERE conversation_id=?",
                    (thread_id,),
                ).fetchone()[0]
            )
            conn.execute(
                """
                INSERT INTO conversation_messages(
                    message_id,turn_id,conversation_id,sequence,role,mode,status,content,
                    process_json,created_at,completed_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    str(uuid.uuid4()), turn_id, thread_id, sequence, role, mode, status,
                    content, "[]", now, now,
                ),
            )
            conn.execute(
                "UPDATE conversations SET last_message_at=?,updated_at=? WHERE thread_id=?",
                (now, now, thread_id),
            )

    def update_conversation_summary(
        self, thread_id: str, summary: dict[str, Any], through_sequence: int
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE conversations SET summary_json=?,summary_through_sequence=?,updated_at=?
                WHERE thread_id=?
                """,
                (json.dumps(summary, ensure_ascii=False), through_sequence, _now(), thread_id),
            )

    def list_paper_children(self, paper_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM papers WHERE parent_paper_id=? ORDER BY updated_at DESC",
                (paper_id,),
            ).fetchall()
        return [self._decode_paper(row) for row in rows]

    def affected_conversations_for_paper(self, paper_id: str) -> list[dict[str, Any]]:
        paper = self.get_paper(paper_id)
        root_id = (paper or {}).get("parent_paper_id") or paper_id
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT c.* FROM conversations c
                LEFT JOIN conversation_papers cp ON cp.conversation_id=c.thread_id
                LEFT JOIN conversation_messages m ON m.conversation_id=c.thread_id
                LEFT JOIN message_citations mc ON mc.message_id=m.message_id
                WHERE cp.paper_id IN (?,?) OR mc.paper_id=?
                ORDER BY COALESCE(c.last_message_at,c.updated_at) DESC
                """,
                (paper_id, root_id, paper_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def mark_paper_conversations_read_only(self, paper_id: str, title: str) -> int:
        affected = self.affected_conversations_for_paper(paper_id)
        if not affected:
            return 0
        now = _now()
        reason = f"论文已删除，历史会话仅供查看：{title} ({paper_id})"
        with self.connect() as conn:
            conn.executemany(
                """
                UPDATE conversations SET archived_at=COALESCE(archived_at,?),
                    read_only_reason=?,updated_at=? WHERE thread_id=?
                """,
                [(now, reason, now, row["thread_id"]) for row in affected],
            )
        return len(affected)

    def delete_paper_metadata(self, paper_id: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM papers WHERE paper_id=?", (paper_id,))
            conn.execute("DELETE FROM paper_comparisons")

    def save_retrieval_trace(
        self,
        *,
        trace_id: str,
        thread_id: str | None,
        question: str,
        standalone_query: str,
        paper_ids: list[str],
        paper_versions: dict[str, int],
        retrieved_chunk_ids: list[str],
        used_chunk_ids: list[str],
        prompt_version: str,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO retrieval_traces VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    trace_id,
                    thread_id,
                    question,
                    standalone_query,
                    json.dumps(paper_ids),
                    json.dumps(paper_versions),
                    json.dumps(retrieved_chunk_ids),
                    json.dumps(used_chunk_ids),
                    prompt_version,
                    _now(),
                ),
            )

    def get_retrieval_trace(self, trace_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM retrieval_traces WHERE trace_id=?", (trace_id,)
            ).fetchone()
        if not row:
            return None
        result = dict(row)
        for key in (
            "paper_ids_json",
            "paper_versions_json",
            "retrieved_chunk_ids_json",
            "used_chunk_ids_json",
        ):
            result[key.removesuffix("_json")] = json.loads(result.pop(key))
        return result

    def list_retrieval_traces_for_thread(self, thread_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM retrieval_traces WHERE thread_id=? ORDER BY created_at",
                (thread_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def arxiv_id_exists(self, arxiv_id: str) -> bool:
        with self.connect() as conn:
            row = conn.execute("SELECT 1 FROM papers WHERE arxiv_id=?", (arxiv_id,)).fetchone()
        return row is not None

    @staticmethod
    def _decode_paper(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["authors"] = json.loads(result.pop("authors_json"))
        return result
