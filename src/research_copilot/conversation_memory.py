from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from research_copilot.errors import ResearchCopilotError
from research_copilot.models import (
    ConversationMode,
    ConversationScope,
    ConversationScopeType,
    ConversationSummary,
    StandaloneQuestion,
)
from research_copilot.storage import SQLiteRepository

logger = logging.getLogger(__name__)

REFERENCE_RE = re.compile(
    r"(^|[，。；、\s])(它|它的|这个|该|上述|前者|后者|其|那|这些|刚才|上面)"
    r"|再(详细|展开|具体)|继续(说|分析)|那.{0,8}(呢|如何)|与前者相比",
    re.IGNORECASE,
)

CONTEXT_PROMPT = """你负责把论文对话中的省略式追问改写为可独立检索的问题。
只解决指代和省略，不回答论文事实，不添加历史中没有的技术结论。
输出符合 StandaloneQuestion schema 的对象。"""

SUMMARY_PROMPT = """你负责压缩论文助手的会话记忆。只记录用户目标、讨论主题、术语指代、
明确偏好和未解决问题。不要把助手回答中的论文事实写成可信知识，也不要保存 API Key、路径密钥
或隐式思维过程。输出符合 ConversationSummary schema 的对象。"""


class ConversationMemoryService:
    """Canonical visible chat history shared by quick, agent and deep modes."""

    def __init__(
        self,
        repository: SQLiteRepository,
        chat_model: BaseChatModel,
        checkpointer: Any,
    ):
        self.repository = repository
        self.chat_model = chat_model
        self.checkpointer = checkpointer

    def resolve_scope(self, selected_paper_ids: Iterable[str]) -> ConversationScope:
        selected = list(dict.fromkeys(selected_paper_ids))
        if not selected:
            return ConversationScope(
                scope_type=ConversationScopeType.GENERAL,
                scope_key="general",
            )
        roots: dict[str, dict[str, Any]] = {}
        for paper_id in selected:
            paper = self.repository.get_paper(paper_id)
            if paper is None:
                raise ResearchCopilotError(f"论文不存在：{paper_id}")
            root_id = paper.get("parent_paper_id") or paper_id
            root = self.repository.get_paper(root_id) or paper
            roots[root_id] = root
        root_ids = sorted(roots)
        if len(root_ids) == 1:
            scope_type = ConversationScopeType.PAPER_FAMILY
            scope_key = f"paper:{root_ids[0]}"
        else:
            scope_type = ConversationScopeType.PAPER_SET
            scope_key = "papers:" + "|".join(root_ids)
        ready = self.repository.list_papers(status="ready")
        effective = sorted(
            item["paper_id"]
            for item in ready
            if item["paper_id"] in roots or item.get("parent_paper_id") in roots
        )
        snapshots = [
            {
                "paper_id": paper_id,
                "title": roots[paper_id]["title"],
                "active_version": int(roots[paper_id].get("active_version", 0)),
            }
            for paper_id in root_ids
        ]
        return ConversationScope(
            scope_type=scope_type,
            scope_key=scope_key,
            root_paper_ids=root_ids,
            effective_paper_ids=effective,
            paper_snapshots=snapshots,
        )

    def create_conversation(
        self, scope: ConversationScope, *, title: str = "新会话"
    ) -> dict[str, Any]:
        thread_id = str(uuid.uuid4())
        return self.repository.create_conversation(
            thread_id,
            title=title,
            scope_type=scope.scope_type.value,
            scope_key=scope.scope_key,
            papers=scope.paper_snapshots,
        )

    def get_or_create_conversation(
        self, scope: ConversationScope, *, first_question: str | None = None
    ) -> dict[str, Any]:
        conversations = self.repository.list_conversations(scope_key=scope.scope_key)
        if conversations:
            return self.repository.get_conversation(conversations[0]["thread_id"]) or conversations[0]
        title = self.title_from_question(first_question) if first_question else "新会话"
        return self.create_conversation(scope, title=title)

    @staticmethod
    def title_from_question(question: str | None) -> str:
        clean = re.sub(r"\s+", " ", (question or "").strip())
        if not clean:
            return "新会话"
        return clean[:48] + ("…" if len(clean) > 48 else "")

    def list_conversations(
        self, scope: ConversationScope, *, include_archived: bool = False
    ) -> list[dict[str, Any]]:
        return self.repository.list_conversations(
            scope_key=scope.scope_key, include_archived=include_archived
        )

    def related_single_paper_conversations(
        self, scope: ConversationScope
    ) -> list[dict[str, Any]]:
        if scope.scope_type != ConversationScopeType.PAPER_SET:
            return []
        rows: list[dict[str, Any]] = []
        for paper_id in scope.root_paper_ids:
            rows.extend(
                self.repository.list_conversations(scope_key=f"paper:{paper_id}")
            )
        return sorted(rows, key=lambda item: item.get("last_message_at") or "", reverse=True)

    def resolve_question(self, thread_id: str, question: str) -> str:
        question = question.strip()
        if not question or not REFERENCE_RE.search(question):
            return question
        conversation = self.repository.get_conversation(thread_id)
        messages = self.repository.get_conversation_messages(thread_id)
        completed = [item for item in messages if item["status"] == "completed"][-16:]
        if not completed:
            return question
        history = "\n".join(
            f"[{item['turn_id']}] {item['role']}: {item['content']}" for item in completed
        )
        summary = json.dumps(
            (conversation or {}).get("summary") or {}, ensure_ascii=False
        )
        try:
            result = self.chat_model.with_structured_output(
                StandaloneQuestion, method="json_mode"
            ).invoke(
                [
                    SystemMessage(content=CONTEXT_PROMPT),
                    HumanMessage(
                        content=(
                            f"会话摘要：{summary}\n\n最近对话：\n{history}\n\n"
                            f"当前追问：{question}"
                        )
                    ),
                ]
            )
            result = StandaloneQuestion.model_validate(result)
            if result.needs_context and result.standalone_question.strip():
                return result.standalone_question.strip()
        except Exception as exc:  # noqa: BLE001 - deterministic fallback keeps chat usable
            logger.info("会话指代改写失败，使用确定性降级：%s", type(exc).__name__)
        last_user = next(
            (item["content"] for item in reversed(completed) if item["role"] == "user"),
            "",
        )
        return f"基于上一问题“{last_user}”，继续回答：{question}" if last_user else question

    def begin_turn(
        self,
        thread_id: str,
        *,
        mode: ConversationMode,
        original_query: str,
        standalone_query: str,
    ) -> dict[str, str]:
        try:
            turn_id, user_message_id, assistant_message_id = (
                self.repository.start_conversation_turn(
                    thread_id,
                    mode=mode.value,
                    original_query=original_query,
                    standalone_query=standalone_query,
                )
            )
        except (KeyError, ValueError) as exc:
            raise ResearchCopilotError(str(exc)) from exc
        return {
            "turn_id": turn_id,
            "user_message_id": user_message_id,
            "assistant_message_id": assistant_message_id,
        }

    def progress(self, assistant_message_id: str, process: list[str], *, interrupted=False) -> None:
        self.repository.update_turn_progress(
            assistant_message_id,
            process,
            status="interrupted" if interrupted else "running",
        )

    def complete_turn(
        self,
        assistant_message_id: str,
        *,
        content: str,
        process: list[str],
        payload: dict[str, Any] | None = None,
    ) -> None:
        if payload is not None:
            payload = {"schema_version": 1, **payload}
        self.repository.finish_conversation_turn(
            assistant_message_id,
            content=content,
            process=process,
            payload=payload,
            retrieval_trace_id=(payload or {}).get("retrieval_trace_id"),
        )
        message = next(
            item
            for item in self.repository.get_conversation_messages_by_ids(
                [assistant_message_id]
            )
        )
        self._maybe_update_summary(message["conversation_id"])

    def fail_turn(
        self,
        assistant_message_id: str,
        *,
        error: str,
        process: list[str],
        status: str = "failed",
    ) -> None:
        self.repository.finish_conversation_turn(
            assistant_message_id,
            content=f"请求失败：{error}" if status == "failed" else "用户已拒绝该操作。",
            process=process,
            status=status,
            error=error,
        )

    def messages(self, thread_id: str) -> list[dict[str, Any]]:
        return self.repository.get_conversation_messages(thread_id)

    def reconcile_stale_turn(self, thread_id: str, *, after_minutes: int = 30) -> bool:
        conversation = self.repository.get_conversation(thread_id)
        if not conversation or not conversation.get("pending_turn_id"):
            return False
        messages = self.repository.get_conversation_messages(thread_id)
        assistant = next(
            (
                item
                for item in reversed(messages)
                if item["turn_id"] == conversation["pending_turn_id"]
                and item["role"] == "assistant"
            ),
            None,
        )
        if assistant is None or assistant["status"] == "interrupted":
            return False
        created = datetime.fromisoformat(assistant["created_at"])
        if created > datetime.now(UTC) - timedelta(minutes=after_minutes):
            return False
        self.fail_turn(
            assistant["message_id"],
            error="应用在上一轮完成前停止；该请求已标记为可重试。",
            process=assistant.get("process", []),
        )
        return True

    def rename(self, thread_id: str, title: str) -> None:
        if not title.strip():
            raise ResearchCopilotError("会话标题不能为空")
        self.repository.rename_conversation(thread_id, title[:80])

    def archive(self, thread_id: str, *, archived: bool = True) -> None:
        self.repository.archive_conversation(thread_id, archived=archived)

    def delete_permanently(self, thread_id: str) -> None:
        self.checkpointer.delete_thread(thread_id)
        self.repository.delete_conversation(thread_id)

    def repair_incomplete_agent_checkpoint(self, thread_id: str) -> bool:
        """Drop only a malformed Agent checkpoint while keeping visible history.

        A model response containing tool calls must be followed by one
        ``ToolMessage`` for every call. If a tool or middleware failure happens
        between those two writes, LangGraph can retain an invalid checkpoint;
        the next model request is then rejected before the Agent can continue.
        SQLite conversation messages remain the user-facing source of truth,
        so clearing this execution-only checkpoint is safe and recoverable.
        """

        checkpoint = self.checkpointer.get_tuple(
            {"configurable": {"thread_id": thread_id}}
        )
        if checkpoint is None:
            return False
        messages = checkpoint.checkpoint.get("channel_values", {}).get("messages", [])
        if not self._has_unpaired_tool_call(messages):
            return False

        visible_messages = self.repository.get_conversation_messages(thread_id)
        interrupted = next(
            (
                item
                for item in reversed(visible_messages)
                if item["role"] == "assistant" and item["status"] == "interrupted"
            ),
            None,
        )
        if interrupted is not None:
            # A Human-in-the-loop pause intentionally has no completed tool
            # message yet and must survive a page refresh or application restart.
            return False

        self.checkpointer.delete_thread(thread_id)
        logger.warning(
            "cleared malformed Agent checkpoint thread_id=%s; visible SQLite history preserved",
            thread_id,
        )
        return True

    @staticmethod
    def _has_unpaired_tool_call(messages: Iterable[Any]) -> bool:
        pending_ids: set[str] = set()
        for message in messages:
            if isinstance(message, AIMessage):
                call_ids = {
                    str(call.get("id"))
                    for call in message.tool_calls
                    if call.get("id")
                }
                if call_ids:
                    # A second tool-calling response before the first response
                    # was completed is invalid as well.
                    if pending_ids:
                        return True
                    pending_ids.update(call_ids)
            elif isinstance(message, ToolMessage):
                if message.tool_call_id:
                    pending_ids.discard(str(message.tool_call_id))
            elif isinstance(message, HumanMessage) and pending_ids:
                return True
        return bool(pending_ids)

    def _maybe_update_summary(self, thread_id: str) -> None:
        messages = self.repository.get_conversation_messages(thread_id)
        assistant_turns = [
            item for item in messages
            if item["role"] == "assistant" and item["status"] == "completed"
        ]
        conversation = self.repository.get_conversation(thread_id) or {}
        through = int(conversation.get("summary_through_sequence") or 0)
        if len(assistant_turns) < 12:
            return
        unsummarized_turns = sum(item["sequence"] > through for item in assistant_turns)
        if through and unsummarized_turns < 6:
            return
        cutoff = max(item["sequence"] for item in messages[:-16]) if len(messages) > 16 else 0
        if cutoff <= through:
            return
        source = [item for item in messages if through < item["sequence"] <= cutoff]
        transcript = "\n".join(f"{item['role']}: {item['content']}" for item in source)
        try:
            summary = self.chat_model.with_structured_output(
                ConversationSummary, method="json_mode"
            ).invoke(
                [
                    SystemMessage(content=SUMMARY_PROMPT),
                    HumanMessage(
                        content=(
                            "已有摘要："
                            + json.dumps(conversation.get("summary") or {}, ensure_ascii=False)
                            + "\n待合并对话：\n"
                            + transcript
                        )
                    ),
                ]
            )
            summary = ConversationSummary.model_validate(summary)
        except Exception:  # noqa: BLE001 - summaries are optional optimization
            return
        self.repository.update_conversation_summary(
            thread_id, summary.model_dump(mode="json"), cutoff
        )

    def import_legacy_checkpoints(self) -> None:
        """Best-effort one-time recovery for pre-memory standard Agent threads."""
        for conversation in self.repository.list_conversations(
            include_archived=True, require_messages=False
        ):
            thread_id = conversation["thread_id"]
            existing = self.repository.get_conversation_messages(thread_id)
            if existing:
                self._refresh_legacy_title(conversation, existing)
                continue
            checkpoint = self.checkpointer.get_tuple(
                {"configurable": {"thread_id": thread_id}}
            )
            messages = (
                checkpoint.checkpoint.get("channel_values", {}).get("messages", [])
                if checkpoint is not None
                else []
            )
            for message in messages:
                if isinstance(message, HumanMessage) and message.content:
                    content = re.sub(
                        r"\n\n本轮只使用这些 paper_id：.*$",
                        "",
                        str(message.content),
                        flags=re.DOTALL,
                    )
                    self.repository.import_conversation_message(
                        thread_id, role="user", content=content
                    )
                elif (
                    isinstance(message, AIMessage)
                    and message.content
                    and not message.tool_calls
                ):
                    self.repository.import_conversation_message(
                        thread_id, role="assistant", content=str(message.content)
                    )
            if not self.repository.get_conversation_messages(thread_id):
                for trace in self.repository.list_retrieval_traces_for_thread(thread_id):
                    turn_id = str(uuid.uuid4())
                    self.repository.import_conversation_message(
                        thread_id,
                        role="user",
                        content=trace["question"],
                        mode="quick",
                        turn_id=turn_id,
                    )
                    self.repository.import_conversation_message(
                        thread_id,
                        role="assistant",
                        content=(
                            "旧版本只保存了问题和 retrieval trace，回答正文无法恢复。"
                            f" trace_id：{trace['trace_id']}"
                        ),
                        mode="quick",
                        status="legacy_incomplete",
                        turn_id=turn_id,
                    )
            self._refresh_legacy_title(
                conversation, self.repository.get_conversation_messages(thread_id)
            )

    def _refresh_legacy_title(
        self, conversation: dict[str, Any], messages: list[dict[str, Any]]
    ) -> None:
        if conversation["title"] not in {"Research Copilot conversation", "新会话"}:
            return
        first_question = next(
            (item["content"] for item in messages if item["role"] == "user"), None
        )
        if first_question:
            self.repository.rename_conversation(
                conversation["thread_id"], self.title_from_question(first_question)
            )
