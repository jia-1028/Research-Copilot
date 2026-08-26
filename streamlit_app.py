from __future__ import annotations

import html
import json
import uuid
from contextlib import suppress
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from research_copilot.agent import build_agent
from research_copilot.background_tasks import (
    TASK_COMPARE,
    TASK_SUMMARIZE,
)
from research_copilot.errors import ResearchCopilotError
from research_copilot.evaluation import validate_eval_dataset
from research_copilot.exports import (
    comparison_markdown,
    conversation_json,
    conversation_markdown,
)
from research_copilot.models import (
    ConversationMode,
    DocumentRole,
    PaperComparison,
    PaperProfile,
)
from research_copilot.presentation import answer_text
from research_copilot.services import build_services

st.set_page_config(page_title="Research Copilot", page_icon="📚", layout="wide")

QUICK_MODE = "快速论文问答"
STANDARD_MODE = "标准模型（Agent）"
DEEP_MODE = "多 Agent 深度分析"
RUNTIME_SCHEMA_VERSION = "service-container-v14-background-task-repository"
MAX_EVIDENCE_EXPANSIONS = 2


def _inject_app_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --rc-ink: #172033;
            --rc-muted: #64748b;
            --rc-line: rgba(148, 163, 184, 0.24);
            --rc-accent: #4667e8;
            --rc-accent-soft: #eef2ff;
            --rc-surface: rgba(255, 255, 255, 0.88);
        }

        .stApp {
            color: var(--rc-ink);
            background:
                radial-gradient(circle at 82% 2%, rgba(99, 102, 241, 0.10), transparent 26rem),
                radial-gradient(circle at 18% 18%, rgba(56, 189, 248, 0.07), transparent 24rem),
                linear-gradient(180deg, #f7f9fe 0%, #fbfcff 42%, #f8fafc 100%);
        }

        [data-testid="stMainBlockContainer"] {
            max-width: 1120px;
            padding-top: 2rem;
            padding-bottom: 10rem;
        }

        [data-testid="stHeader"] {
            background: rgba(247, 249, 254, 0.78);
            backdrop-filter: blur(14px);
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid var(--rc-line);
            background:
                radial-gradient(circle at 12% 4%, rgba(99, 102, 241, 0.11), transparent 15rem),
                linear-gradient(180deg, #f8faff 0%, #f4f7fc 100%);
        }

        [data-testid="stSidebarContent"] {
            padding-bottom: 2rem;
        }

        [data-testid="stSidebarNavLink"] {
            border-radius: 12px;
            margin: 2px 8px;
            transition: background-color 120ms ease, transform 120ms ease;
        }

        [data-testid="stSidebarNavLink"]:hover {
            background: rgba(70, 103, 232, 0.08);
            transform: translateX(2px);
        }

        [data-testid="stSidebarNavLink"][aria-current="page"] {
            background: rgba(70, 103, 232, 0.12);
            color: #314fc5;
            font-weight: 650;
        }

        .rc-sidebar-brand {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin: 0.2rem 0 0.8rem;
            padding: 0.75rem 0.8rem;
            border: 1px solid rgba(99, 102, 241, 0.14);
            border-radius: 15px;
            background: rgba(255, 255, 255, 0.62);
            box-shadow: 0 8px 25px rgba(44, 55, 96, 0.05);
        }

        .rc-sidebar-logo {
            display: grid;
            place-items: center;
            width: 2.15rem;
            height: 2.15rem;
            border-radius: 11px;
            color: white;
            background: linear-gradient(135deg, #536eea, #7c5ce7);
            box-shadow: 0 7px 16px rgba(83, 110, 234, 0.25);
            font-size: 1.05rem;
        }

        .rc-sidebar-brand strong {
            display: block;
            color: #1e293b;
            font-size: 0.94rem;
            line-height: 1.2;
        }

        .rc-sidebar-brand span {
            color: #77839a;
            font-size: 0.73rem;
        }

        [data-testid="stElementContainer"]:has(.rc-conversation-header) {
            position: sticky;
            top: 3.65rem;
            z-index: 90;
            margin-bottom: 0.85rem;
        }

        .rc-conversation-header {
            display: grid;
            grid-template-columns: minmax(0, 0.9fr) 1px minmax(0, 1.35fr);
            align-items: center;
            gap: 0.9rem;
            min-height: 3.7rem;
            padding: 0.62rem 0.9rem;
            border: 1px solid rgba(99, 102, 241, 0.16);
            border-radius: 15px;
            background: rgba(255, 255, 255, 0.90);
            box-shadow: 0 10px 28px rgba(36, 48, 86, 0.10);
            backdrop-filter: blur(18px);
        }

        .rc-context-item {
            min-width: 0;
        }

        .rc-context-label {
            display: block;
            margin-bottom: 0.08rem;
            color: #818da3;
            font-size: 0.66rem;
            font-weight: 700;
            letter-spacing: 0.04em;
        }

        .rc-context-value {
            display: block;
            overflow: hidden;
            color: #26334d;
            font-size: 0.83rem;
            font-weight: 650;
            line-height: 1.35;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .rc-context-paper .rc-context-value {
            color: #465a85;
        }

        .rc-context-divider {
            align-self: stretch;
            width: 1px;
            background: linear-gradient(180deg, transparent, rgba(99, 102, 241, 0.22), transparent);
        }

        [data-testid="stChatMessage"] {
            margin: 0.55rem 0;
            padding: 1.05rem 1.08rem;
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.83);
            box-shadow: 0 9px 30px rgba(39, 51, 89, 0.055);
            backdrop-filter: blur(8px);
        }

        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
            border-color: rgba(99, 102, 241, 0.18);
            background: linear-gradient(135deg, rgba(238, 242, 255, 0.92), rgba(248, 250, 255, 0.92));
        }

        [data-testid="stChatMessageAvatarAssistant"] {
            color: #ffffff;
            background: linear-gradient(135deg, #4f6be6, #7259d8);
            box-shadow: 0 6px 16px rgba(79, 107, 230, 0.22);
        }

        [data-testid="stChatMessageAvatarUser"] {
            color: #334155;
            background: #e6eaf4;
        }

        [data-testid="stExpander"] {
            overflow: hidden;
            border-color: rgba(148, 163, 184, 0.22);
            border-radius: 13px;
            background: rgba(248, 250, 252, 0.62);
        }

        [data-testid="stAlert"] {
            border-radius: 13px;
        }

        [data-testid="stBaseButton-primary"] {
            border: 0;
            border-radius: 11px;
            background: linear-gradient(135deg, #4c68e3, #6d56d8);
            box-shadow: 0 7px 18px rgba(76, 104, 227, 0.18);
        }

        [data-testid="stBaseButton-secondary"] {
            border-color: rgba(148, 163, 184, 0.32);
            border-radius: 11px;
            background: rgba(255, 255, 255, 0.72);
        }

        [data-testid="stBottom"] {
            border-top: 1px solid rgba(148, 163, 184, 0.20);
            background: rgba(248, 250, 253, 0.86);
            box-shadow: 0 -14px 35px rgba(41, 52, 87, 0.07);
            backdrop-filter: blur(16px);
        }

        [data-testid="stBottomBlockContainer"] {
            max-width: 1120px;
            padding-top: 0.75rem;
            padding-bottom: 0.78rem;
        }

        [data-testid="stButtonGroup"] {
            padding: 0.16rem;
            border: 1px solid rgba(148, 163, 184, 0.20);
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.72);
        }

        [data-testid="stChatInput"] {
            border: 1px solid rgba(99, 102, 241, 0.20);
            border-radius: 16px;
            background: #ffffff;
            box-shadow: 0 8px 24px rgba(47, 58, 98, 0.08);
        }

        @media (max-width: 720px) {
            [data-testid="stMainBlockContainer"] {
                padding-top: 1rem;
            }
            .rc-conversation-header {
                grid-template-columns: minmax(0, 0.85fr) 1px minmax(0, 1.15fr);
                gap: 0.65rem;
                padding: 0.55rem 0.7rem;
                border-radius: 13px;
            }
            [data-testid="stChatMessage"] {
                padding: 0.85rem;
                border-radius: 15px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


_inject_app_styles()


@st.cache_resource
def _cached_runtime(schema_version: str):
    if schema_version != RUNTIME_SCHEMA_VERSION:
        raise RuntimeError("运行时缓存版本不匹配")
    services = build_services()
    return services, build_agent(services, enable_hitl=True)


def runtime():
    services, agent = _cached_runtime(RUNTIME_SCHEMA_VERSION)
    required_services = (
        "rag",
        "deep_analysis",
        "library",
        "checkpointer",
        "memory",
        "tasks",
    )
    required_repository_methods = (
        "create_background_task",
        "get_background_task",
        "list_background_tasks",
        "update_background_task",
    )
    runtime_is_stale = not all(hasattr(services, name) for name in required_services)
    runtime_is_stale = runtime_is_stale or not all(
        hasattr(services.repository, name) for name in required_repository_methods
    )
    if runtime_is_stale:
        # A Streamlit hot reload can retain an instance created from an older
        # ServiceContainer class. Clear only the function cache and rebuild it.
        with suppress(Exception):
            services.close()
        _cached_runtime.clear()
        services, agent = _cached_runtime(RUNTIME_SCHEMA_VERSION)
    return services, agent


def ready_papers() -> list[dict]:
    return runtime()[0].repository.list_papers(status="ready")


def _render_comparison_result(comparison: PaperComparison, *, key_prefix: str) -> None:
    for row in comparison.rows:
        st.subheader(row.dimension)
        st.dataframe(
            [
                {
                    "paper_id": paper_id,
                    "value": value.value,
                    "citations": ", ".join(value.citation_ids),
                    "evidence_missing": value.insufficient_evidence,
                }
                for paper_id, value in row.values.items()
            ],
            use_container_width=True,
            hide_index=True,
        )
    if comparison.non_comparable_items:
        st.warning("不可直接比较：\n- " + "\n- ".join(comparison.non_comparable_items))
    export_col1, export_col2 = st.columns(2)
    export_col1.download_button(
        "下载 Markdown 比较报告",
        data=comparison_markdown(comparison),
        file_name="paper-comparison.md",
        mime="text/markdown",
        key=f"{key_prefix}-comparison-md",
        use_container_width=True,
    )
    export_col2.download_button(
        "下载 JSON 数据",
        data=comparison.model_dump_json(indent=2),
        file_name="paper-comparison.json",
        mime="application/json",
        key=f"{key_prefix}-comparison-json",
        use_container_width=True,
    )
    with st.expander("全部证据"):
        for citation in comparison.citations:
            st.markdown(
                f"**[{citation.citation_id}] {citation.paper_title}，PDF 第 {citation.pdf_page} 页**"
            )
            st.caption(citation.chunk_id)
            st.write(citation.evidence_text)


def _render_summary_result(profile: PaperProfile) -> None:
    labels = {
        "research_problem": "研究问题",
        "core_contributions": "核心贡献",
        "method_architecture": "方法与架构",
        "datasets": "数据集",
        "experimental_setup": "实验设置",
        "metrics": "指标",
        "main_results": "主要结果",
        "efficiency": "效率",
        "limitations": "局限",
    }
    for field, label in labels.items():
        value = getattr(profile, field)
        st.markdown(f"**{label}**")
        st.write(value.value)
        if value.citation_ids:
            st.caption("引用：" + "、".join(value.citation_ids))


@st.fragment(run_every=2.0)
def _render_background_task(task_id: str, *, key_prefix: str) -> None:
    services, _ = runtime()
    task = services.repository.get_background_task(task_id)
    if not task:
        st.error("后台任务不存在或已被清理。")
        return
    status_labels = {
        "queued": "排队中",
        "running": "执行中",
        "completed": "已完成",
        "failed": "失败",
        "cancelled": "已取消",
    }
    with st.container(border=True):
        st.markdown(
            f"**{status_labels.get(task['status'], task['status'])}** · "
            f"`{task['task_type']}`"
        )
        st.progress(float(task["progress"]), text=task["current_step"])
        st.caption(f"task_id: {task_id}")
        if task["status"] in {"queued", "running"}:
            if st.button(
                "取消任务",
                key=f"{key_prefix}-cancel-{task_id}",
                disabled=task["cancel_requested"],
            ):
                services.tasks.cancel(task_id)
                st.rerun(scope="fragment")
            return
        if task["status"] == "failed":
            st.error(task["error"] or "后台任务执行失败")
            return
        if task["status"] == "cancelled":
            st.warning("任务已取消；运行中的模型/API 请求可能在当前步骤返回后才真正结束。")
            return
        result = task.get("result") or {}
        if task["task_type"] == TASK_COMPARE:
            _render_comparison_result(
                PaperComparison.model_validate(result), key_prefix=f"{key_prefix}-{task_id}"
            )
        elif task["task_type"] == TASK_SUMMARIZE:
            _render_summary_result(PaperProfile.model_validate(result))
        else:
            st.success(
                f"论文已导入：{result.get('paper_id', '')} · v{result.get('version', '')} · "
                f"{result.get('chunk_count', 0)} chunks"
            )


def render_answer_details(payload: dict) -> None:
    if payload.get("fallback_used"):
        st.info(
            "首选模型暂时不可用，本轮已自动切换至备用模型 "
            f"{payload.get('generation_model') or ''}；PDF 检索、引用和校验规则保持不变。"
        )
    if payload.get("insufficient_evidence"):
        st.warning("证据不足：" + "；".join(payload.get("limitations") or []))
    citations = payload.get("citations") or []
    reports = payload.get("facet_reports") or []
    if reports:
        with st.expander(f"Specialist 子报告（{len(reports)} 个）"):
            for report in reports:
                state = "证据不足" if report.get("insufficient_evidence") else "完成"
                st.markdown(f"### {report['focus']} · {state}")
                st.markdown(report["answer"])
                if report.get("limitations"):
                    st.caption("；".join(report["limitations"]))
    if citations:
        services, _ = runtime()
        with st.expander(f"PDF 出处与原文证据（{len(citations)} 条）"):
            for citation in citations:
                st.markdown(
                    f"**[{citation['citation_id']}] {citation['paper_title']} · "
                    f"PDF 第 {citation['pdf_page']} 页**"
                )
                st.caption(citation["chunk_id"])
                image_path = citation.get("image_path")
                if not image_path:
                    version = services.repository.get_version(
                        citation["paper_id"], int(citation["paper_version"])
                    )
                    if version:
                        candidate = (
                            Path(version["parsed_dir"])
                            / "page_images"
                            / f"page_{int(citation['pdf_page']):04d}.jpg"
                        )
                        image_path = str(candidate) if candidate.is_file() else None
                if image_path and Path(image_path).is_file():
                    with st.popover(f"查看 PDF 第 {citation['pdf_page']} 页"):
                        st.image(
                            image_path,
                            caption=f"[{citation['citation_id']}] · PDF 第 {citation['pdf_page']} 页",
                        )
                st.write(citation["evidence_text"])


def _normalize_history_item(item) -> dict:
    if isinstance(item, dict):
        return item
    role, content = item
    return {"role": role, "content": content, "process": [], "payload": None}


def _record_progress(
    status, process: list[str], message: str, persist_callback=None
) -> None:
    if process and process[-1] == message:
        return
    process.append(message)
    status.write(message)
    if persist_callback:
        persist_callback(process)


def _agent_update_messages(node: str, update) -> list[str]:
    if node == "__interrupt__":
        return ["Agent 已暂停，等待人工确认高成本操作。"]
    if not isinstance(update, dict):
        return []
    messages = update.get("messages") or []
    if not isinstance(messages, list):
        messages = [messages]
    lines: list[str] = []
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            names = "、".join(call["name"] for call in message.tool_calls)
            lines.append(f"模型已选择高层工具：{names}")
        elif isinstance(message, ToolMessage):
            state = "失败" if message.status == "error" else "完成"
            lines.append(f"工具 {message.name or 'unknown'} 执行{state}。")
        elif isinstance(message, AIMessage) and message.content:
            lines.append("模型已生成最终回复。")
    if not lines and node not in {"model", "tools"}:
        lines.append(f"工作流步骤完成：{node}")
    return lines


def _run_agent_stream(
    agent, input_value, config: dict, status, process: list[str], persist_callback=None
) -> dict:
    for stream_kind, update in agent.stream(
        input_value,
        config=config,
        stream_mode=["updates", "custom"],
    ):
        if stream_kind == "custom":
            if (
                isinstance(update, dict)
                and update.get("type") == "tool_progress"
                and update.get("message")
            ):
                _record_progress(
                    status,
                    process,
                    f"工具 {update.get('tool', 'unknown')}：{update['message']}",
                    persist_callback,
                )
            continue
        if not isinstance(update, dict):
            continue
        for node, value in update.items():
            for message in _agent_update_messages(node, value):
                _record_progress(status, process, message, persist_callback)
    snapshot = agent.get_state(config)
    result = dict(snapshot.values)
    if snapshot.interrupts:
        result["__interrupt__"] = list(snapshot.interrupts)
    return result


def _requires_standard_agent(prompt: str) -> bool:
    agent_intents = ("导入", "上传", "下载", "arxiv", "论文库", "导出", "删除", "重建")
    lowered = prompt.lower()
    return any(keyword in lowered for keyword in agent_intents)


def _queue_evidence_expansion(request: dict) -> None:
    st.session_state.evidence_expansion_request = request


def _render_conversation_header(services, selected_ids, current) -> None:
    current_title = current["title"] if current else "新会话"

    paper_titles: list[str] = []
    for paper_id in selected_ids:
        paper = services.repository.get_paper(paper_id)
        if paper:
            paper_titles.append(paper["title"])
    if not paper_titles:
        paper_context = "尚未选择论文"
    elif len(paper_titles) <= 2:
        paper_context = " · ".join(paper_titles)
    else:
        paper_context = " · ".join(paper_titles[:2]) + f" · 另 {len(paper_titles) - 2} 篇"

    st.markdown(
        f"""
        <section class="rc-conversation-header" aria-label="当前会话信息">
          <div class="rc-context-item">
            <span class="rc-context-label">当前会话</span>
            <strong class="rc-context-value" title="{html.escape(current_title)}">{html.escape(current_title)}</strong>
          </div>
          <span class="rc-context-divider" aria-hidden="true"></span>
          <div class="rc-context-item rc-context-paper">
            <span class="rc-context-label">当前论文</span>
            <strong class="rc-context-value" title="{html.escape(paper_context)}">{html.escape(paper_context)}</strong>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_chat_sidebar(services):
    papers = ready_papers()
    options = {
        f"{item['title']} ({item['paper_id']})": item["paper_id"] for item in papers
    }
    with st.sidebar:
        st.markdown(
            """
            <div class="rc-sidebar-brand">
              <div class="rc-sidebar-logo">R</div>
              <div><strong>Research Copilot</strong><span>论文阅读与证据问答</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.subheader("研究工作区")
        st.caption(
            f"🤖 {services.settings.chat_provider}/{services.settings.chat_model}"
        )
        selected_labels = st.multiselect(
            "论文范围",
            list(options),
            key="chat_papers",
            placeholder="选择一篇或多篇论文",
        )
        selected_ids = [options[label] for label in selected_labels]
        scope = services.memory.resolve_scope(selected_ids)
        if scope.effective_paper_ids:
            st.caption(
                f"{scope.scope_type.value} · 检索 {len(scope.effective_paper_ids)} 份主文/补充材料"
            )
        else:
            st.caption("通用工作区 · 尚未选择论文")

        st.subheader("会话")
        include_archived = st.toggle(
            "显示已归档", value=False, key="show_archived_chats"
        )
        conversations = services.repository.list_conversations(
            scope_key=scope.scope_key,
            include_archived=include_archived,
            require_messages=False,
        )
        if st.session_state.get("conversation_scope_key") != scope.scope_key:
            st.session_state.conversation_scope_key = scope.scope_key
            st.session_state.thread_id = (
                conversations[0]["thread_id"] if conversations else None
            )
        conversation_ids = [item["thread_id"] for item in conversations]
        if st.session_state.get("thread_id") not in conversation_ids:
            st.session_state.thread_id = conversation_ids[0] if conversations else None

        if conversations:
            labels_by_id = {
                item["thread_id"]: (
                    ("[已归档] " if item.get("archived_at") else "") + item["title"]
                )
                for item in conversations
            }
            selected_thread = st.selectbox(
                "当前会话",
                conversation_ids,
                index=conversation_ids.index(st.session_state.thread_id),
                format_func=lambda value: labels_by_id[value],
                key=f"conversation-picker-{scope.scope_key}",
                label_visibility="collapsed",
            )
            st.session_state.thread_id = selected_thread
        else:
            st.caption("这个工作区还没有会话，首次提问时自动创建。")

        if st.button("＋ 新建会话", use_container_width=True, type="primary"):
            created = services.memory.create_conversation(scope)
            st.session_state.thread_id = created["thread_id"]
            st.rerun()

        current = (
            services.repository.get_conversation(st.session_state.thread_id)
            if st.session_state.get("thread_id")
            else None
        )
        if current and services.memory.reconcile_stale_turn(current["thread_id"]):
            st.info("检测到上次异常中断的请求，已保留问题并标记为可重试。")
            current = services.repository.get_conversation(current["thread_id"])

        if current:
            with st.expander("会话管理"):
                rename_value = st.text_input(
                    "会话标题",
                    value=current["title"],
                    key=f"rename-{current['thread_id']}",
                )
                if st.button(
                    "保存标题",
                    key=f"save-title-{current['thread_id']}",
                    use_container_width=True,
                ):
                    services.memory.rename(current["thread_id"], rename_value)
                    st.rerun()
                export_messages = services.memory.messages(current["thread_id"])
                export_col1, export_col2 = st.columns(2)
                export_col1.download_button(
                    "导出 Markdown",
                    data=conversation_markdown(current, export_messages),
                    file_name=f"conversation-{current['thread_id'][:8]}.md",
                    mime="text/markdown",
                    key=f"export-md-{current['thread_id']}",
                    use_container_width=True,
                )
                export_col2.download_button(
                    "导出 JSON",
                    data=conversation_json(current, export_messages),
                    file_name=f"conversation-{current['thread_id'][:8]}.json",
                    mime="application/json",
                    key=f"export-json-{current['thread_id']}",
                    use_container_width=True,
                )
                archive_col, delete_col = st.columns(2)
                if archive_col.button(
                    "取消归档" if current.get("archived_at") else "归档",
                    key=f"archive-{current['thread_id']}",
                    use_container_width=True,
                ):
                    services.memory.archive(
                        current["thread_id"],
                        archived=not bool(current.get("archived_at")),
                    )
                    st.session_state.thread_id = None
                    st.rerun()
                if delete_col.button(
                    "永久删除",
                    key=f"delete-{current['thread_id']}",
                    type="secondary",
                    use_container_width=True,
                ):
                    st.session_state.confirm_delete_conversation = current["thread_id"]
                if current.get("read_only_reason"):
                    st.warning(current["read_only_reason"])
                for snapshot in current.get("paper_snapshots", []):
                    live = services.repository.get_paper(snapshot["paper_id"])
                    if live and int(live["active_version"]) != int(
                        snapshot["paper_version_snapshot"]
                    ):
                        st.info(
                            f"{snapshot['paper_title_snapshot']} 的历史会话起始于 "
                            f"v{snapshot['paper_version_snapshot']}；新问题使用 "
                            f"active v{live['active_version']}。"
                        )

        if st.session_state.get("confirm_delete_conversation") and current:
            with st.container(border=True):
                st.error("永久删除会清除消息、引用和 Agent Checkpoint，无法恢复。")
                confirmation = st.text_input(
                    "输入会话标题确认", key="conversation-delete-confirmation"
                )
                confirm_col, cancel_col = st.columns(2)
                if confirm_col.button(
                    "确认删除",
                    type="primary",
                    disabled=confirmation != current["title"],
                    use_container_width=True,
                ):
                    services.memory.delete_permanently(current["thread_id"])
                    st.session_state.pop("confirm_delete_conversation", None)
                    st.session_state.thread_id = None
                    st.rerun()
                if cancel_col.button("取消", use_container_width=True):
                    st.session_state.pop("confirm_delete_conversation", None)
                    st.rerun()

        related = services.memory.related_single_paper_conversations(scope)
        if related:
            with st.expander(f"相关单论文会话（{len(related)}）"):
                st.caption("这些会话不会自动注入当前多论文上下文。")
                for item in related:
                    st.write(item["title"])
                    root_id = item["scope_key"].removeprefix("paper:")
                    root_label = next(
                        (label for label, value in options.items() if value == root_id),
                        None,
                    )
                    if root_label and st.button(
                        "打开",
                        key=f"open-related-{item['thread_id']}",
                        use_container_width=True,
                    ):
                        st.session_state.chat_papers = [root_label]
                        st.session_state.thread_id = item["thread_id"]
                        st.session_state.conversation_scope_key = item["scope_key"]
                        st.rerun()
    return selected_ids, scope, current


def chat_page() -> None:
    services, agent = runtime()
    selected_ids, scope, current = _render_chat_sidebar(services)
    current_title = current["title"] if current else "新会话"
    _render_conversation_header(services, selected_ids, current)

    if (
        current
        and st.session_state.get("chat_history")
        and not services.memory.messages(current["thread_id"])
        and not st.session_state.get("legacy_session_history_imported")
    ):
        for raw_item in st.session_state.chat_history:
            item = _normalize_history_item(raw_item)
            services.repository.import_conversation_message(
                current["thread_id"],
                role=item["role"],
                content=item["content"],
                mode="standard_agent",
            )
        st.session_state.legacy_session_history_imported = True
    st.session_state.pop("chat_history", None)

    history = services.memory.messages(current["thread_id"]) if current else []
    conversation_busy = any(
        item["role"] == "assistant" and item["status"] in {"pending", "running"}
        for item in history
    )
    for item in history:
        with st.chat_message(item["role"]):
            if item["role"] == "assistant":
                st.caption(f"{item['mode']} · {item['status']} · {item['created_at']}")
            if item.get("process"):
                with st.expander("查看执行过程"):
                    st.caption("仅展示可观测的检索与工具状态，不展示模型的隐式思维链。")
                    for step in item["process"]:
                        st.write(step)
            if item["content"]:
                st.markdown(answer_text(item["content"], item.get("payload")))
            elif item["status"] in {"pending", "running"}:
                st.info("请求正在处理中……")
            elif item["status"] == "interrupted":
                st.warning("Agent 已暂停，等待人工确认。")
            if item.get("payload"):
                render_answer_details(item["payload"])
                expansion_attempt = int(
                    item["payload"].get("evidence_expansion_attempt", 0)
                )
                can_expand = (
                    item["role"] == "assistant"
                    and item["status"] == "completed"
                    and item["payload"].get("insufficient_evidence")
                    and item["payload"].get("limitations")
                    and item["payload"].get("retrieval_trace_id")
                    and scope.effective_paper_ids
                    and not (current or {}).get("read_only_reason")
                    and not conversation_busy
                )
                if can_expand and expansion_attempt < MAX_EVIDENCE_EXPANSIONS:
                    label = (
                        "🔎 补充检索证据并重新回答"
                        if expansion_attempt == 0
                        else "🔎 再次扩大证据检索"
                    )
                    request = {
                            "original_query": item.get("original_query") or item["content"],
                            "standalone_query": item.get("standalone_query")
                            or item.get("original_query")
                            or item["content"],
                            "limitations": item["payload"]["limitations"],
                            "previous_trace_id": item["payload"]["retrieval_trace_id"],
                            "attempt": expansion_attempt + 1,
                    }
                    st.button(
                        label,
                        key=f"expand-evidence-{item['message_id']}",
                        on_click=_queue_evidence_expansion,
                        args=(request,),
                    )
                elif can_expand:
                    st.caption("已达到两次补充检索上限；当前缺口可能不在已导入的 PDF 中。")
            if (
                item["role"] == "assistant"
                and item["status"] == "failed"
                and item.get("original_query")
                and not (current or {}).get("read_only_reason")
                and st.button("重试本轮", key=f"retry-{item['message_id']}")
            ):
                st.session_state.retry_prompt = item["original_query"]

    interrupted = next(
        (item for item in reversed(history) if item["role"] == "assistant" and item["status"] == "interrupted"),
        None,
    )
    if interrupted and current:
        config = {"configurable": {"thread_id": current["thread_id"]}}
        snapshot = agent.get_state(config)
        interrupt = snapshot.interrupts[0] if snapshot.interrupts else None
        st.warning("Agent 请求执行需要确认的论文导入操作。")
        if interrupted.get("process"):
            with st.expander("查看确认前的执行过程"):
                st.caption("仅展示可观测的检索与工具状态，不展示模型的隐式思维链。")
                for step in interrupted["process"]:
                    st.write(step)
        if interrupt:
            st.json(interrupt.value)
        approve_col, reject_col = st.columns(2)
        if approve_col.button("确认执行", type="primary", disabled=interrupt is None):
            process = interrupted.get("process", [])
            with (
                st.chat_message("assistant"),
                st.status("正在继续已确认的操作……", expanded=True) as status,
            ):
                result = _run_agent_stream(
                    agent,
                    Command(resume={"decisions": [{"type": "approve"}]}),
                    config,
                    status,
                    process,
                    lambda value: services.memory.progress(interrupted["message_id"], value),
                )
                status.update(label="已执行确认操作", state="complete", expanded=False)
            services.memory.complete_turn(
                interrupted["message_id"],
                content=_agent_result_content(result),
                process=process,
                payload=_agent_result_payload(result),
            )
            st.rerun()
        if reject_col.button("拒绝", disabled=interrupt is None):
            agent.invoke(
                Command(
                    resume={
                        "decisions": [
                            {"type": "reject", "message": "用户在界面中拒绝了论文导入"}
                        ]
                    }
                ),
                config=config,
            )
            services.memory.fail_turn(
                interrupted["message_id"],
                error="用户在界面中拒绝了论文导入",
                process=interrupted.get("process", []),
                status="rejected",
            )
            st.rerun()

    composer = st.bottom.container(key="chat-composer")
    mode_col, context_col = composer.columns(
        [3, 2], vertical_alignment="center", gap="small"
    )
    qa_mode = mode_col.segmented_control(
        "回答模式",
        [QUICK_MODE, STANDARD_MODE, DEEP_MODE],
        default=QUICK_MODE,
        required=True,
        format_func={
            QUICK_MODE: "⚡ 快速",
            STANDARD_MODE: "Agent",
            DEEP_MODE: "深度",
        }.get,
        key="chat_qa_mode",
        help=(
            "快速：一次 RAG；Agent：自动选择高层工具；"
            "深度：多个 specialist 协同分析。"
        ),
        disabled=bool(interrupted) or conversation_busy,
        label_visibility="collapsed",
        width="stretch",
    )
    context_col.caption(
        (f"📚 {len(scope.effective_paper_ids)} 份文档" if selected_ids else "📚 通用工作区")
        + f" · {current_title}"
    )
    mode_value = {
        QUICK_MODE: ConversationMode.QUICK,
        STANDARD_MODE: ConversationMode.STANDARD_AGENT,
        DEEP_MODE: ConversationMode.DEEP_ANALYSIS,
    }[qa_mode]
    prompt = composer.chat_input(
        "询问论文方法、实验、结论，或让 Agent 检索/导入 arXiv",
        disabled=(
            bool(current and current.get("read_only_reason"))
            or bool(interrupted)
            or conversation_busy
        ),
        key="chat_prompt",
    )
    evidence_request = None
    if not prompt and st.session_state.get("retry_prompt"):
        prompt = st.session_state.pop("retry_prompt")
    if not prompt and st.session_state.get("evidence_expansion_request"):
        evidence_request = st.session_state.pop("evidence_expansion_request")
        prompt = evidence_request["original_query"]
    if prompt:
        if current is None:
            current = services.memory.create_conversation(
                scope, title=services.memory.title_from_question(prompt)
            )
            st.session_state.thread_id = current["thread_id"]
        elif current["title"] == "新会话":
            services.memory.rename(current["thread_id"], services.memory.title_from_question(prompt))
        standalone_query = (
            evidence_request["standalone_query"]
            if evidence_request
            else services.memory.resolve_question(current["thread_id"], prompt)
        )
        turn = services.memory.begin_turn(
            current["thread_id"],
            mode=(
                ConversationMode.EVIDENCE_EXPANSION if evidence_request else mode_value
            ),
            original_query=prompt,
            standalone_query=standalone_query,
        )
        with st.chat_message("user"):
            st.markdown(prompt)
        config = {
            "configurable": {"thread_id": current["thread_id"]},
            "metadata": {"turn_id": turn["turn_id"]},
        }
        process: list[str] = []
        persist = lambda value: services.memory.progress(turn["assistant_message_id"], value)
        with st.chat_message("assistant"):
            try:
                payload: dict | None = None
                with st.status("正在处理你的问题……", expanded=True) as status:
                    if evidence_request:
                        _record_progress(
                            status,
                            process,
                            f"根据上一轮的 {len(evidence_request['limitations'])} 个证据缺口启动补充检索……",
                            persist,
                        )
                        answer = services.rag.ask(
                            prompt,
                            scope.effective_paper_ids,
                            retrieval_question=standalone_query,
                            thread_id=current["thread_id"],
                            evidence_hints=evidence_request["limitations"],
                            previous_trace_id=evidence_request["previous_trace_id"],
                            evidence_expansion_attempt=evidence_request["attempt"],
                            progress_callback=lambda message: _record_progress(
                                status, process, message, persist
                            ),
                        )
                        payload = answer.model_dump(mode="json")
                        status.update(
                            label="补充证据检索与重新回答完成",
                            state="complete",
                            expanded=False,
                        )
                        content = answer.answer
                    elif qa_mode == QUICK_MODE:
                        if not selected_ids:
                            raise ResearchCopilotError("快速论文问答需要先选择论文范围")
                        if _requires_standard_agent(prompt):
                            raise ResearchCopilotError(
                                "论文导入、下载或论文库操作请切换到“标准模型（Agent）”"
                            )
                        answer = services.rag.ask(
                            prompt,
                            scope.effective_paper_ids,
                            retrieval_question=standalone_query,
                            thread_id=current["thread_id"],
                            progress_callback=lambda message: _record_progress(
                                status, process, message, persist
                            ),
                        )
                        payload = answer.model_dump(mode="json")
                        status.update(
                            label="证据检索与引用校验完成",
                            state="complete",
                            expanded=False,
                        )
                        content = answer.answer
                    elif qa_mode == DEEP_MODE:
                        if not selected_ids:
                            raise ResearchCopilotError("多 Agent 深度分析需要先选择论文范围")
                        if _requires_standard_agent(prompt):
                            raise ResearchCopilotError(
                                "论文导入、下载或论文库操作请切换到“标准模型（Agent）”"
                            )
                        answer = services.deep_analysis.analyze(
                            standalone_query,
                            scope.effective_paper_ids,
                            trace_question=prompt,
                            thread_id=current["thread_id"],
                            progress_callback=lambda message: _record_progress(
                                status, process, message, persist
                            ),
                        )
                        payload = answer.model_dump(mode="json")
                        status.update(
                            label="多 Agent 深度分析完成",
                            state="complete",
                            expanded=False,
                        )
                        content = answer.answer
                    elif (
                        qa_mode == STANDARD_MODE
                        and selected_ids
                        and not _requires_standard_agent(prompt)
                    ):
                        # A selected-paper content question has only one valid
                        # high-level route: ask_papers.  Calling the outer model
                        # merely to rediscover that route adds latency and a
                        # second timeout point without improving the answer.
                        _record_progress(
                            status,
                            process,
                            "已识别为当前论文内容问答，跳过外层 Agent 规划并直接检索证据……",
                            persist,
                        )
                        answer = services.rag.ask(
                            prompt,
                            scope.effective_paper_ids,
                            retrieval_question=standalone_query,
                            thread_id=current["thread_id"],
                            progress_callback=lambda message: _record_progress(
                                status, process, message, persist
                            ),
                        )
                        payload = answer.model_dump(mode="json")
                        status.update(
                            label="标准论文问答与引用校验完成",
                            state="complete",
                            expanded=False,
                        )
                        content = answer.answer
                    else:
                        scoped_prompt = standalone_query
                        if scope.effective_paper_ids:
                            scoped_prompt += (
                                "\n\n本轮只使用这些 paper_id（均已由界面校验为 ready，"
                                "无需调用 list_papers）："
                                f"{scope.effective_paper_ids}"
                            )
                        _record_progress(
                            status,
                            process,
                            "Agent 正在识别意图并选择高层工具……",
                            persist,
                        )
                        try:
                            result = _run_agent_stream(
                                agent,
                                {"messages": [{"role": "user", "content": scoped_prompt}]},
                                config,
                                status,
                                process,
                                persist,
                            )
                        except Exception as exc:
                            if (
                                "insufficient tool messages following tool_calls"
                                not in str(exc).lower()
                                or not services.memory.repair_incomplete_agent_checkpoint(
                                    current["thread_id"]
                                )
                            ):
                                raise
                            _record_progress(
                                status,
                                process,
                                "检测到上一轮遗留的未完成 Tool Call；已保留对话记录并"
                                "重置执行状态，正在自动重试本轮……",
                                persist,
                            )
                            result = _run_agent_stream(
                                agent,
                                {"messages": [{"role": "user", "content": scoped_prompt}]},
                                config,
                                status,
                                process,
                                persist,
                            )
                        if "__interrupt__" in result:
                            status.update(
                                label="等待人工确认",
                                state="running",
                                expanded=False,
                            )
                            services.memory.progress(
                                turn["assistant_message_id"], process, interrupted=True
                            )
                            st.rerun()
                        status.update(
                            label="Agent 工作流完成", state="complete", expanded=False
                        )
                        content = _agent_result_content(result)
                        payload = _agent_result_payload(result)
                st.markdown(content)
                if payload:
                    render_answer_details(payload)
                services.memory.complete_turn(
                    turn["assistant_message_id"],
                    content=content,
                    process=process,
                    payload=payload,
                )
                st.rerun()
            except Exception as exc:  # noqa: BLE001 - UI boundary shows recoverable errors
                message = str(exc)
                if "grammar validation or compilation failed" in message:
                    message = (
                        "百炼结构化输出语法编译超时。请保持“快速论文问答”开启后重试；"
                        "该模式会绕过外层 Agent 的二次模型调用。"
                    )
                if "status" in locals():
                    status.update(label="请求失败", state="error", expanded=False)
                st.error(message)
                services.memory.fail_turn(
                    turn["assistant_message_id"], error=message, process=process
                )
                st.rerun()


def _agent_result_content(result: dict) -> str:
    messages = result.get("messages") or []
    direct_tool = next(
        (
            item
            for item in reversed(messages)
            if isinstance(item, ToolMessage)
            and item.name in {"ask_papers", "import_arxiv_paper"}
            and item.status != "error"
        ),
        None,
    )
    if direct_tool is not None:
        if direct_tool.name == "import_arxiv_paper":
            return str(direct_tool.content)
        artifact = direct_tool.artifact if isinstance(direct_tool.artifact, dict) else None
        return answer_text(direct_tool.content, artifact)
    message = next((item for item in reversed(messages) if isinstance(item, AIMessage)), None)
    if message is None:
        content = "Agent 未返回文本消息。"
    elif isinstance(message.content, str):
        content = answer_text(message.content)
    else:
        content = answer_text(message.content)
    return content


def _agent_result_payload(result: dict) -> dict | None:
    supported = {"ask_papers", "summarize_paper", "compare_papers"}
    for message in reversed(result.get("messages") or []):
        if not isinstance(message, ToolMessage) or message.name not in supported:
            continue
        if isinstance(message.artifact, dict):
            return message.artifact
        if isinstance(message.content, dict):
            return message.content
        if isinstance(message.content, str):
            with suppress(json.JSONDecodeError):
                value = json.loads(message.content)
                if isinstance(value, dict):
                    return value
    return None


def library_page() -> None:
    services, _ = runtime()
    st.title("论文库与导入状态")
    ready_main_papers = [
        item
        for item in services.repository.list_papers(status="ready")
        if item["document_role"] == DocumentRole.MAIN.value
    ]
    upload = st.file_uploader("上传 PDF", type=["pdf"])
    title = st.text_input("论文标题（可留空，将从 PDF metadata/首页内容自动识别）")
    document_role = st.segmented_control(
        "文档类型",
        [DocumentRole.MAIN, DocumentRole.SUPPLEMENTARY],
        default=DocumentRole.MAIN,
        format_func=lambda value: "主论文" if value == DocumentRole.MAIN else "Supplementary",
    )
    parent_paper_id = None
    if document_role == DocumentRole.SUPPLEMENTARY:
        parent_paper_id = st.selectbox(
            "所属主论文",
            [item["paper_id"] for item in ready_main_papers],
            format_func=lambda value: next(
                item["title"] for item in ready_main_papers if item["paper_id"] == value
            ),
            disabled=not ready_main_papers,
            placeholder="先导入对应主论文",
        )
    use_mineru = st.toggle(
        "使用 MinerU 在线解析（实验性；默认关闭）",
        value=services.settings.mineru_enabled,
        help="关闭时直接使用本地 PyMuPDF 提取文本并渲染页面图像。",
    )
    if upload and st.button(
        "加入后台导入队列",
        type="primary",
        disabled=(document_role == DocumentRole.SUPPLEMENTARY and not parent_paper_id),
    ):
        upload_id = f"{uuid.uuid4().hex}-{Path(upload.name).name}"
        target = services.settings.uploads_dir / upload_id
        target.write_bytes(upload.getvalue())
        st.session_state.library_task_id = services.tasks.submit_local_import(
            target,
            title=title or None,
            document_role=document_role,
            parent_paper_id=parent_paper_id,
            prefer_mineru=use_mineru,
        )

    if st.session_state.get("library_task_id"):
        _render_background_task(
            st.session_state.library_task_id, key_prefix="library-import"
        )

    st.subheader("arXiv 导入")
    arxiv_query = st.text_input("arXiv 检索词")
    if st.button("检索 arXiv") and arxiv_query:
        st.session_state.arxiv_results = services.arxiv.search(arxiv_query, 8)
    for item in st.session_state.get("arxiv_results", []):
        with st.container(border=True):
            st.markdown(f"**{item.title}**")
            st.caption(f"{item.arxiv_id} · {', '.join(item.authors[:4])}")
            st.write(item.abstract)
            if st.button(
                "加入后台下载与导入队列",
                key=f"import-{item.arxiv_id}",
                disabled=item.already_imported,
            ):
                st.session_state.library_task_id = services.tasks.submit_arxiv_import(
                    item.arxiv_id, prefer_mineru=use_mineru
                )
                st.rerun()

    st.subheader("论文库")
    papers = services.repository.list_papers(status=None)
    st.dataframe(papers, use_container_width=True, hide_index=True)
    if papers:
        paper_id = st.selectbox("维护目标", [item["paper_id"] for item in papers])
        image_status = services.library.page_image_status(paper_id)
        st.caption(
            f"页面图像：{image_status['count']}/{image_status['page_count']}；"
            + ("可用于视觉问答" if image_status["complete"] else "需要补全")
        )
        impact = services.library.deletion_impact(paper_id)
        st.caption(
            f"删除将影响 {impact['affected_conversation_count']} 个会话；"
            "历史会话会自动归档并保留为只读。"
        )
        if impact["supplementary_ids"]:
            st.warning(
                "该主论文仍有关联 supplementary，当前禁止直接删除："
                + "、".join(impact["supplementary_ids"])
            )
        confirm = st.text_input("删除需输入 paper_id 二次确认", key="delete_confirm")
        col1, col2, col3, col4 = st.columns(4)
        if col1.button("从 PDF 重新识别标题"):
            detected = services.library.refresh_title_from_pdf(paper_id)
            st.success(f"标题已更新：{detected}")
            st.rerun()
        if col2.button("重建当前版本索引"):
            count = services.library.rebuild_active_version(paper_id, prefer_mineru=False)
            st.success(f"已重建 {count} 个 chunks")
        if col3.button("生成/刷新页面图像"):
            with st.spinner("正在本地渲染 PDF 页面……"):
                count = services.library.render_page_images(paper_id)
            st.success(f"已生成 {count} 张页面图像")
            st.rerun()
        if col4.button(
            "删除论文",
            type="secondary",
            disabled=confirm != paper_id or bool(impact["supplementary_ids"]),
        ):
            services.library.delete_paper(paper_id)
            st.success("论文已删除；相关历史会话已归档并设为只读")
            st.rerun()

        action_col1, action_col2 = st.columns(2)
        if action_col1.button("后台生成全文摘要", use_container_width=True):
            st.session_state.library_task_id = services.tasks.submit_summary(paper_id)
            st.rerun()
        maintained_paper = services.repository.get_paper(paper_id)
        version = services.repository.get_version(
            paper_id, int(maintained_paper["active_version"])
        ) if maintained_paper else None
        if version:
            managed_pdf = Path(version["managed_copy_path"])
            if managed_pdf.is_file():
                action_col2.download_button(
                    "下载原始 PDF",
                    data=managed_pdf.read_bytes(),
                    file_name=f"{paper_id}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
                if st.toggle("加载原始 PDF 阅读器", key=f"pdf-preview-{paper_id}"):
                    st.pdf(str(managed_pdf), height=720, key=f"pdf-reader-{paper_id}")

    st.subheader("最近导入任务")
    st.dataframe(services.repository.list_jobs(), use_container_width=True, hide_index=True)
    st.subheader("持久化后台任务")
    background_tasks = services.repository.list_background_tasks(limit=20)
    st.dataframe(
        [
            {
                "task_id": item["task_id"],
                "type": item["task_type"],
                "status": item["status"],
                "progress": item["progress"],
                "step": item["current_step"],
                "updated_at": item["updated_at"],
            }
            for item in background_tasks
        ],
        use_container_width=True,
        hide_index=True,
    )


def comparison_page() -> None:
    services, _ = runtime()
    st.title("多论文比较")
    papers = ready_papers()
    mapping = {f"{item['title']} ({item['paper_id']})": item["paper_id"] for item in papers}
    labels = st.multiselect("选择 2–5 篇论文", list(mapping), max_selections=5)
    if st.button("后台生成带证据比较", type="primary", disabled=len(labels) < 2):
        st.session_state.comparison_task_id = services.tasks.submit_comparison(
            [mapping[label] for label in labels]
        )
        st.rerun()

    recent_comparisons = services.repository.list_background_tasks(
        task_types=[TASK_COMPARE], limit=10
    )
    if recent_comparisons:
        task_ids = [item["task_id"] for item in recent_comparisons]
        current_task_id = st.session_state.get("comparison_task_id")
        if current_task_id not in task_ids:
            current_task_id = task_ids[0]
        selected_task_id = st.selectbox(
            "比较任务历史",
            task_ids,
            index=task_ids.index(current_task_id),
            format_func=lambda value: next(
                f"{item['status']} · {item['request'].get('paper_ids', [])} · {item['created_at']}"
                for item in recent_comparisons
                if item["task_id"] == value
            ),
        )
        st.session_state.comparison_task_id = selected_task_id
        _render_background_task(selected_task_id, key_prefix="comparison")
    else:
        st.info("尚无比较任务。选择论文后，任务会在后台运行并可在重启后恢复。")


def retrieval_debug_page() -> None:
    services, _ = runtime()
    st.title("检索调试")
    papers = ready_papers()
    mapping = {item["paper_id"]: item["title"] for item in papers}
    paper_ids = st.multiselect("论文范围", list(mapping), format_func=lambda x: mapping[x])
    query = st.text_input("检索查询")
    if st.button("运行逐论文检索", disabled=not query or not paper_ids):
        hits = services.rag.retrieve(query, paper_ids)
        for rank, hit in enumerate(hits, 1):
            with st.expander(
                f"#{rank} {hit.chunk.paper_title} · p.{hit.chunk.page_number} · score={hit.score:.4f}"
            ):
                st.caption(hit.chunk.chunk_id)
                st.write(hit.chunk.text)
    trace_id = st.text_input("查看 retrieval trace ID")
    if trace_id and st.button("读取 trace"):
        trace = services.repository.get_retrieval_trace(trace_id)
        st.json(trace or {"error": "not found"})


def evaluation_page() -> None:
    services, _ = runtime()
    st.title("RAG 评测")
    st.write("评测指标：Hit@k、论文范围准确率、引用有效率、拒答准确率和端到端延迟。")
    dataset = Path(__file__).parent / "data" / "eval" / "questions.v1.jsonl"
    validation = validate_eval_dataset(services.repository, dataset)
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("真实问题", validation["case_count"])
    metric_col2.metric("拒答问题", validation["refusal_count"])
    metric_col3.metric("数据集状态", "通过" if validation["valid"] else "失败")
    if validation["errors"]:
        st.error("\n".join(validation["errors"]))
    reports = {
        "完整问答评测": services.settings.project_data_dir / "reports" / "evaluation.json",
        "当前索引检索评测": services.settings.project_data_dir
        / "reports"
        / "retrieval-evaluation.json",
        "分块参数基准": services.settings.project_data_dir
        / "reports"
        / "chunking-benchmark.json",
    }
    for label, result_path in reports.items():
        with st.expander(label, expanded=result_path.exists()):
            if result_path.exists():
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                st.json(payload)
                st.download_button(
                    f"下载{label}",
                    data=json.dumps(payload, ensure_ascii=False, indent=2),
                    file_name=result_path.name,
                    mime="application/json",
                    key=f"download-eval-{result_path.stem}",
                )
            else:
                st.info("尚无结果，请运行下方对应命令。")
    st.warning("检索评测会调用 Embedding API；分块基准会重新嵌入三组 chunks，可能产生费用。")
    st.code(
        "research-copilot validate-eval\n"
        "research-copilot evaluate-retrieval\n"
        "research-copilot benchmark-chunking\n"
        "research-copilot evaluate --dataset data/eval/questions.v1.jsonl",
        language="powershell",
    )


pages = st.navigation(
    [
        st.Page(chat_page, title="智能对话", icon="💬"),
        st.Page(library_page, title="论文库与导入状态", icon="📚"),
        st.Page(comparison_page, title="多论文比较", icon="⚖️"),
        st.Page(retrieval_debug_page, title="检索调试", icon="🔎"),
        st.Page(evaluation_page, title="RAG 评测", icon="📊"),
    ]
)
pages.run()
