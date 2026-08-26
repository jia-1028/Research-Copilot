from __future__ import annotations

import json
import operator
import re
import uuid
from collections import OrderedDict
from collections.abc import Callable
from typing import Annotated, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from research_copilot.errors import ResearchCopilotError
from research_copilot.models import (
    Citation,
    DeepAnalysisAnswer,
    DeepAnalysisPlan,
    DeepAnalysisTaskDraft,
    DeepFacetReport,
    GroundedAnswerDraft,
)
from research_copilot.rag import PaperRAGService

DEEP_ANALYSIS_PROMPT_VERSION = "paper-deep-analysis-v1"

PLANNER_PROMPT = """你是论文深度分析的协调 Agent。把用户问题拆成 2–3 个互补且不重叠的研究任务。
每个任务必须包含：focus（短标题）、question（该专家要回答的完整问题）和 retrieval_query
（适合论文向量检索的中英文关键词）。优先覆盖用户明确要求的方面，不得开始回答论文内容。
只输出符合给定 JSON Schema 的 JSON 对象。"""

WORKER_PROMPT = """你是论文证据分析 specialist。只能依据本次提供的论文证据完成分配任务。
每个事实必须使用证据中给定的 [T#-C#] 引用；used_citation_ids 只能返回实际使用的 T#-C#。
不得使用训练记忆补全论文事实。证据不足时设置 insufficient_evidence=true 并明确缺口。
回答使用中文，保留必要英文术语。只输出符合给定 JSON Schema 的 JSON 对象。"""

SYNTHESIS_PROMPT = """你是多 Agent 论文分析的协调 Agent。依据 specialist reports 和论文证据，
合并为结构清晰、去除重复、保留细节的中文回答。不得增加 specialist 和证据未支持的事实；
每个事实使用 [T#-C#]，used_citation_ids 只返回实际使用的引用。若任一关键方面缺证据，
保留相应 limitation；不得把不同论文、版本、任务或指标混淆。只输出符合 JSON Schema 的对象。"""


class AssignedTask(TypedDict):
    task_id: str
    focus: str
    question: str
    retrieval_query: str


class DeepAnalysisState(TypedDict, total=False):
    question: str
    paper_ids: list[str]
    tasks: list[AssignedTask]
    planner_fallback: bool
    reports: Annotated[list[DeepFacetReport], operator.add]
    citations: Annotated[list[Citation], operator.add]
    retrieved_chunk_ids: Annotated[list[str], operator.add]
    final_answer: DeepAnalysisAnswer
    synthesis_fallback: bool


class DeepAnalysisService:
    """Bounded supervisor/worker/reducer graph for evidence-grounded deep analysis."""

    def __init__(self, rag: PaperRAGService, chat_model: BaseChatModel):
        self.rag = rag
        self.chat_model = chat_model
        self.graph = self._build_graph()

    def analyze(
        self,
        question: str,
        paper_ids: list[str],
        *,
        trace_question: str | None = None,
        thread_id: str | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> DeepAnalysisAnswer:
        def progress(message: str) -> None:
            if progress_callback:
                progress_callback(message)

        if not question.strip():
            raise ResearchCopilotError("问题不能为空")
        unique_ids = list(OrderedDict.fromkeys(paper_ids))
        if not unique_ids or len(unique_ids) > 3:
            raise ResearchCopilotError("多 Agent 深度分析需要选择 1–3 篇 ready 论文")
        papers = self.rag.resolve_ready_papers(unique_ids)
        progress("协调 Agent 正在规划互补的分析维度（最多 3 个）……")

        final_answer: DeepAnalysisAnswer | None = None
        retrieved_chunk_ids: list[str] = []
        stream = self.graph.stream(
            {
                "question": question.strip(),
                "paper_ids": unique_ids,
                "reports": [],
                "citations": [],
                "retrieved_chunk_ids": [],
            },
            stream_mode="updates",
        )
        for update in stream:
            if plan_update := update.get("plan"):
                tasks = plan_update["tasks"]
                labels = "、".join(task["focus"] for task in tasks)
                suffix = "（规划模型失败，已使用安全默认计划）" if plan_update.get(
                    "planner_fallback"
                ) else ""
                progress(f"已拆分 {len(tasks)} 个 specialist：{labels}{suffix}")
            if worker_update := update.get("worker"):
                retrieved_chunk_ids.extend(worker_update.get("retrieved_chunk_ids", []))
                report = worker_update["reports"][0]
                state = "证据不足" if report.insufficient_evidence else "完成"
                progress(f"specialist {report.task_id}：{report.focus}——{state}")
            if synthesis_update := update.get("synthesize"):
                final_answer = synthesis_update["final_answer"]
                if synthesis_update.get("synthesis_fallback"):
                    progress("协调模型不可用，已确定性合并各 specialist 报告。")

        if final_answer is None:
            raise ResearchCopilotError("多 Agent 深度分析未生成最终结果")

        trace_id = final_answer.retrieval_trace_id
        self.rag.repository.save_retrieval_trace(
            trace_id=trace_id,
            thread_id=thread_id,
            question=trace_question or question,
            standalone_query="multi-agent:" + " | ".join(task.focus for task in final_answer.facet_reports),
            paper_ids=unique_ids,
            paper_versions={
                paper_id: int(paper["active_version"]) for paper_id, paper in papers.items()
            },
            retrieved_chunk_ids=list(OrderedDict.fromkeys(retrieved_chunk_ids)),
            used_chunk_ids=[
                citation.chunk_id
                for citation in final_answer.citations
            ],
            prompt_version=DEEP_ANALYSIS_PROMPT_VERSION,
        )
        if thread_id:
            state = self.rag.repository.get_conversation_state(thread_id) or {}
            self.rag.repository.save_conversation_state(
                thread_id,
                active_paper_ids=unique_ids,
                last_arxiv_result_ids=state.get("last_arxiv_result_ids", []),
                last_retrieval_trace_id=trace_id,
            )
        progress(f"多 Agent 汇总与引用校验完成，retrieval trace：{trace_id}")
        return final_answer

    def _build_graph(self):
        builder = StateGraph(DeepAnalysisState)
        builder.add_node("plan", self._plan)
        builder.add_node("worker", self._worker)
        builder.add_node("synthesize", self._synthesize)
        builder.add_edge(START, "plan")
        builder.add_conditional_edges("plan", self._dispatch, ["worker"])
        builder.add_edge("worker", "synthesize")
        builder.add_edge("synthesize", END)
        return builder.compile()

    def _plan(self, state: DeepAnalysisState) -> dict:
        fallback = False
        schema = json.dumps(DeepAnalysisPlan.model_json_schema(), ensure_ascii=False)
        try:
            plan = self.chat_model.with_structured_output(
                DeepAnalysisPlan, method="json_mode"
            ).invoke(
                [
                    SystemMessage(content=PLANNER_PROMPT + "\nJSON Schema:\n" + schema),
                    HumanMessage(content="用户问题：" + state["question"]),
                ]
            )
            plan = DeepAnalysisPlan.model_validate(plan)
        except Exception:  # noqa: BLE001 - deterministic fallback keeps the graph usable
            fallback = True
            plan = self._fallback_plan(state["question"])

        tasks = [
            AssignedTask(
                task_id=f"T{index}",
                focus=task.focus.strip(),
                question=task.question.strip(),
                retrieval_query=task.retrieval_query.strip(),
            )
            for index, task in enumerate(plan.tasks, start=1)
        ]
        return {"tasks": tasks, "planner_fallback": fallback}

    @staticmethod
    def _fallback_plan(question: str) -> DeepAnalysisPlan:
        return DeepAnalysisPlan(
            tasks=[
                DeepAnalysisTaskDraft(
                    focus="研究目标与整体架构",
                    question=f"围绕“{question}”分析研究目标、总体方案与信息流。",
                    retrieval_query="Abstract proposed method overall architecture framework pipeline",
                ),
                DeepAnalysisTaskDraft(
                    focus="核心模块与实现机制",
                    question=f"围绕“{question}”分析关键模块、公式、连接关系与实现机制。",
                    retrieval_query="proposed method module architecture equation algorithm implementation",
                ),
                DeepAnalysisTaskDraft(
                    focus="训练验证与证据局限",
                    question=f"围绕“{question}”分析训练设置、损失函数、消融验证与证据缺口。",
                    retrieval_query="training loss implementation details ablation limitations",
                ),
            ]
        )

    @staticmethod
    def _dispatch(state: DeepAnalysisState) -> list[Send]:
        return [
            Send(
                "worker",
                {
                    "question": state["question"],
                    "paper_ids": state["paper_ids"],
                    "tasks": [task],
                },
            )
            for task in state["tasks"]
        ]

    def _worker(self, state: DeepAnalysisState) -> dict:
        task = state["tasks"][0]
        retrieved = []
        citations: list[Citation] = []
        try:
            query = f"{task['question']}\n检索重点：{task['retrieval_query']}"
            retrieved = self.rag.retrieve(query, state["paper_ids"], per_paper_k=16)
            local_citations, context = self.rag._build_evidence(retrieved)
            citations, context = self._scope_citations(task["task_id"], local_citations, context)
            if not citations:
                raise ResearchCopilotError("未检索到证据")
            schema = json.dumps(GroundedAnswerDraft.model_json_schema(), ensure_ascii=False)
            draft = self.chat_model.with_structured_output(
                GroundedAnswerDraft, method="json_mode"
            ).invoke(
                [
                    SystemMessage(content=WORKER_PROMPT + "\nJSON Schema:\n" + schema),
                    HumanMessage(
                        content=f"分析任务：{task['question']}\n\n论文证据：\n{context}"
                    ),
                ]
            )
            draft = self._validate_draft(GroundedAnswerDraft.model_validate(draft), citations)
            used = set(draft.used_citation_ids)
            citations = [item for item in citations if item.citation_id in used]
            report = DeepFacetReport(
                task_id=task["task_id"],
                focus=task["focus"],
                **draft.model_dump(),
            )
        except Exception as exc:  # noqa: BLE001 - one specialist must not abort its peers
            report = DeepFacetReport(
                task_id=task["task_id"],
                focus=task["focus"],
                answer=f"{task['focus']}未获得可验证的完整结论。",
                used_citation_ids=[],
                insufficient_evidence=True,
                limitations=[f"specialist 执行失败：{type(exc).__name__}"],
            )
            citations = []
        return {
            "reports": [report],
            "citations": citations,
            "retrieved_chunk_ids": [item.chunk.chunk_id for item in retrieved],
        }

    def _synthesize(self, state: DeepAnalysisState) -> dict:
        reports = sorted(state.get("reports", []), key=lambda item: item.task_id)
        citations = sorted(state.get("citations", []), key=lambda item: item.citation_id)
        fallback = False
        if not citations:
            draft = GroundedAnswerDraft(
                answer="所有 specialist 均未获得可引用证据，无法完成深度分析。",
                used_citation_ids=[],
                insufficient_evidence=True,
                limitations=[item for report in reports for item in report.limitations]
                or ["没有可用论文证据"],
            )
        else:
            schema = json.dumps(GroundedAnswerDraft.model_json_schema(), ensure_ascii=False)
            evidence = "\n\n".join(
                f"[{item.citation_id}] 论文={item.paper_title}; PDF页={item.pdf_page}\n"
                f"{item.evidence_text}"
                for item in citations
            )
            try:
                raw = self.chat_model.with_structured_output(
                    GroundedAnswerDraft, method="json_mode"
                ).invoke(
                    [
                        SystemMessage(
                            content=SYNTHESIS_PROMPT + "\nJSON Schema:\n" + schema
                        ),
                        HumanMessage(
                            content="用户问题："
                            + state["question"]
                            + "\n\nspecialist reports：\n"
                            + json.dumps(
                                [report.model_dump() for report in reports],
                                ensure_ascii=False,
                            )
                            + "\n\n论文证据：\n"
                            + evidence
                        ),
                    ]
                )
                draft = self._validate_draft(
                    GroundedAnswerDraft.model_validate(raw), citations
                )
            except Exception:  # noqa: BLE001 - reports are already citation validated
                fallback = True
                draft = self._deterministic_synthesis(reports)

        used = set(draft.used_citation_ids)
        final_citations = [item for item in citations if item.citation_id in used]
        return {
            "final_answer": DeepAnalysisAnswer(
                **draft.model_dump(),
                citations=final_citations,
                retrieval_trace_id=str(uuid.uuid4()),
                facet_reports=reports,
            ),
            "synthesis_fallback": fallback,
        }

    @staticmethod
    def _scope_citations(
        task_id: str, citations: list[Citation], context: str
    ) -> tuple[list[Citation], str]:
        scoped = []
        for citation in citations:
            new_id = f"{task_id}-{citation.citation_id}"
            context = context.replace(
                f"[{citation.citation_id}]", f"[{new_id}]"
            )
            scoped.append(citation.model_copy(update={"citation_id": new_id}))
        return scoped, context

    @staticmethod
    def _validate_draft(
        draft: GroundedAnswerDraft, citations: list[Citation]
    ) -> GroundedAnswerDraft:
        valid = {citation.citation_id for citation in citations}
        normalized = [item.strip().strip("[]") for item in draft.used_citation_ids]
        inline = set(re.findall(r"\[(T\d+-C\d+)\]", draft.answer))
        invalid = (set(normalized) | inline) - valid
        if invalid:
            raise ResearchCopilotError(f"多 Agent 返回非法引用 ID：{sorted(invalid)}")
        used = list(OrderedDict.fromkeys(normalized + sorted(inline)))
        if not used and not draft.insufficient_evidence:
            return draft.model_copy(
                update={
                    "answer": "当前 specialist 证据不足，不能给出确定性结论。",
                    "used_citation_ids": [],
                    "insufficient_evidence": True,
                    "limitations": draft.limitations + ["specialist 未引用检索证据"],
                }
            )
        return draft.model_copy(update={"used_citation_ids": used})

    @staticmethod
    def _deterministic_synthesis(reports: list[DeepFacetReport]) -> GroundedAnswerDraft:
        usable = [report for report in reports if report.used_citation_ids]
        if not usable:
            return GroundedAnswerDraft(
                answer="所有 specialist 均未获得可引用证据，无法完成深度分析。",
                insufficient_evidence=True,
                limitations=[item for report in reports for item in report.limitations],
            )
        answer = "\n\n".join(
            f"## {report.focus}\n\n{report.answer}" for report in usable
        )
        limitations = list(
            OrderedDict.fromkeys(item for report in reports for item in report.limitations)
        )
        return GroundedAnswerDraft(
            answer=answer,
            used_citation_ids=list(
                OrderedDict.fromkeys(
                    citation_id
                    for report in usable
                    for citation_id in report.used_citation_ids
                )
            ),
            insufficient_evidence=any(report.insufficient_evidence for report in reports),
            limitations=limitations,
        )
