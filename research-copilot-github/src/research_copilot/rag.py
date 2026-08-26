from __future__ import annotations

import base64
import hashlib
import json
import re
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Iterable
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from research_copilot.config import Settings
from research_copilot.errors import ResearchCopilotError
from research_copilot.models import (
    Citation,
    ComparisonDimension,
    ComparisonNarrative,
    ComparisonRow,
    EvidenceValue,
    GroundedAnswer,
    GroundedAnswerDraft,
    PaperComparison,
    PaperProfile,
    PaperProfileDraft,
    RetrievedChunk,
)
from research_copilot.storage import SQLiteRepository
from research_copilot.vector_index import VectorIndex

QA_PROMPT_VERSION = "paper-qa-v3"
QA_EXPANSION_PROMPT_VERSION = "paper-qa-evidence-expansion-v1"
PROFILE_PROMPT_VERSION = "paper-profile-v2"
FAST_PROFILE_PROMPT_VERSION = "paper-profile-fast-v1"
FAST_PROFILE_MODE = "dimension_retrieval"
COMPARISON_PROMPT_VERSION = "paper-comparison-fast-v1"
SUMMARY_MAX_BATCH_CHUNKS = 20
SUMMARY_MIN_BATCH_CHUNKS = 8
SUMMARY_MAP_VALUE_MAX_CHARS = 70
SUMMARY_MAP_CITATION_LIMIT = 2
SUMMARY_MERGE_ITEMS_PER_DIMENSION = 3
SUMMARY_MERGE_CITATION_LIMIT = 6
DIMENSIONS: tuple[ComparisonDimension, ...] = (
    "research_problem",
    "core_contributions",
    "method_architecture",
    "datasets",
    "experimental_setup",
    "metrics",
    "main_results",
    "efficiency",
    "limitations",
)

DIMENSION_QUERIES: dict[ComparisonDimension, str] = {
    "research_problem": "研究问题、任务定义、现有方法局限、研究目标",
    "core_contributions": "核心贡献、创新点、主要工作、本文提出",
    "method_architecture": "方法架构、模型结构、关键模块、算法流程",
    "datasets": "实验数据集、训练集、验证集、测试集、数据划分",
    "experimental_setup": "实验设置、训练配置、实现细节、对比基线、消融实验",
    "metrics": "评价指标、DSC、Dice、HD95、准确率、参数量、FLOPs、BOPs",
    "main_results": "主要实验结果、定量结果、对比结果、表格、性能提升",
    "efficiency": "模型效率、参数量、计算量、推理速度、显存、模型大小、延迟",
    "limitations": "局限性、失败案例、未来工作、适用范围、部署限制",
}

QA_SYSTEM_PROMPT = """你是严谨的论文证据问答助手。只能依据本次提供的证据回答。
规则：
1. 每个可核验事实都必须在答案正文中引用一个或多个 [C#] 或 [F#]。
2. 不得使用训练记忆补全论文事实，不得编造实验数值、数据集、结论或页码。
3. 证据不足时设置 insufficient_evidence=true，明确缺少什么；不要用猜测填空。
4. 不同论文、不同版本和不同指标不可混淆。回答使用中文，保留必要英文术语。
5. [C#] 表示文本证据，[F#] 表示 PDF 页面图像证据；引用 ID 必须来自本次证据块。
6. 只能使用 C1、C2、F1、F2 这类短引用 ID，不得输出 paper_id、version 或内部 chunk_id 作为引用。
7. “它们、这些论文、各篇论文”指本次列出的全部论文范围；分类或比较问题应逐篇作答，
   不得因为代词而声称用户没有指定对象。
8. “补充检索关注点”只是检索目标，不是事实证据；只有标记为 [C#] 的 PDF 原文可以支撑回答。
9. 先输出带 [C#] 引用的自然语言答案，然后严格在末尾附加以下状态块；不要输出 JSON：
<EVIDENCE_STATUS>
insufficient_evidence: true 或 false
limitations:
- 证据缺口；没有缺口时不写项目
</EVIDENCE_STATUS>"""

PROFILE_SYSTEM_PROMPT = """从给定论文证据构建结构化论文画像。每个字段都必须是 EvidenceValue：
value 只写证据支持的内容；citation_ids 只使用给定 [C#]；若无证据则 value 写“证据不足”，
并设置 insufficient_evidence=true。不得把不同数据集、任务或指标合并成可直接比较的结果。"""

SUMMARY_MAP_SYSTEM_PROMPT = PROFILE_SYSTEM_PROMPT + """
你正在为全文摘要提取一个短小的局部证据画像，而不是撰写长篇综述。
每个 value 最多 70 个中文字符或 140 个英文字符，只保留最关键、可核验的事实；
每个字段最多引用 2 个 [C#]。不得重复论文标题、背景套话或未在本段证据中出现的内容。
输出必须完整覆盖全部九个字段，无法支持的字段明确标记为证据不足。"""


class PaperRAGService:
    def __init__(
        self,
        settings: Settings,
        repository: SQLiteRepository,
        vector_index: VectorIndex,
        chat_model: BaseChatModel,
        fallback_chat_model: BaseChatModel | None = None,
    ):
        self.settings = settings
        self.repository = repository
        self.vector_index = vector_index
        self.chat_model = chat_model
        self.fallback_chat_model = fallback_chat_model
        # Experimental chat models may support text, vision and tool calls but
        # reject response_format. Reuse the fallback for schema-heavy profile
        # jobs while keeping normal RAG answers on the primary model.
        self.structured_chat_model = fallback_chat_model or chat_model
        self._primary_unavailable_until = 0.0

    def resolve_ready_papers(self, paper_ids: Iterable[str]) -> dict[str, dict]:
        resolved: dict[str, dict] = {}
        for paper_id in OrderedDict.fromkeys(paper_ids):
            paper = self.repository.get_paper(paper_id)
            if paper is None:
                raise ResearchCopilotError(f"论文不存在：{paper_id}")
            if paper["status"] != "ready" or int(paper["active_version"]) < 1:
                raise ResearchCopilotError(f"论文尚未 ready：{paper_id}")
            resolved[paper_id] = paper
        if not resolved:
            raise ResearchCopilotError("至少选择一篇 ready 论文")
        return resolved

    def retrieve(
        self,
        question: str,
        paper_ids: list[str],
        *,
        per_paper_k: int | None = None,
        context_per_paper: int | None = None,
    ) -> list[RetrievedChunk]:
        papers = self.resolve_ready_papers(paper_ids)
        k = per_paper_k or self.settings.retrieval_candidate_k
        context_k = context_per_paper or self.settings.retrieval_context_k
        selected: list[RetrievedChunk] = []
        # VectorIndex embeds the query once, then applies a separate Chroma
        # filter for every paper.  This preserves per-paper Top-k without making
        # the same paid/network embedding request N times for N papers.
        hits_by_paper: dict[str, list[RetrievedChunk]] = {
            paper_id: [] for paper_id in papers
        }
        all_hits = self.vector_index.similarity_search(
            question,
            {
                paper_id: int(paper["active_version"])
                for paper_id, paper in papers.items()
            },
            k,
        )
        for hit in all_hits:
            if hit.chunk.paper_id in hits_by_paper:
                hits_by_paper[hit.chunk.paper_id].append(hit)

        # Rerank and cap each paper independently: no global Top-k starvation.
        for paper_id in papers:
            hits = hits_by_paper[paper_id]
            hits = self._rerank_retrieved(question, hits)
            seen_hashes: set[str] = set()
            for hit in hits:
                if hit.chunk.text_hash in seen_hashes:
                    continue
                seen_hashes.add(hit.chunk.text_hash)
                selected.append(hit)
                if len(seen_hashes) >= context_k:
                    break
        return selected

    def ask(
        self,
        question: str,
        paper_ids: list[str],
        *,
        retrieval_question: str | None = None,
        thread_id: str | None = None,
        progress_callback: Callable[[str], None] | None = None,
        evidence_hints: list[str] | None = None,
        previous_trace_id: str | None = None,
        evidence_expansion_attempt: int = 0,
    ) -> GroundedAnswer:
        def progress(message: str) -> None:
            if progress_callback:
                progress_callback(message)

        if not question.strip():
            raise ResearchCopilotError("问题不能为空")
        progress("正在检查论文状态与 active version……")
        papers = self.resolve_ready_papers(paper_ids)
        progress(
            "已锁定论文范围："
            + "；".join(
                f"{paper['title']} (v{paper['active_version']})" for paper in papers.values()
            )
        )
        contextual_question = (retrieval_question or question).strip()
        standalone_query = self._expand_retrieval_query(contextual_question)
        if contextual_question != question.strip():
            progress("已结合当前会话将省略式追问改写为独立检索问题。")
        elif standalone_query != question.strip():
            progress("已根据问题意图扩展检索词，不额外调用模型。")
        expanded_query = standalone_query != contextual_question
        trace_query = standalone_query
        if evidence_hints:
            progress("正在把上一轮的证据缺口转换为补充检索关注点……")
            retrieved, trace_query = self._retrieve_expanded_evidence(
                standalone_query,
                evidence_hints,
                paper_ids,
                papers,
                previous_trace_id=previous_trace_id,
                progress_callback=progress,
            )
        else:
            progress("正在逐篇执行向量检索、参考文献降权和章节重排……")
            retrieved = self.retrieve(
                standalone_query,
                paper_ids,
                per_paper_k=20 if expanded_query else None,
            )
        progress(
            f"已选出 {len(retrieved)} 个证据块："
            + "、".join(
                f"{item.chunk.paper_title} PDF第{item.chunk.page_number}页"
                for item in retrieved
            )
        )
        trace_id = str(uuid.uuid4())
        citations, context = self._build_evidence(retrieved)
        if not citations:
            progress("未检索到可引用证据，正在生成证据不足响应。")
            answer = GroundedAnswer(
                answer="当前论文证据不足，无法回答该问题。",
                used_citation_ids=[],
                insufficient_evidence=True,
                limitations=["检索未返回可用证据"],
                citations=[],
                retrieval_trace_id=trace_id,
                evidence_expansion_attempt=evidence_expansion_attempt,
                source_trace_id=previous_trace_id,
            )
            self._save_trace(
                trace_id,
                thread_id,
                question,
                papers,
                [],
                [],
                standalone_query=trace_query,
                prompt_version=(
                    QA_EXPANSION_PROMPT_VERSION if evidence_hints else QA_PROMPT_VERSION
                ),
            )
            return answer

        paper_scope = "\n".join(
            f"- {paper['title']} (paper_id={paper_id}, v{paper['active_version']})"
            for paper_id, paper in papers.items()
        )
        hint_context = ""
        if evidence_hints:
            hint_context = (
                "\n\n补充检索关注点（仅用于说明需要寻找什么，不属于论文证据）：\n- "
                + "\n- ".join(evidence_hints[:6])
            )
        visual_citations, visual_parts = self._build_visual_evidence(
            contextual_question, retrieved, papers
        )
        citations.extend(visual_citations)
        if visual_citations:
            progress(
                f"已加载 {len(visual_citations)} 张相关 PDF 页面图像作为视觉证据："
                + "、".join(
                    f"{item.paper_title} 第{item.pdf_page}页"
                    for item in visual_citations
                )
            )
        elif self._visual_page_limit(contextual_question):
            progress("问题可能需要视觉证据，但当前版本尚无可用页面图像；继续使用文本证据。")
        human_text = (
            f"本次论文范围（问题中的‘它们/这些论文’均指这里的全部论文）：\n"
            f"{paper_scope}\n\n问题：{contextual_question}{hint_context}"
            f"\n\n论文文本证据：\n{context}"
        )
        human_content = (
            [{"type": "text", "text": human_text}, *visual_parts]
            if visual_parts
            else human_text
        )
        messages = [
            SystemMessage(content=QA_SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]
        progress(
            f"主模型 {self.settings.chat_provider}/{self.settings.chat_model} "
            "正在仅依据候选证据生成带引用回答……"
        )
        response, generation_model, fallback_used = self._invoke_answer_model(
            messages, progress
        )
        progress("模型已返回，正在校验引用 ID、论文范围与版本……")
        draft = self._validate_answer_draft(
            self._parse_grounded_response(response), citations
        )
        used = [citation for citation in citations if citation.citation_id in draft.used_citation_ids]
        self._save_trace(
            trace_id,
            thread_id,
            question,
            papers,
            [item.chunk.chunk_id for item in retrieved]
            + [item.chunk_id for item in visual_citations],
            [item.chunk_id for item in used],
            standalone_query=trace_query,
            prompt_version=(
                QA_EXPANSION_PROMPT_VERSION if evidence_hints else QA_PROMPT_VERSION
            ),
        )
        if thread_id:
            state = self.repository.get_conversation_state(thread_id) or {}
            self.repository.save_conversation_state(
                thread_id,
                active_paper_ids=paper_ids,
                last_arxiv_result_ids=state.get("last_arxiv_result_ids", []),
                last_retrieval_trace_id=trace_id,
            )
        progress(f"引用校验完成，retrieval trace：{trace_id}")
        return GroundedAnswer(
            **draft.model_dump(),
            citations=used,
            retrieval_trace_id=trace_id,
            evidence_expansion_attempt=evidence_expansion_attempt,
            source_trace_id=previous_trace_id,
            generation_model=generation_model,
            fallback_used=fallback_used,
        )

    def _retrieve_expanded_evidence(
        self,
        question: str,
        evidence_hints: list[str],
        paper_ids: list[str],
        papers: dict[str, dict],
        *,
        previous_trace_id: str | None,
        progress_callback: Callable[[str], None],
    ) -> tuple[list[RetrievedChunk], str]:
        clean_hints = [
            re.sub(r"\s+", " ", hint).strip()[:300]
            for hint in evidence_hints
            if hint and hint.strip()
        ][:6]
        focused_query = (
            f"{question}\n补充查找以下缺失证据：\n- " + "\n- ".join(clean_hints)
        )
        progress_callback("正在使用原问题和证据缺口执行两路逐论文检索……")
        base_hits = self.retrieve(
            question,
            paper_ids,
            per_paper_k=24,
            context_per_paper=8,
        )
        focused_hits = self.retrieve(
            focused_query,
            paper_ids,
            per_paper_k=24,
            context_per_paper=8,
        )

        previous_retrieved: set[str] = set()
        previous_used: list[RetrievedChunk] = []
        previous = (
            self.repository.get_retrieval_trace(previous_trace_id)
            if previous_trace_id
            else None
        )
        active_versions = {
            paper_id: int(paper["active_version"]) for paper_id, paper in papers.items()
        }
        if previous and set(previous["paper_ids"]) == set(paper_ids):
            previous_retrieved = set(previous["retrieved_chunk_ids"])
            for chunk in self.vector_index.get_chunks(previous["used_chunk_ids"]):
                if active_versions.get(chunk.paper_id) == chunk.paper_version:
                    previous_used.append(RetrievedChunk(chunk=chunk, score=1.25))

        cap = 10 if len(papers) == 1 else 8
        selected: list[RetrievedChunk] = []
        for paper_id in papers:
            prior = [item for item in previous_used if item.chunk.paper_id == paper_id]
            focused = [item for item in focused_hits if item.chunk.paper_id == paper_id]
            base = [item for item in base_hits if item.chunk.paper_id == paper_id]
            new_focused = [
                item for item in focused if item.chunk.chunk_id not in previous_retrieved
            ]
            ordered = prior + new_focused + base + focused
            seen_chunks: set[str] = set()
            seen_hashes: set[str] = set()
            paper_selected: list[RetrievedChunk] = []
            # First pass favors page diversity, then fills the remaining slots.
            for prefer_new_page in (True, False):
                seen_pages = {item.chunk.page_number for item in paper_selected}
                for item in ordered:
                    chunk = item.chunk
                    if chunk.chunk_id in seen_chunks or chunk.text_hash in seen_hashes:
                        continue
                    if prefer_new_page and chunk.page_number in seen_pages:
                        continue
                    paper_selected.append(item)
                    seen_chunks.add(chunk.chunk_id)
                    seen_hashes.add(chunk.text_hash)
                    seen_pages.add(chunk.page_number)
                    if len(paper_selected) >= cap:
                        break
                if len(paper_selected) >= cap:
                    break
            selected.extend(paper_selected)

        new_count = sum(
            item.chunk.chunk_id not in previous_retrieved for item in selected
        )
        progress_callback(
            f"补充检索选出 {len(selected)} 个证据块，其中 {new_count} 个不在上一轮候选中。"
        )
        return selected, focused_query

    def summarize(self, paper_id: str) -> PaperProfile:
        profile, _ = self._summarize_with_citations(paper_id)
        return profile

    def _summarize_with_citations(
        self, paper_id: str
    ) -> tuple[PaperProfile, list[Citation]]:
        paper = self.resolve_ready_papers([paper_id])[paper_id]
        chunks = self.vector_index.get_paper_chunks(paper_id, int(paper["active_version"]))
        if not chunks:
            raise ResearchCopilotError(f"论文没有可用文本块：{paper_id}")

        # Map: all pages participate, but each model response is deliberately
        # compact. A long PaperProfile JSON can otherwise hit the provider's
        # completion limit and become unparsable.
        map_profiles: list[PaperProfileDraft] = []
        all_citations: list[Citation] = []
        batch_size = min(self.settings.summary_batch_chunks, SUMMARY_MAX_BATCH_CHUNKS)
        pending_batches = [
            chunks[batch_start : batch_start + batch_size]
            for batch_start in range(0, len(chunks), batch_size)
        ]
        while pending_batches:
            raw_batch = pending_batches.pop(0)
            batch = [RetrievedChunk(chunk=chunk) for chunk in raw_batch]
            citations, context = self._build_evidence(batch, start=len(all_citations) + 1)
            try:
                profile = self.structured_chat_model.with_structured_output(
                    PaperProfileDraft
                ).invoke(
                    [
                        SystemMessage(content=SUMMARY_MAP_SYSTEM_PROMPT),
                        HumanMessage(
                            content=f"这是论文的一部分，请提取有证据的局部画像：\n{context}"
                        ),
                    ]
                )
            except Exception as exc:
                if self._is_summary_length_error(exc) and len(raw_batch) > SUMMARY_MIN_BATCH_CHUNKS:
                    midpoint = len(raw_batch) // 2
                    pending_batches[0:0] = [raw_batch[:midpoint], raw_batch[midpoint:]]
                    continue
                if self._is_summary_length_error(exc):
                    raise ResearchCopilotError(
                        "全文摘要的最小分段仍超出模型输出长度，请稍后重试或降低摘要详细度。"
                    ) from exc
                raise
            all_citations.extend(citations)
            validated = self._validate_profile(profile, citations)
            map_profiles.append(self._compact_summary_profile(validated))

        # Reduce locally, rather than asking for a second long JSON response.
        # This makes every model response bounded and keeps already-completed
        # map results useful even for a long paper.
        reduced = self._merge_summary_profiles(map_profiles)
        used_ids = {
            citation_id
            for dimension in DIMENSIONS
            for citation_id in getattr(reduced, dimension).citation_ids
        }
        # Public profile citation IDs include paper identity, preventing C1 collisions
        # when several independently summarized papers are combined.
        remapped_values = {
            dimension: getattr(reduced, dimension).model_copy(
                update={
                    "citation_ids": [
                        f"{paper_id}:{citation_id}"
                        for citation_id in getattr(reduced, dimension).citation_ids
                    ]
                }
            )
            for dimension in DIMENSIONS
        }
        return (
            PaperProfile(
                paper_id=paper_id, paper_title=paper["title"], **remapped_values
            ),
            [
                item.model_copy(update={"citation_id": f"{paper_id}:{item.citation_id}"})
                for item in all_citations
                if item.citation_id in used_ids
            ],
        )

    @staticmethod
    def _is_summary_length_error(exc: Exception) -> bool:
        message = str(exc).casefold()
        return "length limit" in message or (
            "completion_tokens" in message and "parse" in message
        )

    @staticmethod
    def _compact_summary_profile(profile: PaperProfileDraft) -> PaperProfileDraft:
        updates: dict[ComparisonDimension, EvidenceValue] = {}
        for dimension in DIMENSIONS:
            value: EvidenceValue = getattr(profile, dimension)
            text = " ".join(value.value.split())
            if len(text) > SUMMARY_MAP_VALUE_MAX_CHARS:
                text = text[: SUMMARY_MAP_VALUE_MAX_CHARS - 1].rstrip() + "…"
            updates[dimension] = value.model_copy(
                update={
                    "value": text,
                    "citation_ids": list(
                        OrderedDict.fromkeys(value.citation_ids)
                    )[:SUMMARY_MAP_CITATION_LIMIT],
                }
            )
        return profile.model_copy(update=updates)

    @staticmethod
    def _merge_summary_profiles(profiles: list[PaperProfileDraft]) -> PaperProfileDraft:
        updates: dict[ComparisonDimension, EvidenceValue] = {}
        for dimension in DIMENSIONS:
            values: list[EvidenceValue] = []
            seen_text: set[str] = set()
            for profile in profiles:
                value: EvidenceValue = getattr(profile, dimension)
                text = " ".join(value.value.split())
                normalized = text.casefold()
                if (
                    value.insufficient_evidence
                    or not text
                    or normalized == "证据不足"
                    or normalized in seen_text
                ):
                    continue
                seen_text.add(normalized)
                values.append(value)
                if len(values) >= SUMMARY_MERGE_ITEMS_PER_DIMENSION:
                    break
            if not values:
                updates[dimension] = EvidenceValue(
                    value="证据不足",
                    insufficient_evidence=True,
                )
                continue
            citation_ids = list(
                OrderedDict.fromkeys(
                    citation_id for value in values for citation_id in value.citation_ids
                )
            )[:SUMMARY_MERGE_CITATION_LIMIT]
            updates[dimension] = EvidenceValue(
                value="\n\n".join(value.value for value in values),
                citation_ids=citation_ids,
            )
        return PaperProfileDraft(**updates)

    def compare(self, paper_ids: list[str]) -> PaperComparison:
        if len(OrderedDict.fromkeys(paper_ids)) < 2:
            raise ResearchCopilotError("多论文比较至少需要两篇不同论文")
        papers = self.resolve_ready_papers(paper_ids)
        cache_payload = [
            [paper_id, int(papers[paper_id]["active_version"])] for paper_id in paper_ids
        ]
        cache_key = hashlib.sha256(
            json.dumps(cache_payload, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        cached = self.repository.get_cached_comparison(
            cache_key, COMPARISON_PROMPT_VERSION
        )
        if cached:
            return PaperComparison.model_validate(cached)
        profile_results = [
            self._comparison_profile_with_citations(paper_id) for paper_id in paper_ids
        ]
        profiles = [item[0] for item in profile_results]
        rows = [
            ComparisonRow(
                dimension=dimension,
                values={profile.paper_id: getattr(profile, dimension) for profile in profiles},
            )
            for dimension in DIMENSIONS
        ]
        narrative = self._build_comparison_narrative(profiles)
        citations = [citation for _, items in profile_results for citation in items]
        comparison = PaperComparison(
            paper_ids=[profile.paper_id for profile in profiles],
            rows=rows,
            similarities=narrative.similarities,
            differences=narrative.differences,
            non_comparable_items=narrative.non_comparable_items,
            citations=citations,
        )
        self.repository.save_cached_comparison(
            cache_key,
            COMPARISON_PROMPT_VERSION,
            comparison.model_dump(mode="json"),
        )
        return comparison

    def _build_comparison_narrative(
        self, profiles: list[PaperProfile]
    ) -> ComparisonNarrative:
        """Build conservative guidance locally; evidence remains in the aligned rows."""
        shared_dimensions = [
            dimension
            for dimension in DIMENSIONS
            if all(not getattr(profile, dimension).insufficient_evidence for profile in profiles)
        ]
        differing_dimensions = [
            dimension
            for dimension in DIMENSIONS
            if len({getattr(profile, dimension).value for profile in profiles}) > 1
        ]
        labels = {
            "research_problem": "研究问题",
            "core_contributions": "核心贡献",
            "method_architecture": "方法架构",
            "datasets": "数据集",
            "experimental_setup": "实验设置",
            "metrics": "评价指标",
            "main_results": "主要结果",
            "efficiency": "效率",
            "limitations": "局限性",
        }
        similarities = [
            "所有论文均提供了这些维度的证据："
            + "、".join(labels[item] for item in shared_dimensions)
        ] if shared_dimensions else ["部分维度证据不足，请直接查看带引用单元格。"]
        differences = [
            "论文在这些维度的证据表述不同："
            + "、".join(labels[item] for item in differing_dimensions)
        ] if differing_dimensions else []
        non_comparable_items = []
        for dimension in ("datasets", "experimental_setup", "metrics"):
            if dimension in differing_dimensions:
                non_comparable_items.append(
                    f"{labels[dimension]}不同；相关数值不可直接横向比较，需以各自证据为准。"
                )
        return ComparisonNarrative(
            similarities=similarities,
            differences=differences,
            non_comparable_items=non_comparable_items,
        )

    def _comparison_profile_with_citations(
        self, paper_id: str
    ) -> tuple[PaperProfile, list[Citation]]:
        paper = self.resolve_ready_papers([paper_id])[paper_id]
        version = int(paper["active_version"])
        cached = self.repository.get_cached_profile(
            paper_id,
            version,
            profile_mode=FAST_PROFILE_MODE,
            prompt_version=FAST_PROFILE_PROMPT_VERSION,
        )
        if cached:
            return (
                PaperProfile.model_validate(cached["profile"]),
                [Citation.model_validate(item) for item in cached["citations"]],
            )

        retrieved: list[RetrievedChunk] = []
        seen_chunks: set[str] = set()
        for dimension in DIMENSIONS:
            query = f"{paper['title']}：{DIMENSION_QUERIES[dimension]}"
            hits = self.vector_index.similarity_search(query, {paper_id: version}, 3)
            for hit in hits:
                if hit.chunk.chunk_id not in seen_chunks:
                    retrieved.append(hit)
                    seen_chunks.add(hit.chunk.chunk_id)
        citations, context = self._build_evidence(retrieved[:24])
        schema_text = json.dumps(
            PaperProfileDraft.model_json_schema(), ensure_ascii=False
        )
        draft = self.structured_chat_model.with_structured_output(
            PaperProfileDraft, method="json_mode"
        ).invoke(
            [
                SystemMessage(
                    content=PROFILE_SYSTEM_PROMPT
                    + "\n必须只输出符合以下 JSON Schema 的 JSON 对象：\n"
                    + schema_text
                ),
                HumanMessage(
                    content=f"为论文生成用于多论文比较的九维画像：\n{context}"
                ),
            ]
        )
        draft = self._validate_profile(draft, citations)
        used_ids = {
            citation_id
            for dimension in DIMENSIONS
            for citation_id in getattr(draft, dimension).citation_ids
        }
        remapped_values = {
            dimension: getattr(draft, dimension).model_copy(
                update={
                    "citation_ids": [
                        f"{paper_id}:{citation_id}"
                        for citation_id in getattr(draft, dimension).citation_ids
                    ]
                }
            )
            for dimension in DIMENSIONS
        }
        profile = PaperProfile(
            paper_id=paper_id,
            paper_title=paper["title"],
            **remapped_values,
        )
        used_citations = [
            citation.model_copy(
                update={"citation_id": f"{paper_id}:{citation.citation_id}"}
            )
            for citation in citations
            if citation.citation_id in used_ids
        ]
        self.repository.save_cached_profile(
            paper_id,
            version,
            profile_mode=FAST_PROFILE_MODE,
            prompt_version=FAST_PROFILE_PROMPT_VERSION,
            profile=profile.model_dump(mode="json"),
            citations=[item.model_dump(mode="json") for item in used_citations],
        )
        return profile, used_citations

    @staticmethod
    def _expand_retrieval_query(question: str) -> str:
        stripped = question.strip()
        architecture_terms = (
            "cnn",
            "卷积神经网络",
            "架构类型",
            "网络类型",
            "骨干网络",
            "backbone",
            "transformer",
            "mamba",
        )
        if any(term in stripped.casefold() for term in architecture_terms):
            return (
                f"{stripped}\n重点检索：整体网络架构、encoder、decoder、backbone、"
                "CNN、convolution、卷积层、Transformer、self-attention、Mamba、"
                "混合架构，以及论文对模型组成的明确描述。"
            )
        experiment_terms = ("实验", "结果", "性能", "效果", "指标", "对比")
        if any(term in stripped for term in experiment_terms):
            return (
                f"{stripped}\n重点检索：实验设置、数据集、评价指标、对比基线、"
                "定量结果、消融实验、模型效率、计算量、模型大小、推理速度、局限。"
            )
        research_terms = (
            "研究什么",
            "研究内容",
            "研究问题",
            "研究动机",
            "解决什么",
            "核心目标",
            "主要贡献",
            "核心方法",
            "方法",
            "模型结构",
            "网络结构",
            "技术路线",
            "摘要",
        )
        if any(term in stripped for term in research_terms):
            return (
                f"{stripped}\n重点检索：论文摘要 Abstract、研究动机、现有方法局限、"
                "要解决的问题、本文提出的方法、核心贡献、整体模型架构。"
            )
        return stripped

    @staticmethod
    def _rerank_retrieved(
        query: str, hits: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        lowered_query = query.casefold()
        research_intent = any(
            marker in lowered_query
            for marker in ("abstract", "研究动机", "研究问题", "核心贡献", "模型架构")
        )
        experiment_intent = any(
            marker in lowered_query
            for marker in ("实验设置", "定量结果", "评价指标", "消融实验")
        )
        architecture_intent = any(
            marker in lowered_query
            for marker in ("cnn", "convolution", "backbone", "self-attention", "mamba")
        )
        reranked: list[tuple[float, RetrievedChunk]] = []
        for hit in hits:
            text = hit.chunk.text
            lowered = text.casefold()
            adjusted = float(hit.score or 0.0)
            first_part = lowered[:500]
            year_count = len(re.findall(r"\b(?:19|20)\d{2}[a-z]?\b", lowered))
            citation_count = lowered.count(" et al.") + lowered.count("arxiv")
            looks_like_references = (
                "references" in first_part
                or (year_count >= 4 and citation_count >= 2)
                or (year_count >= 6 and hit.chunk.page_number > 3)
            )
            if looks_like_references:
                adjusted -= 0.75
            if research_intent:
                if hit.chunk.page_number == 1:
                    adjusted += 0.38
                elif hit.chunk.page_number <= 3:
                    adjusted += 0.2
                if "abstract" in first_part:
                    adjusted += 0.3
                if any(
                    marker in lowered
                    for marker in (
                        "to address",
                        "we propose",
                        "we present",
                        "本文提出",
                        "为了解决",
                        "our contribution",
                    )
                ):
                    adjusted += 0.3
                if "introduction" in first_part:
                    adjusted += 0.12
            if experiment_intent and any(
                    marker in lowered
                    for marker in (
                        "experimental results",
                        "experiments demonstrate",
                        "table ",
                        "dataset",
                        "dsc",
                        "dice",
                        "hd95",
                    )
                ):
                adjusted += 0.22
            if architecture_intent:
                architecture_markers = (
                    "architecture",
                    "encoder",
                    "decoder",
                    "backbone",
                    "convolution",
                    "cnn",
                    "self-attention",
                    "transformer",
                    "mamba",
                    "network consists",
                    "we design",
                    "we propose",
                )
                marker_hits = sum(marker in lowered for marker in architecture_markers)
                adjusted += min(marker_hits, 4) * 0.09
            reranked.append((adjusted, hit.model_copy(update={"score": adjusted})))
        reranked.sort(key=lambda item: item[0], reverse=True)
        return [item for _, item in reranked]

    @staticmethod
    def _build_evidence(
        retrieved: list[RetrievedChunk], *, start: int = 1
    ) -> tuple[list[Citation], str]:
        citations: list[Citation] = []
        blocks: list[str] = []
        for index, item in enumerate(retrieved, start=start):
            citation_id = f"C{index}"
            chunk = item.chunk
            citation = Citation(
                citation_id=citation_id,
                paper_id=chunk.paper_id,
                paper_title=chunk.paper_title,
                paper_version=chunk.paper_version,
                pdf_page=chunk.page_number,
                chunk_id=chunk.chunk_id,
                evidence_text=chunk.text,
                retrieval_score=item.score,
            )
            citations.append(citation)
            blocks.append(
                f"[{citation_id}] 论文={chunk.paper_title}; paper_id={chunk.paper_id}; "
                f"version={chunk.paper_version}; PDF页={chunk.page_number}\n"
                f"{chunk.text}"
            )
        return citations, "\n\n".join(blocks)

    @staticmethod
    def _validate_answer_draft(
        draft: GroundedAnswerDraft, citations: list[Citation]
    ) -> GroundedAnswerDraft:
        valid = {citation.citation_id for citation in citations}
        aliases = {
            alias: citation.citation_id
            for citation in citations
            for alias in (citation.citation_id, citation.chunk_id)
        }

        def normalize(value: str) -> str:
            stripped = value.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                stripped = stripped[1:-1].strip()
            return aliases.get(stripped, stripped)

        normalized_ids = [normalize(item) for item in draft.used_citation_ids]
        answer = draft.answer
        for citation in citations:
            answer = answer.replace(
                f"[{citation.chunk_id}]", f"[{citation.citation_id}]"
            )
        # Compatible models occasionally decorate a supplied ID as ``C3a``.
        # If the numeric base exists, map it back to that real evidence block.
        def normalize_inline_alias(match: re.Match) -> str:
            raw = match.group(1).upper()
            base_match = re.match(r"^([CF]\d+)", raw)
            base = base_match.group(1) if base_match else raw
            return f"[{base}]" if base in valid else f"[{raw}]"

        answer = re.sub(
            r"\[([CF]\d+[A-Za-z]+)\]",
            normalize_inline_alias,
            answer,
            flags=re.IGNORECASE,
        )
        invalid = [item for item in normalized_ids if item not in valid]
        inline = set(re.findall(r"\[([CF]\d+)\]", answer))
        invalid.extend(sorted(inline - valid))
        if invalid:
            raise ResearchCopilotError(f"模型返回非法引用 ID：{sorted(set(invalid))}")
        used = list(OrderedDict.fromkeys(normalized_ids + sorted(inline)))
        if not used and not draft.insufficient_evidence:
            return draft.model_copy(
                update={
                    "answer": "当前证据不足，不能给出可验证的确定性回答。",
                    "used_citation_ids": [],
                    "insufficient_evidence": True,
                    "limitations": draft.limitations + ["模型回答未引用任何检索证据"],
                }
            )
        return draft.model_copy(update={"answer": answer, "used_citation_ids": used})

    @staticmethod
    def _parse_grounded_response(response) -> GroundedAnswerDraft:
        """Parse a plain model response and leave citation enforcement local.

        DashScope's OpenAI-compatible structured-output endpoint can spend a
        long time compiling response grammars.  The answer itself therefore
        remains normal text; this small status footer is parsed locally and all
        referenced IDs still pass through ``_validate_answer_draft``.
        """
        content = getattr(response, "content", response)
        if isinstance(content, list):
            text = "".join(
                str(item.get("text", "")) if isinstance(item, dict) else str(item)
                for item in content
            )
        else:
            text = str(content or "")
        marker = "<EVIDENCE_STATUS>"
        answer, separator, footer = text.partition(marker)
        answer = answer.strip()
        insufficient = False
        limitations: list[str] = []
        if separator:
            footer = footer.split("</EVIDENCE_STATUS>", 1)[0]
            match = re.search(
                r"insufficient_evidence\s*:\s*(true|false)", footer, re.IGNORECASE
            )
            insufficient = bool(match and match.group(1).lower() == "true")
            limitations_section = footer.split("limitations:", 1)
            if len(limitations_section) == 2:
                limitations = [
                    line.lstrip("-• ").strip()
                    for line in limitations_section[1].splitlines()
                    if line.strip().startswith(("-", "•"))
                    and line.lstrip("-• ").strip()
                ][:6]
        if not answer:
            answer = "模型未生成可用回答。"
            insufficient = True
            limitations.append("模型返回内容为空")
        used_ids = list(
            OrderedDict.fromkeys(
                re.findall(r"\[([CF]\d+)\]", answer, re.IGNORECASE)
            )
        )
        used_ids = [item.upper() for item in used_ids]
        return GroundedAnswerDraft(
            answer=answer,
            used_citation_ids=used_ids,
            insufficient_evidence=insufficient,
            limitations=limitations,
        )

    def _build_visual_evidence(
        self,
        question: str,
        retrieved: list[RetrievedChunk],
        papers: dict[str, dict],
    ) -> tuple[list[Citation], list[dict]]:
        limit = self._visual_page_limit(question)
        if not self.settings.multimodal_enabled or not limit:
            return [], []

        candidates: list[tuple[str, int, int, str]] = []
        seen: set[tuple[str, int, int]] = set()
        # Give each paper one page first, then fill by retrieval rank.
        for paper_id in papers:
            for item in retrieved:
                chunk = item.chunk
                key = (chunk.paper_id, chunk.paper_version, chunk.page_number)
                if chunk.paper_id == paper_id and key not in seen:
                    candidates.append((*key, chunk.paper_title))
                    seen.add(key)
                    break
        for item in retrieved:
            chunk = item.chunk
            key = (chunk.paper_id, chunk.paper_version, chunk.page_number)
            if key not in seen:
                candidates.append((*key, chunk.paper_title))
                seen.add(key)

        citations: list[Citation] = []
        content_parts: list[dict] = []
        for paper_id, version_number, page_number, title in candidates:
            version = self.repository.get_version(paper_id, version_number)
            if version is None:
                continue
            parsed_dir = Path(version["parsed_dir"]).resolve()
            image_path = (
                parsed_dir / "page_images" / f"page_{page_number:04d}.jpg"
            ).resolve()
            if image_path.parent != (parsed_dir / "page_images").resolve():
                continue
            if not image_path.is_file():
                continue
            citation_id = f"F{len(citations) + 1}"
            citation = Citation(
                citation_id=citation_id,
                paper_id=paper_id,
                paper_title=title,
                paper_version=version_number,
                pdf_page=page_number,
                chunk_id=(
                    f"{paper_id}:v{version_number}:p{page_number:04d}:page-image"
                ),
                evidence_text=(
                    f"{title} PDF 第 {page_number} 页的完整页面图像；"
                    "包含该页图、表、公式、图注和版面关系。"
                ),
                evidence_type="page_image",
                image_path=str(image_path),
            )
            citations.append(citation)
            content_parts.extend(
                [
                    {
                        "type": "text",
                        "text": (
                            f"[{citation_id}] 视觉证据：论文={title}; "
                            f"paper_id={paper_id}; version={version_number}; "
                            f"PDF页={page_number}。紧随其后的图片属于该引用。"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,"
                            + base64.b64encode(image_path.read_bytes()).decode("ascii"),
                            "detail": "high",
                        },
                    },
                ]
            )
            if len(citations) >= limit:
                break
        return citations, content_parts

    def _visual_page_limit(self, question: str) -> int:
        lowered = question.lower()
        explicit = (
            "图中",
            "图片",
            "插图",
            "架构图",
            "流程图",
            "示意图",
            "可视化",
            "曲线",
            "热力图",
            "figure",
            "fig.",
            "diagram",
            "visualization",
            "heatmap",
            "table",
            "表格",
        )
        method = (
            "方法",
            "架构",
            "结构",
            "机制",
            "模块",
            "流程",
            "moe",
            "method",
            "architecture",
            "mechanism",
            "module",
            "pipeline",
        )
        if any(marker in lowered for marker in explicit):
            return self.settings.max_visual_pages
        if any(marker in lowered for marker in method):
            return min(2, self.settings.max_visual_pages)
        return 0

    def _invoke_answer_model(self, messages, progress_callback):
        now = time.monotonic()
        fallback_name = self.settings.fallback_chat_model or "备用模型"
        if self.fallback_chat_model and now < self._primary_unavailable_until:
            remaining = max(1, int(self._primary_unavailable_until - now))
            progress_callback(
                f"{self.settings.chat_model} 仍在熔断保护期（约 {remaining} 秒），"
                f"本轮直接使用 {fallback_name}，不再等待超时。"
            )
            return (
                self.fallback_chat_model.invoke(self._text_only_messages(messages)),
                fallback_name,
                True,
            )
        try:
            return self.chat_model.invoke(messages), self.settings.chat_model, False
        except Exception as exc:
            if not self.fallback_chat_model or not self._is_model_availability_error(exc):
                raise
            self._primary_unavailable_until = (
                time.monotonic() + self.settings.model_circuit_breaker_seconds
            )
            progress_callback(
                f"{self.settings.chat_model} 暂时不可用（{type(exc).__name__}），"
                f"正在切换备用模型 {fallback_name} 继续生成……"
            )
            return (
                self.fallback_chat_model.invoke(self._text_only_messages(messages)),
                fallback_name,
                True,
            )

    @staticmethod
    def _text_only_messages(messages):
        cleaned = []
        for message in messages:
            content = message.content
            if isinstance(content, list):
                text_parts = [
                    str(part.get("text", ""))
                    for part in content
                    if isinstance(part, dict)
                    and part.get("type") == "text"
                    and not str(part.get("text", "")).lstrip().startswith("[F")
                ]
                content = "\n".join(part for part in text_parts if part)
            cleaned.append(message.model_copy(update={"content": content}))
        return cleaned

    @staticmethod
    def _is_model_availability_error(exc: Exception) -> bool:
        if isinstance(exc, (TimeoutError, ConnectionError)):
            return True
        if type(exc).__name__ in {
            "APITimeoutError",
            "APIConnectionError",
            "ReadTimeout",
            "ConnectTimeout",
        }:
            return True
        status_code = getattr(exc, "status_code", None)
        return isinstance(status_code, int) and status_code >= 500

    @staticmethod
    def _validate_profile(
        profile: PaperProfileDraft, citations: list[Citation]
    ) -> PaperProfileDraft:
        valid = {citation.citation_id for citation in citations}
        updates = {}
        for dimension in DIMENSIONS:
            value: EvidenceValue = getattr(profile, dimension)
            cleaned = [item for item in OrderedDict.fromkeys(value.citation_ids) if item in valid]
            if not cleaned and not value.insufficient_evidence:
                value = value.model_copy(
                    update={"value": "证据不足", "insufficient_evidence": True}
                )
            updates[dimension] = value.model_copy(update={"citation_ids": cleaned})
        return profile.model_copy(update=updates)

    def _save_trace(
        self,
        trace_id: str,
        thread_id: str | None,
        question: str,
        papers: dict[str, dict],
        retrieved_ids: list[str],
        used_ids: list[str],
        standalone_query: str | None = None,
        prompt_version: str = QA_PROMPT_VERSION,
    ) -> None:
        self.repository.save_retrieval_trace(
            trace_id=trace_id,
            thread_id=thread_id,
            question=question,
            standalone_query=standalone_query or question,
            paper_ids=list(papers),
            paper_versions={key: int(value["active_version"]) for key, value in papers.items()},
            retrieved_chunk_ids=retrieved_ids,
            used_chunk_ids=used_ids,
            prompt_version=prompt_version,
        )
