from __future__ import annotations

import json
from typing import Any

from research_copilot.models import PaperComparison


def conversation_markdown(
    conversation: dict[str, Any], messages: list[dict[str, Any]]
) -> str:
    lines = [f"# {conversation['title']}", ""]
    snapshots = conversation.get("paper_snapshots") or []
    if snapshots:
        lines.extend(
            [
                "## 论文范围",
                "",
                *[
                    f"- {item['paper_title_snapshot']} (v{item['paper_version_snapshot']})"
                    for item in snapshots
                ],
                "",
            ]
        )
    for message in messages:
        role = "用户" if message["role"] == "user" else "Research Copilot"
        lines.extend([f"## {role}", "", message.get("content") or "（无正文）", ""])
        payload = message.get("payload") or {}
        citations = payload.get("citations") or []
        if citations:
            lines.extend(["### PDF 证据", ""])
            for citation in citations:
                lines.extend(
                    [
                        (
                            f"- **[{citation['citation_id']}] {citation['paper_title']}，"
                            f"PDF 第 {citation['pdf_page']} 页**"
                        ),
                        f"  - chunk: `{citation['chunk_id']}`",
                        f"  - {citation['evidence_text'].strip()}",
                    ]
                )
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def conversation_json(
    conversation: dict[str, Any], messages: list[dict[str, Any]]
) -> str:
    return json.dumps(
        {"conversation": conversation, "messages": messages},
        ensure_ascii=False,
        indent=2,
        default=str,
    )


def comparison_markdown(comparison: PaperComparison) -> str:
    lines = ["# 多论文证据比较", "", "论文：" + "、".join(comparison.paper_ids), ""]
    for row in comparison.rows:
        lines.extend([f"## {row.dimension}", ""])
        for paper_id, value in row.values.items():
            citations = " ".join(f"[{item}]" for item in value.citation_ids)
            missing = "（证据不足）" if value.insufficient_evidence else ""
            lines.append(f"- **{paper_id}**：{value.value} {citations}{missing}".rstrip())
        lines.append("")
    if comparison.similarities:
        lines.extend(["## 相似点", "", *[f"- {item}" for item in comparison.similarities], ""])
    if comparison.differences:
        lines.extend(["## 差异", "", *[f"- {item}" for item in comparison.differences], ""])
    if comparison.non_comparable_items:
        lines.extend(
            [
                "## 不可直接比较项",
                "",
                *[f"- {item}" for item in comparison.non_comparable_items],
                "",
            ]
        )
    if comparison.citations:
        lines.extend(["## PDF 证据", ""])
        for citation in comparison.citations:
            lines.extend(
                [
                    (
                        f"- **[{citation.citation_id}] {citation.paper_title}，"
                        f"PDF 第 {citation.pdf_page} 页**"
                    ),
                    f"  - chunk: `{citation.chunk_id}`",
                    f"  - {citation.evidence_text.strip()}",
                ]
            )
    return "\n".join(lines).strip() + "\n"
