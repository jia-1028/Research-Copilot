from __future__ import annotations

from research_copilot.presentation import answer_text


def test_current_tool_content_is_unchanged() -> None:
    assert answer_text("正常回答 [C1]", {"answer": "正常回答 [C1]"}) == "正常回答 [C1]"


def test_legacy_json_tool_content_is_unwrapped() -> None:
    content = '{"answer":"基础模型是 Swin-Unet [C1]","citations":[{"citation_id":"C1"}]}'
    assert answer_text(content) == "基础模型是 Swin-Unet [C1]"


def test_dict_tool_content_is_unwrapped() -> None:
    assert answer_text({"answer": "使用 TransUNet [C2]"}) == "使用 TransUNet [C2]"
