from __future__ import annotations

import pytest
from pydantic import SecretStr

from research_copilot import model_factory
from research_copilot.errors import ConfigurationError


def test_chat_model_disables_thinking_and_caps_latency(monkeypatch, settings) -> None:
    captured: dict = {}

    def fake_init_chat_model(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(model_factory, "init_chat_model", fake_init_chat_model)

    model_factory.create_chat_model(settings)

    assert captured["timeout"] == 15
    assert captured["max_tokens"] == 1200
    assert captured["max_retries"] == 0
    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}


def test_chat_model_can_leave_thinking_to_provider(monkeypatch, settings) -> None:
    captured: dict = {}

    def fake_init_chat_model(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(model_factory, "init_chat_model", fake_init_chat_model)
    configured = settings.model_copy(update={"disable_model_thinking": False})

    model_factory.create_chat_model(configured)

    assert "extra_body" not in captured


def test_fallback_chat_model_uses_separate_name_and_timeout(monkeypatch, settings) -> None:
    captured: dict = {}

    def fake_init_chat_model(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(model_factory, "init_chat_model", fake_init_chat_model)

    model_factory.create_fallback_chat_model(settings)

    assert captured["model"] == "openai:qwen-plus"
    assert captured["timeout"] == 45
    assert captured["base_url"] == "https://example.test/v1"


def test_primary_chat_model_can_use_official_deepseek(monkeypatch, settings) -> None:
    captured: dict = {}

    def fake_init_chat_model(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(model_factory, "init_chat_model", fake_init_chat_model)
    configured = settings.model_copy(
        update={
            "chat_provider": "deepseek",
            "chat_model": "deepseek-v4-flash-vision-exp",
            "deepseek_api_key": SecretStr("deepseek-test-key"),
            "deepseek_base_url": "https://api.deepseek.com",
        }
    )

    model_factory.create_chat_model(configured)

    assert captured["model"] == "openai:deepseek-v4-flash-vision-exp"
    assert captured["api_key"] == "deepseek-test-key"
    assert captured["base_url"] == "https://api.deepseek.com"


def test_official_deepseek_provider_requires_its_own_key(settings) -> None:
    configured = settings.model_copy(
        update={"chat_provider": "deepseek", "deepseek_api_key": None}
    )

    with pytest.raises(ConfigurationError, match="DEEPSEEK_API_KEY"):
        model_factory.create_chat_model(configured)
