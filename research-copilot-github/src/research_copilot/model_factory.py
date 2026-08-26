from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain_community.embeddings import DashScopeEmbeddings

from research_copilot.config import Settings, get_settings
from research_copilot.errors import ConfigurationError


def create_chat_model(settings: Settings | None = None):
    settings = settings or get_settings()
    return _create_chat_model(
        settings,
        settings.chat_provider,
        settings.chat_model,
        settings.model_timeout_seconds,
    )


def create_fallback_chat_model(settings: Settings | None = None):
    settings = settings or get_settings()
    if not settings.fallback_chat_model or (
        settings.fallback_chat_provider == settings.chat_provider
        and settings.fallback_chat_model == settings.chat_model
    ):
        return None
    return _create_chat_model(
        settings,
        settings.fallback_chat_provider,
        settings.fallback_chat_model,
        settings.fallback_model_timeout_seconds,
    )


def _create_chat_model(
    settings: Settings, provider: str, model_name: str, timeout_seconds: int
):
    if provider == "deepseek":
        if settings.deepseek_api_key is None:
            raise ConfigurationError("CHAT_PROVIDER=deepseek 时必须配置 DEEPSEEK_API_KEY")
        api_key = settings.deepseek_api_key.get_secret_value()
        base_url = settings.deepseek_base_url
    else:
        api_key = settings.dashscope_api_key.get_secret_value()
        base_url = settings.dashscope_base_url
    model_kwargs = {
        "model": f"openai:{model_name}",
        "api_key": api_key,
        "base_url": base_url,
        "timeout": timeout_seconds,
        "max_tokens": settings.model_max_output_tokens,
        # ChatOpenAI retries twice by default.  A nested RAG call would therefore
        # wait up to three full timeout windows while the UI appears frozen.
        # Agent-level transient retries are already handled by middleware; tools
        # deliberately fail once and surface a retry button to the user.
        "max_retries": 0,
    }
    if settings.disable_model_thinking:
        # DashScope's DeepSeek-compatible endpoint otherwise may spend most of
        # the request budget on hidden reasoning before returning any answer.
        model_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    return init_chat_model(
        **model_kwargs,
    )


def create_embedding_model(settings: Settings | None = None) -> DashScopeEmbeddings:
    settings = settings or get_settings()
    return DashScopeEmbeddings(
        model=settings.embedding_model,
        dashscope_api_key=settings.dashscope_api_key.get_secret_value(),
    )
