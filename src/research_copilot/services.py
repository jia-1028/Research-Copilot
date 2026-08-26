from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.sqlite import SqliteSaver

from research_copilot.arxiv_service import ArxivService
from research_copilot.background_tasks import BackgroundTaskService
from research_copilot.config import Settings, get_settings
from research_copilot.conversation_memory import ConversationMemoryService
from research_copilot.deep_analysis import DeepAnalysisService
from research_copilot.ingestion import PaperIngestionService
from research_copilot.library import PaperLibraryService
from research_copilot.model_factory import (
    create_chat_model,
    create_embedding_model,
    create_fallback_chat_model,
)
from research_copilot.rag import PaperRAGService
from research_copilot.storage import SQLiteRepository
from research_copilot.vector_index import ChromaVectorIndex


@dataclass
class ServiceContainer:
    settings: Settings
    repository: SQLiteRepository
    vector_index: ChromaVectorIndex
    ingestion: PaperIngestionService
    arxiv: ArxivService
    rag: PaperRAGService
    deep_analysis: DeepAnalysisService = field(init=False)
    library: PaperLibraryService = field(init=False)
    memory: ConversationMemoryService = field(init=False)
    tasks: BackgroundTaskService = field(init=False)
    chat_model: BaseChatModel
    checkpointer: SqliteSaver
    checkpoint_connection: sqlite3.Connection

    def close(self) -> None:
        self.tasks.close()
        self.checkpoint_connection.close()


def build_services(
    settings: Settings | None = None,
    *,
    chat_model: BaseChatModel | None = None,
    fallback_chat_model: BaseChatModel | None = None,
    embedding_model: Embeddings | None = None,
) -> ServiceContainer:
    settings = settings or get_settings()
    settings.ensure_directories()
    chat_model = chat_model or create_chat_model(settings)
    fallback_chat_model = fallback_chat_model or create_fallback_chat_model(settings)
    embedding_model = embedding_model or create_embedding_model(settings)
    repository = SQLiteRepository(
        settings.sqlite_path, checkpoint_path=settings.checkpoint_path
    )
    vector_index = ChromaVectorIndex(settings, embedding_model)
    ingestion = PaperIngestionService(settings, repository, vector_index)
    arxiv_service = ArxivService(settings, repository, ingestion)
    rag = PaperRAGService(
        settings, repository, vector_index, chat_model, fallback_chat_model
    )
    checkpoint_connection = sqlite3.connect(
        settings.checkpoint_path, check_same_thread=False, timeout=30
    )
    checkpointer = SqliteSaver(checkpoint_connection)
    checkpointer.setup()
    container = ServiceContainer(
        settings=settings,
        repository=repository,
        vector_index=vector_index,
        ingestion=ingestion,
        arxiv=arxiv_service,
        rag=rag,
        chat_model=chat_model,
        checkpointer=checkpointer,
        checkpoint_connection=checkpoint_connection,
    )
    container.library = PaperLibraryService(container)
    structured_chat_model = fallback_chat_model or chat_model
    container.deep_analysis = DeepAnalysisService(rag, structured_chat_model)
    container.memory = ConversationMemoryService(
        repository, structured_chat_model, checkpointer
    )
    container.memory.import_legacy_checkpoints()
    container.tasks = BackgroundTaskService(repository, ingestion, arxiv_service, rag)
    return container
