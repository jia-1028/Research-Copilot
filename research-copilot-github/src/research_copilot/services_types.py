from __future__ import annotations

from typing import Protocol

from research_copilot.config import Settings
from research_copilot.storage import SQLiteRepository
from research_copilot.vector_index import VectorIndex


class CoreServices(Protocol):
    settings: Settings
    repository: SQLiteRepository
    vector_index: VectorIndex
