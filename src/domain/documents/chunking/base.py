from abc import ABC, abstractmethod
from typing import List, Literal, Optional
from pydantic import BaseModel
from src.domain.documents.models import ChunkMetadata

class ChunkerConfig(BaseModel):
    strategy: Literal["fixed", "semantic", "recursive", "markdown"] = "semantic"
    chunk_size: int = 512
    overlap: int = 50
    # Semantic specific
    embedding_model: str = "all-MiniLM-L6-v2"
    breakpoint_threshold_amount: int = 95

class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str, doc_id: str) -> List[ChunkMetadata]:
        pass
