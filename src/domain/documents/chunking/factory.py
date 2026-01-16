from src.domain.documents.chunking.base import BaseChunker, ChunkerConfig
from src.domain.documents.chunking.strategies import FixedSizeChunker, RecursiveChunker
from src.domain.documents.chunking.advanced_strategies import SemanticChunker, MarkdownChunker

class ChunkerFactory:
    @staticmethod
    def get_chunker(config: ChunkerConfig) -> BaseChunker:
        if config.strategy == "fixed":
            return FixedSizeChunker(config)
        elif config.strategy == "recursive":
            return RecursiveChunker(config)
        elif config.strategy == "markdown":
            return MarkdownChunker(config)
        elif config.strategy == "semantic":
            return SemanticChunker(config)
        else:
            raise ValueError(f"Unknown chunking strategy: {config.strategy}")
