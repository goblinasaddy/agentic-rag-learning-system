from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from src.infrastructure.db.qdrant import QdrantHandler
from src.infrastructure.llm.embeddings import EmbeddingService

class RetrievalResult(BaseModel):
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]

class Retriever:
    def __init__(self, qdrant_handler: Optional[QdrantHandler] = None):
        self.qdrant = qdrant_handler or QdrantHandler()
        self.embedding_service = EmbeddingService()

    async def retrieve(
        self, 
        query: str, 
        top_k: int = 5, 
        score_threshold: Optional[float] = 0.2
    ) -> List[RetrievalResult]:
        """
        Retrieves relevant documents for a given query.
        """
        # 1. Embed Query
        query_vector = self.embedding_service.embed_query(query)
        
        # 2. Search Qdrant
        points = await self.qdrant.search(
            query_vector=query_vector, 
            limit=top_k, 
            score_threshold=score_threshold
        )
        
        # 3. Format Results
        results = []
        for point in points:
            # Safely get payload
            payload = point.payload or {}
            
            results.append(RetrievalResult(
                chunk_id=payload.get("chunk_id", ""),
                content=payload.get("content", ""),
                score=point.score,
                metadata=payload
            ))
            
        return results
