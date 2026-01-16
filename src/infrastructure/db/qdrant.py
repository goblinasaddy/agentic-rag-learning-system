from typing import List, Dict, Any, Optional
from uuid import UUID
from qdrant_client import QdrantClient,models
from src.app.core.config import settings

class QdrantHandler:
    def __init__(self, use_memory: bool = False):
        if use_memory:
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                api_key=settings.QDRANT_API_KEY
            )
        self.collection_name = "documents"
        self.vector_size = 384 # Default for all-MiniLM-L6-v2. Configurable? 

    def create_collection_if_not_exists(self):
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                )
            )

    async def upsert_points(self, points: List[models.PointStruct]):
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    async def mark_as_outdated(self, logical_doc_id: str):
        """
        Updates all chunks of a doc to is_latest=False.
        """
        # 1. Filter by logical_doc_id
        filter_query = models.Filter(
            must=[
                models.FieldCondition(
                    key="logical_doc_id",
                    match=models.MatchValue(value=logical_doc_id)
                )
            ]
        )
        
        # 2. Update Payload
        self.client.set_payload(
            collection_name=self.collection_name,
            payload={"is_latest": False},
            points=filter_query
        )

    async def search(self, query_vector: List[float], limit: int = 5, score_threshold: Optional[float] = None) -> List[models.ScoredPoint]:
        # Refactored to use query_points as search seems unavailable in this environment
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            score_threshold=score_threshold
        )
        return response.points
