from typing import List
from sentence_transformers import SentenceTransformer
from src.app.core.config import settings

class EmbeddingService:
    def __init__(self, model_name: str = settings.EMBEDDING_MODEL):
        self.model_name = model_name
        # Lazy load in production, or load on startup. 
        # For now, load in init.
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # sentence-transformers encode returns numpy array or list of tensors
        # we convert to list of floats for JSON/Qdrant compatibility
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
