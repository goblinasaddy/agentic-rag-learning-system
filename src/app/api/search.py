from typing import List, Optional
from fastapi import APIRouter, Query, HTTPException
from src.services.retrieval import Accountant, Retriever, RetrievalResult # Wait, Accountant? Bad copy paste potential. 
# Fixed import
from src.services.retrieval import Retriever, RetrievalResult

router = APIRouter()

@router.post("/", response_model=List[RetrievalResult])
async def search_documents(
    query: str = Query(..., min_length=1),
    top_k: int = Query(5, ge=1, le=20),
    threshold: Optional[float] = Query(0.1, ge=0.0, le=1.0)
):
    """
    Search for documents using semantic similarity.
    """
    try:
        retriever = Retriever()
        results = await retriever.retrieve(query, top_k=top_k, score_threshold=threshold)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")
