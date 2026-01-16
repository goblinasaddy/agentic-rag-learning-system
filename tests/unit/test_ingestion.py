import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from pathlib import Path
from src.services.ingestion.py import IngestionService
from src.domain.documents.models import DocumentMetadata, ChunkMetadata

@pytest.fixture
def mock_ingestion_components():
    with patch("src.services.ingestion.DocumentParser") as mock_parser, \
         patch("src.services.ingestion.ChunkerFactory") as mock_chunker_factory, \
         patch("src.services.ingestion.QdrantHandler") as mock_qdrant, \
         patch("src.services.ingestion.EmbeddingService") as mock_embedding:
         
        yield mock_parser, mock_chunker_factory, mock_qdrant, mock_embedding

@pytest.mark.asyncio
async def test_ingest_file_flow(mock_ingestion_components):
    mock_parser_cls, mock_chunker_cls, mock_qdrant_cls, mock_embed_cls = mock_ingestion_components
    
    # Setup Instances
    mock_parser = mock_parser_cls.return_value
    mock_qdrant = mock_qdrant_cls.return_value
    mock_embed = mock_embed_cls.return_value
    mock_chunker = MagicMock()
    mock_chunker_cls.return_value.get_chunker.return_value = mock_chunker

    # Setup Returns
    doc_id = uuid4()
    mock_parser.parse = AsyncMock(return_value=(
        "Hello World. This is a test.", 
        DocumentMetadata(filename="test.txt", file_type="txt", content_hash="abc", doc_id=doc_id)
    ))
    
    mock_chunker.chunk.return_value = [
        ChunkMetadata(doc_id=doc_id, start_char_idx=0, end_char_idx=11), # "Hello World"
        ChunkMetadata(doc_id=doc_id, start_char_idx=13, end_char_idx=27) # "This is a test"
    ]
    
    mock_embed.embed_documents.return_value = [[0.1]*384, [0.2]*384]
    
    # Run
    service = IngestionService()
    await service.ingest_file(Path("dummy.txt"))
    
    # Verify
    mock_parser.parse.assert_called_once()
    mock_chunker.chunk.assert_called_once()
    mock_embed.embed_documents.assert_called_once()
    mock_qdrant.upsert_points.assert_called_once()
    
    # Verify payload in upsert
    call_args = mock_qdrant.upsert_points.call_args[0][0]
    assert len(call_args) == 2
    assert call_args[0].payload['content'] == "Hello World"
