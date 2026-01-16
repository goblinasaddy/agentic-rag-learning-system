import shutil
import tempfile
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, status
from src.services.ingestion import IngestionService
from src.domain.documents.models import DocumentMetadata
from src.domain.documents.exceptions import UnsupportedFileTypeError, ParsingError

router = APIRouter()

@router.post("/", response_model=DocumentMetadata, status_code=status.HTTP_201_CREATED)
async def ingest_document(
    file: UploadFile = File(...),
    strategy: str = Query("semantic", description="Chunking strategy: semantic, fixed, recursive, markdown")
):
    """
    Uploads and ingests a document into the RAG system.
    """
    service = IngestionService()
    
    # Create temp file
    # In production, might want to stream or use specific volume
    suffix = Path(file.filename).suffix if file.filename else ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
        
    try:
        metadata = await service.ingest_file(tmp_path, strategy=strategy)
        # Update filename to original name as tmp name is random
        metadata.filename = file.filename or "unknown" 
        return metadata
        
    except UnsupportedFileTypeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ParsingError as e:
        raise HTTPException(status_code=422, detail=f"Parsing failed: {str(e)}")
    except Exception as e:
        # Log error here
        raise HTTPException(status_code=500, detail=f"Internal processing error: {str(e)}")
    finally:
        # Cleanup
        if tmp_path.exists():
            tmp_path.unlink()
