from uuid import UUID, uuid4
from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field

class DocumentMetadata(BaseModel):
    doc_id: UUID = Field(default_factory=uuid4)
    filename: str
    file_type: Literal['pdf', 'docx', 'txt']
    upload_time: datetime = Field(default_factory=datetime.now)
    page_count: int = 0
    content_hash: str 

class ChunkMetadata(BaseModel):
    chunk_id: UUID = Field(default_factory=uuid4)
    doc_id: UUID
    # content: str # Content is distinct from metadata in some designs, but often useful to keep together or separate in VectorDB payload. 
                 # The plan says "Retrieval must return both content and metadata", so we might map this to VectorDB payload.
                 # For the pure metadata model, we'll keep it metadata-only as requested.
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    start_char_idx: int
    end_char_idx: int
