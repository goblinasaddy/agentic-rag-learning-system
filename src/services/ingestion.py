from pathlib import Path
from uuid import uuid4
from qdrant_client import models
import time
from src.infrastructure.db.registry import DocumentRegistry

from src.domain.documents.parser import DocumentParser
from src.domain.documents.models import DocumentMetadata, ChunkMetadata
from src.domain.documents.chunking.factory import ChunkerFactory, ChunkerConfig
from src.infrastructure.db.qdrant import QdrantHandler
from src.infrastructure.llm.embeddings import EmbeddingService

from typing import Optional, Tuple
import hashlib
from src.infrastructure.db.registry import DocumentRegistry

class IngestionService:
    def __init__(self, qdrant_handler: Optional[QdrantHandler] = None, registry: Optional[DocumentRegistry] = None):
        self.parser = DocumentParser()
        self.chunker_factory = ChunkerFactory()
        self.qdrant = qdrant_handler or QdrantHandler()
        self.registry = registry or DocumentRegistry()
        self.embedding_service = EmbeddingService()

        
        # Ensure DB is ready
        self.qdrant.create_collection_if_not_exists()

    def _compute_hash(self, valid_file_path: Path) -> str:
        sha256_hash = hashlib.sha256()
        with open(valid_file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    async def ingest_file(self, file_path: Path, strategy: str = "semantic") -> Optional[DocumentMetadata]:
        """
        Ingests a file with Version Control:
        1. Check Registry (Hash Check).
        2. If New/Updated:
           - Deprecate old chunks (is_latest=False).
           - Parse & Upsert new chunks (is_latest=True).
           - Update Registry.
        """
        # 0. Hash & Registry Check
        new_hash = self._compute_hash(file_path)
        filename = file_path.name
        
        record = self.registry.get_by_filename(filename)
        
        if record and record.content_hash == new_hash:
            # Unchanged
            print(f"Skipping {filename}: Unchanged.")
            return None
            
        # Determine Version
        version = 1
        logical_id = None
        
        if record:
            version = record.current_version + 1
            logical_id = record.logical_id
            print(f"Updating {filename} to Version {version}...")
            
            # PHASE A: DEPRECATE OLD VERSION
            # We assume qdrant_handler has a method to bulk-update payload
            # For now, we will perform a 'Delete' of 'is_latest' by ... actually we need a method updates payload.
            # Simpler approach for prototype: We just upsert the NEW ones as latest.
            # Queries dealing with old ones might need 'version' filter.
            # To do this correctly:
            await self.qdrant.mark_as_outdated(logical_id) 
        
        # 1. Parse
        content, doc_metadata = await self.parser.parse(file_path)
        # Override doc_id with logical_id if exists, else keep parser's or generate new
        final_doc_id = logical_id or str(uuid4()) # Use stable ID
        
        # 2. Chunk
        config = ChunkerConfig(strategy=strategy) # type: ignore
        chunker = self.chunker_factory.get_chunker(config)
        chunks_metadata = chunker.chunk(content, final_doc_id)
        
        # 3. Prepare Points logic ...
        points = []
        texts_to_embed = []
        payloads = []
        
        for chunk_meta in chunks_metadata:
            text = content[chunk_meta.start_char_idx : chunk_meta.end_char_idx]
            texts_to_embed.append(text)
            
            payload = {
                "logical_doc_id": final_doc_id,
                "chunk_id": str(chunk_meta.chunk_id),
                "content": text,
                "filename": filename,
                "version_number": version,
                "is_latest": True, # Always true for new ingestion
                "effective_date": "2024-01-01", # Placeholder for extraction logic
                "ingestion_timestamp": time.time()
            }
            payloads.append(payload)
            
        vectors = self.embedding_service.embed_documents(texts_to_embed)
        
        for i, vector in enumerate(vectors):
            points.append(models.PointStruct(
                id=str(uuid4()), 
                vector=vector,
                payload=payloads[i]
            ))
            
        if points:
            await self.qdrant.upsert_points(points)
            
            # 4. Update Registry
            self.registry.upsert_document(filename, new_hash, final_doc_id, version)
            
        return doc_metadata
