import asyncio
import sys
import shutil
import time
from pathlib import Path
import tempfile

sys.path.append(str(Path(__file__).parent.parent))

from src.services.ingestion import IngestionService
from src.infrastructure.db.registry import DocumentRegistry
from src.infrastructure.db.qdrant import QdrantHandler

async def main():
    print("🚀 Verifying Versioning Extension (In-Memory)...")
    
    # Setup
    shared_qdrant = QdrantHandler(use_memory=True)
    shared_qdrant.create_collection_if_not_exists()
    
    # Use tmp db path provided by tempfile but handle closure explicitly
    tmp_db_path = Path(tempfile.gettempdir()) / f"test_registry_{int(time.time())}.db"
    
    try:
        registry = DocumentRegistry(db_path=str(tmp_db_path))
        
        # Ingestor Service with mocked components
        ingestor = IngestionService(qdrant_handler=shared_qdrant, registry=registry)
        
        # Test Data
        filename = "Policy.txt"
        
        # 1. Ingest V1
        # 1. Ingest V1
        print("\n[Action] Ingesting V1...")
        v1_content = "Rule 1: No running in the halls."
        target_path = Path(tempfile.gettempdir()) / filename
        
        with open(target_path, "w") as f:
            f.write(v1_content)
            
        await ingestor.ingest_file(target_path)
        
        # Check Registry
        rec1 = registry.get_by_filename(filename)
        print(f"Registry: Ver {rec1.current_version}, ID: {rec1.logical_id}")
        assert rec1.current_version == 1
        
        # 2. Ingest V1 Again (No Change)
        print("\n[Action] Ingesting V1 Again (Should Skip)...")
        await ingestor.ingest_file(target_path)
        rec1_b = registry.get_by_filename(filename)
        print(f"Registry: Ver {rec1_b.current_version}")
        assert rec1_b.current_version == 1
        assert rec1_b.updated_at == rec1.updated_at
        
        # 3. Ingest V2 (Change Content)
        print("\n[Action] Ingesting V2 (Modified)...")
        v2_content = "Rule 1: Running is allowed on Fridays."
        with open(target_path, "w") as f:
            f.write(v2_content)
            
        await ingestor.ingest_file(target_path)
        
        # Check Registry
        rec2 = registry.get_by_filename(filename)
        print(f"Registry: Ver {rec2.current_version}, ID: {rec2.logical_id}")
        
        assert rec2.current_version == 2
        assert rec2.logical_id == rec1.logical_id # Same Logical ID
        assert rec2.content_hash != rec1.content_hash
        
        print("\n✅ PASSED: Versioning Logic Verified.")
        
    finally:
        # Cleanup
        if target_path.exists():
            target_path.unlink()
        if tmp_db_path.exists():
            tmp_db_path.unlink()

if __name__ == "__main__":
    asyncio.run(main())
