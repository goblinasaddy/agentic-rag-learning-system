import asyncio
import os
from pathlib import Path
from src.services.ingestion import IngestionService

SOURCE_DOCS_DIR = Path("data/source_docs")

async def background_ingestion_task():
    """
    Periodically checks for new/updated files and ingests them.
    """
    print("🚀 Background Ingestion Worker Started.")
    # Ensure dir exists
    SOURCE_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    
    ingestor = IngestionService()
    
    while True:
        try:
            # 1. List Files
            files = [f for f in SOURCE_DOCS_DIR.iterdir() if f.is_file()]
            
            for file_path in files:
                if file_path.suffix.lower() in [".pdf", ".docx", ".txt"]:
                    # Ingest Logic handles hashing & skipping
                    await ingestor.ingest_file(file_path)
            
        except Exception as e:
            print(f"⚠️ Background Worker Error: {e}")
            
        # Wait 60s
        await asyncio.sleep(60)
