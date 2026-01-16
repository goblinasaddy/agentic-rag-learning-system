import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.services.ingestion import IngestionService
from src.services.retrieval import Retriever
from src.domain.documents.models import DocumentMetadata

async def main():
    print("🚀 Starting Baseline Retrieval Evaluation (In-Memory)...")
    
    # 1. Setup Services (Patching QdrantHandler to use memory)
    # Ideally we'd dependency inject better, but for script we can patch class or use the updated init
    
    # We need to make sure IngestionService and Retriever share the SAME QdrantHandler instance 
    # if using in-memory, otherwise they point to different memory spaces? 
    # QdrantClient(":memory:") creates a new instance.
    # So we must share the instance.
    
    from src.infrastructure.db.qdrant import QdrantHandler
    shared_qdrant = QdrantHandler(use_memory=True)
    shared_qdrant.create_collection_if_not_exists()
    
    # Monkey patch the classes to return our shared instance
    # This is "hacky" but efficient for a script without full DI container
    from unittest.mock import MagicMock
    
    # Custom Ingestion Service that uses shared qdrant
    ingestor = IngestionService()
    ingestor.qdrant = shared_qdrant 
    
    # Custom Retriever that uses shared qdrant
    retriever = Retriever()
    retriever.qdrant = shared_qdrant
    
    # 2. Ingest Data
    documents = [
        ("Paris.txt", "Paris is the capital of France. It is known for the Eiffel Tower."),
        ("Berlin.txt", "Berlin is the capital of Germany. It has the Brandenburg Gate."),
        ("Tokyo.txt", "Tokyo is the capital of Japan. It is famous for Shibuya Crossing.")
    ]
    
    print(f"\n📥 Ingesting {len(documents)} validation documents...")
    import tempfile
    
    for filename, content in documents:
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".txt") as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        
        try:
            # We assume ingest_file works with Path
            # Docling might need real file, our parser handles txt natively
            await ingestor.ingest_file(tmp_path, strategy="semantic")
            print(f"  - Ingested {filename}")
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
                
    # 3. Benchmark Queries
    # Query -> Keyword that MUST be in retrieved content to count as "Hit"
    queries = [
        ("What is the capital of France?", "Paris"),
        ("Tell me about Germany", "Berlin"),
        ("What city has Shibuya Crossing?", "Tokyo"),
        ("Where is the Eiffel Tower?", "Paris"),
        ("Capital of Japan?", "Tokyo")
    ]
    
    print("\n🔍 Running Queries...")
    hits = 0
    k = 1
    
    print(f"{'QUERY':<40} | {'RETRIEVED':<30} | {'SCORE':<5} | {'HIT'}")
    print("-" * 90)
    
    for query, expected_keyword in queries:
        results = await retriever.retrieve(query, top_k=k)
        
        top_content = results[0].content if results else "NO RESULTS"
        score = f"{results[0].score:.2f}" if results else "0.00"
        
        is_hit = expected_keyword in top_content
        if is_hit:
            hits += 1
            
        print(f"{query:<40} | {top_content[:30]}... | {score:<5} | {'✅' if is_hit else '❌'}")

    # 4. Results
    accuracy = (hits / len(queries)) * 100
    print("-" * 90)
    print(f"📊 Results: {hits}/{len(queries)} Hits")
    print(f"🎯 Accuracy: {accuracy:.1f}%")

    if accuracy < 80:
        print("❌ FAILED: Accuracy below 80%")
        sys.exit(1)
    else:
        print("✅ PASSED: Retrieval Baseline OK")

if __name__ == "__main__":
    asyncio.run(main())
