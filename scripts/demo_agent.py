import asyncio
import sys
from pathlib import Path
import tempfile
from qdrant_client import models

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.infrastructure.db.qdrant import QdrantHandler
from src.services.ingestion import IngestionService
from src.services.retrieval import Retriever
from src.domain.chat.tools import AgentTools
from src.domain.chat.agent import AgentRouter

async def main():
    print("\n[START] Starting Agentic RAG Demo (In-Memory)...\n")
    
    # Clean up registry for demo consistency
    registry_path = Path("data/registry.db")
    if registry_path.exists():
        try:
            registry_path.unlink()
            print("[INFO] Cleared existing registry for fresh start.")
        except Exception as e:
            print(f"[WARN] Could not clear registry: {e}")

    # 1. Setup Shared In-Memory Qdrant
    print("[INFO] Setting up in-memory vector DB...")
    shared_qdrant = QdrantHandler(use_memory=True)
    shared_qdrant.create_collection_if_not_exists()
    
    # 2. Wire Components
    ingestor = IngestionService(qdrant_handler=shared_qdrant)
    
    retriever = Retriever(qdrant_handler=shared_qdrant)
    tools = AgentTools(retriever=retriever)
    agent = AgentRouter(tools=tools)
    
    # 3. Ingest Real Data from data/source_docs
    source_dir = Path("data/source_docs")
    print(f"\n[INFO] Ingesting Real Documents from {source_dir}...")
    
    if not source_dir.exists():
        print(f"[ERROR] Directory {source_dir} not found!")
        return

    files = [f for f in source_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.pdf', '.txt', '.docx']]
    
    if not files:
        print("[WARN] No documents found in source directory.")
    
    for file_path in files:
        print(f"   -> Processing: {file_path.name}")
        await ingestor.ingest_file(file_path)
    
    print("[OK] Ingestion Complete.")
    
    # Debug: Check extraction
    count = shared_qdrant.client.count(collection_name="documents")
    print(f"\n[DEBUG] Total Chunks in DB: {count.count}")
    
    # 4. Run Agent with Real Questions
    queries = [
        "what is STUDY ABROAD PROGRAMME (SAP) in amity?",
        "Summarize the DISCIPLINARY CONTROL OF STUDENTS IN EXAMINATIONS",
        "What are the grading system for Very Good to fail"
    ]
    
    for query in queries:
        print(f"\n\n[USER QUERY]: {query}")
        print("-" * 60)
        
        async for step in agent.run(query):
            print(f"[{step.state.value.upper()}] {step.thought}")
            
            if step.action:
                print(f"   [ACTION]: {step.action.action_type}")
                if step.action.action_type == 'retrieve':
                    print(f"      Query: {step.action.query}")
                elif step.action.action_type == 'answer':
                    print(f"      Answer: {step.action.answer}")
                    print(f"      Confidence: {step.action.confidence_score}")
                elif step.action.action_type == 'refuse':
                    print(f"      Reason: {step.action.reason}")
            
            if step.observation:
                print(f"   [OBSERVATION]: {step.observation[:100]}...") # Truncate
                
        print("-" * 60)

if __name__ == "__main__":
    asyncio.run(main())
