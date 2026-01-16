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
    
    # 1. Setup Shared In-Memory Qdrant
    print("[INFO] Setting up in-memory vector DB...")
    shared_qdrant = QdrantHandler(use_memory=True)
    shared_qdrant.create_collection_if_not_exists()
    
    # 2. Wire Components
    ingestor = IngestionService(qdrant_handler=shared_qdrant)
    
    retriever = Retriever(qdrant_handler=shared_qdrant)
    tools = AgentTools(retriever=retriever)
    agent = AgentRouter(tools=tools)
    
    # 3. Ingest Dummy Data
    print("[INFO] Ingesting Knowledge Base...")
    
    kb_content = """
    The Eiffel Tower is located in Paris, France. It was constructed in 1889.
    The height of Eiffel Tower is 330 meters.
    
    Big Ben is the nickname for the Great Bell of the striking clock at the north end of the Palace of Westminster in London.
    It was completed in 1859.
    """
    
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".txt") as tmp:
        tmp.write(kb_content)
        tmp_path = Path(tmp.name)
        
    try:
        await ingestor.ingest_file(tmp_path)
        print("[OK] Ingestion Complete.")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
            
    # 4. Ingest Outdated Data for Testing
    print("\n[INFO] Ingesting Outdated Policy for Testing Warning...")
    # We manually inject a doc with is_latest=False into Qdrant to test the Agent's reaction
    # Since we can't easily force IngestionService to do this (it tries to manage state),
    # We will just ingest a new file, and then manually patch its payload in Qdrant.
    
    outdated_policy = "Policy 99: Students must wear red hats."
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".txt") as tmp:
        tmp.write(outdated_policy)
        tmp_path = Path(tmp.name)
        
    await ingestor.ingest_file(tmp_path)
    
    # Manually patch to make it 'outdated' (is_latest=False)
    # We need the Qdrant client from the shared handler
    shared_qdrant.client.set_payload(
        collection_name="documents",
        payload={"is_latest": False, "version_number": 1},
        points=models.Filter(
            must=[models.FieldCondition(key="content", match=models.MatchValue(value=outdated_policy))]
        )
    )
    if tmp_path.exists():
        tmp_path.unlink()

    # 5. Run Agent
    queries = [
        "What is the policy on students wearing hats?", # Should warn about outdated
        "Where is the Eiffel Tower located and how tall is it?",
        "Tell me about Big Ben.",
        "What is the capital of Mars?" 
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
