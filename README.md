# Agentic Document Intelligence Platform (RAG Learning System)

> **Note**: This project is a hands-on learning system built to explore **Agentic RAG**, **Versioning**, and **Failure Handling**. It is NOT a production SaaS product.

## 1. Project Overview
This project solves the problem of **"Blind RAG"**—where a system retrieves outdated or irrelevant chunks and confidently hallucinates an answer. 
We explore this using **University Student Handbooks**, which are notoriously difficult due to:
*   Yearly updates (Versioning/Drift).
*   Complex formatting (Tables, Policies).
*   Ambiguous queries ("Can I wear a hat?" vs "Policy 4.2").

**Why Build This?**
To move beyond "Chat with PDF" demos and handle the messy reality of data: **Versioning**, **Staleness**, and **Refusal**.

## 2. Core Ideas & Techniques

### Agentic Decision Making (FSM)
Instead of a linear chain (Ingest -> Search -> Answer), we use a **Finite State Machine (FSM)**. The agent *decides* if it needs to Search, Refuse, or Clarify.

**Why FSM instead of Chains?**
*   **Predictability**: We define strict states (`THINKING`, `RETRIEVING`, `REFUSING`). The agent cannot "wander" into an undefined loop.
*   **Debuggability**: When the system fails, we know exactly which state transition was invalid (e.g., trying to Answer without first Retrieving).
*   **Explicit Refusal**: Refusal is a first-class state, not an "error". This forces the system to treat "I don't know" as a valid, handled outcome.

### Drift Awareness
The system detects if a document is outdated (e.g., "Handbook 2024" vs "Handbook 2025"). If the retrieved info is old (marked `is_latest=False`), the Agent proactively **Warns** the user that the answer might be superseded.

### Confidence Guardrails
"I don't know" is better than a lie. If the confidence score is low (< 0.5) or retrieval is empty, the system proactively **Refuses** to answer.

## 3. System Architecture
The system follows a Modular Monolith design, optimized for local understanding:

### A. Data Layer (`src/infrastructure`)
*   **DocumentRegistry (SQLite)**: The "Brain" that tracks file hashes, versions, and metadata.
*   **Vector Store (Qdrant)**: Stores semantic embeddings of document chunks.
*   **Ingestion Service**: Handles File -> Text -> Chunks -> Embeddings pipeline.

### B. Agent Layer (`src/domain/chat`)
*   **Retriever**: Fetches relevant chunks.
*   **AgentRouter**: The FSM logic (Thinking -> Retrieving -> Analyzing -> Answering).

### C. Application Layer (`src/app`)
*   **FastAPI**: Exposes the Agent via REST API (`POST /chat`).
*   **Background Worker**: 
    *   *Design Choice*: This is a simple, **single-process asyncio task** that runs locally. 
    *   *Scope*: It is NOT a distributed queue (like Celery/Kafka). This intentional constraint allows the system to run on a single laptop while still demonstrating the *concept* of continuous ingestion.

## 4. Real Data Evaluation
The system was verified against the **Student Handbook 2025 (Real Data)**.

### Results
1.  **Question: "What is the Study Abroad Programme (SAP)?"**
    *   **Result**: ✅ **ANSWERED**
    *   **Why**: The relevant sections were text-heavy paragraphs, which parsed correctly. The Agent synthesized the chunks into a clear summary.

2.  **Question: "Summarize Disciplinary Control in Examinations"**
    *   **Result**: 🛡️ **REFUSED (Safety Success)**
    *   **Why**: This information resides in complex tables. With OCR disabled (see below), the parser returned empty/fragmented text.
    *   **Behavior**: Instead of hallucinating a policy, the Agent recognized the lack of context and returned a Refusal. This is the **correct safe behavior** for a compliance system.

### The OCR Safety Tradeoff
Running heavy OCR (Optical Character Recognition) on a standard laptop often causes crashes or huge latency. 
*   **Decision**: We disabled OCR for stability in the demo profile.
*   **Consequence**: The system cannot "see" complex tables involved in Grading or rules.
*   **Outcome**: The system correctly identifies that it *cannot* answer these questions, demonstrating that **Refusal > Hallucination**.

## 5. Design Decisions & Tradeoffs
*   **Local-First**: Everything (DB, Vector Store, LLM logic) runs locally. No cloud dependencies except the LLM API provider.
*   **Versioning**: We assume "Append-Only". Old versions are kept but marked `is_latest=False` to allow historical queries while defaulting to current rules.
*   **Deployment**: Intentionally limited to `docker-compose` or `uv run` for simplicity. No Kubernetes/Cloud complexities.

## 6. What I Learned From This Project
Building this system highlighted several engineering realities often missed in tutorials:
1.  **Retrieval Quality is Logic, not Magic**: 90% of RAG failures stem from the parser losing structure (tables/headers), not the LLM being "dumb".
2.  **The "Safety" of Refusal**: It is far better to have an agent say "I cannot read that table" than to invent a policy. FSMs make this state explicit.
3.  **Real Data Break Naive Chunking**: Naive 500-char chunking destroys the meaning of a 3-page policy. Document-aware chunking is non-negotiable for legal texts.
4.  **Local Constraints**: Running "Heavy" OCR on a laptop is a tradeoff. Disabling it saves RAM but loses data. A production system *must* offload parsing to specialized infrastructure.
5.  **Versioning Matters**: A query is meaningless without a point in time. "What is the rule?" implies "What is the rule *now*?". Metadata is as important as the vector itself.

## 7. How to Run

### Prerequisites
*   Windows/Linux/Mac
*   Python 3.10+
*   `uv` (Universal Venv) or `pip`
*   Ram: 8GB+ (for In-Memory Vector Store)

### Quick Start (Demo Script)
The demo script spins up an in-memory database, ingests documents from `data/source_docs`, and answers sample questions.
```powershell
# 1. Install Dependencies
uv sync

# 2. Run the Demo Agent
uv run scripts/demo_agent.py
```

### Run API Server
To run the full backend with persistent storage:
```powershell
uv run uvicorn src.app.main:app --reload
```
Swagger UI available at `http://localhost:8000/docs`.

### Docker
```bash
docker-compose up --build
```

## 8. Scope & Non-Goals
*   **User UI**: We rely on API/CLI. No React/Streamlit frontend was built (out of scope).
*   **Multi-Tenancy**: Single user/workspace assumed.
*   **Permissioning**: No "Admin vs Student" RBAC.
*   **High Availability**: Single instance architecture.

---
*Built for learning agentic patterns in RAG systems.*