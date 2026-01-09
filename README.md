# AI-powered NEET Exam Preparation – Agentic RAG Assistant

A **production-grade Agentic Retrieval-Augmented Generation (RAG) system** built for **NEET Physics exam preparation**.  
The system delivers **concept explanations, exam-style answers, and MCQ generation**, strictly grounded in NCERT/NEET source PDFs with **automatic verification and citation enforcement**.

This project demonstrates **real-world GenAI architecture**, not a toy LLM demo.

---

## Why This Project Matters 

Most GenAI demos stop at *“upload PDF → ask LLM”*.  
This project goes further by implementing:

- Multi-agent orchestration
- Verification-driven retries
- Persistent vector search
- Hallucination control
- Transparent AI decision-making

This is the **same architectural pattern used in enterprise AI assistants**, EdTech platforms, and regulated-domain RAG systems.

---

## Core Capabilities

### 1. Agentic RAG Pipeline
A LangGraph-based workflow with clearly separated responsibilities:

- **QueryRouterAgent** – understands user intent (concept, exam Q&A, MCQ)
- **RetrievalAgent** – semantic search over vectorized PDFs
- **AnswerAgent** – generates grounded responses with citations
- **VerificationAgent** – checks factual support and enforces retry if needed

---

### 2. Strict Source Grounding
- Answers and MCQs are generated **only from retrieved chunks**
- Mandatory citations like `[S1]`, `[S2]`
- Unsupported claims are penalized
- Automatic revision if citations are missing

---

### 3. NEET-style MCQ Generator
- Generates **exactly 5 MCQs**
- 4 options per question (A–D)
- Difficulty tagging: Easy / Medium / Hard
- Each MCQ grounded to source PDFs

---

### 4. Persistent Vector Database
- **ChromaDB** with disk persistence
- Idempotent ingestion
- Deterministic semantic retrieval
- Supports filtering and retry with broader scope

---

### 5. Verification-Driven Retry Loop
- Each answer is scored (0–1)
- If score < threshold → automatic retry
- Retrieval scope widens on retry
- Prevents hallucinations and weak grounding

---

### 6. Transparent UI
- Streamlit interface
- Shows:
  - Final answer
  - Source chunks
  - Agent decision steps
  - Verification score & notes

---

## Architecture Overview

    User Query
    ↓
    QueryRouterAgent
    ↓
    RetrievalAgent (ChromaDB)
    ↓
    AnswerAgent (LLM + citations)
    ↓
    VerificationAgent
    ↓
    (Optional retry with broader retrieval)


    Clean separation of:
    - Ingestion vs Query
    - Reasoning vs Verification
    - Generation vs Validation


## Tech Stack

- **Language:** Python 3.10+
- **LLM:** OpenAI (gpt-4o-mini)
- **Embeddings:** sentence-transformers / all-MiniLM-L6-v2
- **Vector DB:** Chroma (persistent)
- **Agent Framework:** LangGraph
- **UI:** Streamlit
- **PDF Processing:** LangChain loaders

---

## Project Structure

    app/
    ├── agents/
    │ ├── query_router.py
    │ ├── retrieval_agent.py
    │ ├── answer_agent.py
    │ └── verification_agent.py
    ├── core/
    │ ├── graph.py
    │ ├── state.py
    │ └── agent_logger.py
    ├── ingestion/
    │ ├── pdf_loader.py
    │ ├── chunker.py
    │ └── embedder.py
    ├── utils/
    │ └── config.py
    ├── ui/
    │ └── streamlit_app.py
    data/
    ├── raw_pdfs/
    │ └── NEET Physics source PDFs
    requirements.txt
    README.md


---

## How to Run Locally

### 1. Clone
    git clone https://github.com/himanshu-dandle/ai-powered-neet-agentic-rag-assistant.git
    cd ai-powered-neet-agentic-rag-assistant

### 2. Environment Setup

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
### 3. API Key
    Create .env:
        OPENAI_API_KEY=your_key_here

### 4. Ingest PDFs (one-time)
    python test_chroma.py

### 5. Launch UI
    streamlit run streamlit_app.py


## Example Queries

--Explain Newton's second law in simple terms
--Generate 5 MCQs on Newton laws with difficulty
--Give exam-style solution for work-energy theorem
--Explain momentum conservation for NEET

## What This Demonstrates (for Toptal)

--Agentic AI design (not prompt-only)
--Production-safe RAG
--Verification & governance
--Clear system thinking
--Strong Python + GenAI engineering

## Author
    Himanshu Dandle
    GitHub: https://github.com/himanshu-dandle

## License
    MIT