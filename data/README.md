# AgenticRAG StudyCoach 📚🤖

AgenticRAG StudyCoach is an **agentic Retrieval-Augmented Generation (RAG)** based learning assistant designed to help students and learners understand concepts, answer exam-style questions, and generate MCQs — **strictly grounded in source documents with verification**.

This project demonstrates **agentic AI design**, multi-step reasoning, source-grounded answering, and verification-driven retries using **LangGraph**.

---

## 🚀 Key Features

### 🔹 Concept Explanation
- Explains physics concepts clearly and simply
- Answers are **strictly grounded in retrieved documents**
- Every paragraph includes **citations** like `[S1]`, `[S2]`

### 🔹 Exam-style Q&A
- Structured, exam-ready answers
- Step-by-step reasoning
- No hallucinated content

### 🔹 MCQ Generator
- Generates **exactly 5 MCQs**
- 4 options per question (A–D)
- Includes:
  - Correct answer
  - Difficulty level (Easy / Medium / Hard)
  - Citations per MCQ

### 🔹 Verification Agent
- Automatically checks:
  - Missing citations
  - Unsupported claims
- Assigns a **verification score (0–1)**
- Triggers **retry with broader retrieval** if needed

---

## 🧠 Agentic Architecture

This system is built using **LangGraph**, where each agent has a clear responsibility:

User Query
↓
QueryRouterAgent
↓
RetrievalAgent (ChromaDB)
↓
AnswerAgent
↓
VerificationAgent
↓
(Optional retry with broader scope)


---

## 🧩 Agents Overview

| Agent | Responsibility |
|------|---------------|
| QueryRouterAgent | Determines intent (concept, exam, MCQ) |
| RetrievalAgent | Fetches relevant chunks from ChromaDB |
| AnswerAgent | Generates grounded answers with citations |
| VerificationAgent | Verifies factual grounding & citations |

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **LangGraph**
- **LangChain**
- **ChromaDB (persistent)**
- **Sentence Transformers** (`all-MiniLM-L6-v2`)
- **OpenAI GPT-4o-mini**
- **Streamlit** (UI)

---

## 📂 Project Structure

AgenticRAG-StudyCoach/
│
├── app/
│ ├── agents/
│ │ ├── query_router.py
│ │ ├── retrieval_agent.py
│ │ ├── answer_agent.py
│ │ └── verification_agent.py
│ │
│ ├── core/
│ │ ├── graph.py
│ │ ├── state.py
│ │ └── agent_logger.py
│ │
│ └── utils/
│ └── config.py
│
├── data/
│ └── chroma/ # Persistent vector store
│
├── streamlit.py
├── requirements.txt
└── README.md



---

## ▶️ How to Run

### 1️⃣ Create virtual environment

    python -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate

### 2️⃣ Install dependencies

    pip install -r requirements.txt

### 3️⃣ Set environment variables

    Create a .env file and below test
        OPENAI_API_KEY=your_api_key_here

### 4️⃣ Run the app

    streamlit run streamlit.py

## 🎛️ Streamlit Controls

    🔹Mode
        -Auto (Agent decides)
        -Concept Explanation
        -Exam-style Q&A
        -MCQ Generator

    🔹Max Iterations
        -Controls verification-driven retries

    🔹Live Agent Step Trace
        -See how each agent reasons

## 📊 Example Output
    🔹MCQ Generation
        -Question + options
        -Correct answer
        -Difficulty tag
        -Source citation per MCQ

    🔹Verification

        -Score (e.g., 0.85)
        -Notes explaining any issues
        -Retry decision

## 🎯 Why This Project Matters

This project demonstrates:

🔹True agentic AI design
🔹Grounded RAG (no hallucinations)
🔹Verification-driven workflows
🔹Education-focused AI use case

Ideal for:

🔹AI Engineer portfolios
🔹GenAI interviews
🔹Agentic AI demonstrations

## 📌 Future Enhancements

🔹 PDF upload support
🔹 Per-question verification
🔹 Export MCQs to PDF / CSV
🔹 Difficulty calibration using metadata
🔹 Multi-subject support

## 👨‍💻 Author
 Himanshu Dandle
