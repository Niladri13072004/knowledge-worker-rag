# 🧠 Knowledge Worker
### Conversational RAG (Retrieval-Augmented Generation) Knowledge Assistant

An advanced **Document Intelligence System** that allows users to upload PDFs and DOCX files and interact with them using natural language.

The system combines **Retrieval-Augmented Generation (RAG)**, **Hybrid Search**, **LLM-based Evaluation**, and a **ChatGPT-style interface** to provide accurate, context-aware answers grounded in uploaded documents.


<img width="1600" height="900" alt="Project-Architecture" src="https://github.com/user-attachments/assets/38e9c889-4423-4f19-808d-e9c3e115d47d" />

---

## ✨ Features

### 📄 Multi-Document Support
- Upload multiple PDF and DOCX files
- Query across all uploaded documents simultaneously
- Document-aware retrieval and source citations

---

### 🔍 Advanced Retrieval Pipeline

#### Semantic Search
- Vector embeddings
- Chroma Vector Database
- Dense retrieval

#### Keyword Search
- BM25 retrieval
- Exact keyword matching

#### Hybrid Retrieval
Combines:
- Dense semantic retrieval
- Sparse BM25 retrieval

---

### 🧠 Query Processing

#### Query Normalization
- Removes unnecessary whitespace
- Cleans punctuation
- Handles malformed queries

#### Query Rewriting
Rewrites user questions into retrieval-optimized queries.

Example:

**User**
```
What does the compressor do?
```

**Rewritten**
```
Primary function of a compressor in a refrigeration system
```

#### Multi-Query Generation
Creates multiple search queries to improve recall.

#### Acronym Expansion

Examples:
- ML → Machine Learning
- API → Application Programming Interface
- RAG → Retrieval Augmented Generation

#### Prompt Injection Protection

Filters malicious instructions such as:

```
Ignore previous instructions...
```

---

### 🎯 Intent Classification

Detects user intent:

- Fact Lookup
- Definition
- Comparison
- Analysis
- Summarization
- Troubleshooting
- Document Search
- How-To Questions

---

### 📚 Question Type Classification

Identifies:

- Numerical Questions
- Conceptual Questions
- Holistic Questions
- Multi-hop Questions
- Reasoning Questions

---

### ⚡ Dynamic Retrieval

Automatically adjusts retrieval depth (`k`) based on:

- Query type
- User intent
- Complexity

---

### 📑 Parent Document Retrieval

Retrieves neighboring context from parent chunks to improve answer completeness.

---

### 🗜️ Context Compression

Compresses retrieved documents and keeps only information relevant to the current question.

Benefits:

- Lower token usage
- Faster generation
- Better answer quality

---

### 🧩 Map-Reduce Reasoning

Complex questions are automatically decomposed into smaller sub-questions.

Example:

**Question**
```
Summarize Niladri's technical profile.
```

Pipeline:

```
Question
    ↓
Sub Question 1
Sub Question 2
Sub Question 3
    ↓
Answer Aggregation
```

---

### 💬 Conversational Memory

Maintains:

- Chat history
- Conversation summaries
- Context-aware follow-up questions

Example:

```
User:
What is Niladri's CGPA?

User:
What projects has he built?
```

The system understands that **he = Niladri**.

---

### 🌍 Dual Routing System

```
Question
    ↓
Classifier
 ├── Document Question
 │       ↓
 │      RAG
 │
 └── General Knowledge
         ↓
        LLM
```

Examples:

Document:

```
What is Niladri's CGPA?
```

General:

```
What is Machine Learning?
```

---

### 🤖 Self-Evaluation

The system evaluates its own answers.

Checks:

- Support by context
- Completeness
- Relevance
- Confidence score

---

### 🔄 Self-Correction

Low-confidence answers are automatically improved and regenerated.

---

### ✅ Answer Verification

Fact-checks generated answers against retrieved context.

---

### 📊 Evaluation Framework

Implemented metrics:

- Recall@K
- Coverage
- Mean Reciprocal Rank (MRR)
- LLM-as-a-Judge Evaluation

Judge Metrics:

- Accuracy
- Completeness
- Relevance

---

### 🎨 ChatGPT-Style Interface

Features:

- Modern chat interface
- Dark mode / Light mode
- Document upload button
- Source citations
- Copy responses
- Conversational experience similar to ChatGPT and Claude

---

# 🏗️ Architecture

```
User Question
      │
      ▼
Query Processing
(Normalization, Rewrite,
Expansion, Classification)
      │
      ▼
Hybrid Retrieval
(Vector Search + BM25)
      │
      ▼
Reranking
      │
      ▼
Parent Retrieval
      │
      ▼
Context Compression
      │
      ▼
LLM Generation
      │
      ▼
Self Evaluation
      │
      ▼
Verification
      │
      ▼
Final Answer
```

---

# 🛠️ Tech Stack

### Programming
- Python

### Frontend
- Gradio

### Vector Database
- ChromaDB

### Retrieval
- BM25
- Hybrid Retrieval

### LLM Frameworks
- LangChain
- LangGraph

### Embeddings
- HuggingFace Embeddings

### LLMs
- Ollama
- OpenRouter

### Evaluation
- LLM-as-a-Judge
- MRR
- Recall@K
- Coverage Metrics

---

# 🚀 Installation

## 1. Clone Repository

```bash
git clone https://github.com/Niladri13072004/knowledge-worker-rag.git
cd https://github.com/Niladri13072004/knowledge-worker-rag
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv env
env\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv env
source env/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
```

---

# 🦙 Install Ollama

Download:

https://ollama.com/

Pull a model:

```bash
ollama pull gemma3:4b
```

Start Ollama:

```bash
ollama serve
```

---

# ▶️ Running the Application

Start the UI:

```bash
python UI.py
```

Open:

```
http://127.0.0.1:7860
```

---

# 📖 How to Use

### Step 1
Upload one or more:

- PDF files
- DOCX files

### Step 2
Wait for:

```
Knowledge Base Ready
```

### Step 3
Ask questions:

Examples:

```
What is the primary function of a compressor?
```

```
What projects has Niladri built?
```

```
Summarize the uploaded document.
```

```
Compare the compressor types discussed in the notes.
```

---

# 💡 Example Workflow

```
Upload Resume.pdf
        ↓
Ask:
"What skills does Niladri have?"
        ↓
Hybrid Retrieval
        ↓
RAG Pipeline
        ↓
Verified Answer
        ↓
Sources Displayed
```

---

# Future Improvements

- Streaming responses
- Response caching
- OCR support
- Workspace management
- Query suggestions
- Feedback learning system
- Asynchronous retrieval pipeline
- Multi-agent orchestration

---

# 📜 License

MIT License

---

# ⭐ If you found this project useful, consider giving it a star on GitHub.
