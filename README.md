# Knowledge Worker RAG

A Retrieval-Augmented Generation (RAG) system that transforms PDFs and documents into a searchable knowledge base, enabling users to ask natural language questions and receive grounded answers with source references.

## Overview

Knowledge Worker RAG is designed to bridge the gap between large document collections and conversational AI. Instead of relying solely on an LLM's internal knowledge, the system retrieves relevant information from uploaded documents and uses it to generate accurate, context-aware responses.

## Features

### Current Features

* PDF document ingestion
* Text extraction using PyMuPDF
* Semantic text chunking
* Vector embeddings
* Chroma vector database
* Similarity-based retrieval
* LLM-powered question answering

### Planned Features

* Hybrid Search (Vector + BM25)
* Query Rewriting
* Query Expansion
* Reranking
* Parent-Child Retrieval
* Citation Grounding
* Retrieval Evaluation Metrics
* Answer Evaluation Framework
* Self-Correction Pipeline
* Multi-Document Support
* Agentic Knowledge Worker Capabilities

---

## Architecture

```text
PDF Documents
      ↓
Text Extraction
      ↓
Chunking
      ↓
Embeddings
      ↓
Chroma Vector Store
      ↓
User Query
      ↓
Retriever
      ↓
Context Builder
      ↓
LLM
      ↓
Answer
```

---

## Tech Stack

### Retrieval

* PyMuPDF
* LangChain
* ChromaDB
* Sentence Transformers

### Language Models

* Gemini API
* Ollama (Local Models)
* Qwen Series Models

### Frontend

* Gradio

### Evaluation

* Scikit-Learn
* Custom Retrieval Metrics

---

## Project Structure

```text
knowledge-worker/

├── data/
│   └── pdfs/

├── vectorstore/

├── ingest.py
├── retrieve.py
├── answer.py
├── app.py

├── requirements.txt
├── .env
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/knowledge-worker-rag.git
cd knowledge-worker-rag
```

Create a virtual environment:

```bash
python -m venv env
```

Activate the environment:

### Windows

```bash
env\Scripts\activate
```

### Linux / macOS

```bash
source env/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
```

Never commit your API keys to GitHub.

---

## Future Roadmap

* [ ] Hybrid Retrieval
* [ ] BM25 Integration
* [ ] Query Expansion
* [ ] Query Rewriting
* [ ] Reranker
* [ ] Parent-Child Retrieval
* [ ] Citation System
* [ ] Retrieval Evaluation
* [ ] Answer Evaluation
* [ ] Self-Correcting RAG
* [ ] Agentic Knowledge Worker

---

## Learning Goals

This project is being developed to gain hands-on experience in:

* Retrieval-Augmented Generation (RAG)
* Information Retrieval
* LLM Engineering
* Evaluation Frameworks
* Production AI System Design
* Agentic Workflows

---

## License

MIT License
