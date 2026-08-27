# Intelligent Knowledge Assistant — RAG + Agent

Python + FastAPI + LangGraph + ChromaDB + Sentence Transformers + OpenAI-compatible LLM API.

## Quick Start

1. Create environment:
   `python -m venv .venv`
2. Activate it and install:
   `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your LLM API key.
4. Put PDF/TXT/MD files into `data/documents/`.
5. Index documents:
   `python -m app.rag.ingest`
6. Start:
   `uvicorn app.main:app --reload`
7. Open `http://127.0.0.1:8000/docs`

## API
- `GET /health`
- `POST /documents/upload`
- `POST /search` with `{"query":"...", "top_k":4}`
- `POST /chat` with `{"message":"..."}`

The Agent routes each question to direct LLM answering or the RAG retriever.
