# Intelligent Knowledge Assistant — RAG + Agent

A production-oriented AI knowledge assistant built with Python, FastAPI, LangGraph, ChromaDB, Sentence Transformers, BGE Reranker, Tavily and an OpenAI-compatible LLM API.

The system combines Retrieval-Augmented Generation (RAG) with an Agent-based routing architecture, allowing the assistant to automatically select the appropriate tool for each user query.

## Architecture

```mermaid
flowchart TD
    A[User Query] --> B[FastAPI]
    B --> C[LangGraph Agent]

    C --> D{Router}

    D -->|Private Knowledge| E[RAG Retrieval]
    D -->|Arithmetic| F[Calculator]
    D -->|Current / Public Info| G[Web Search]
    D -->|General Conversation| H[Direct LLM]

    E --> E1[ChromaDB]
    E1 --> E2[Embedding]
    E2 --> E3[BGE Reranker]

    E3 --> I[Answer Generation]
    F --> I
    G --> I
    H --> I

    I --> J[Streaming Response]
    J --> K[Conversation Memory]
