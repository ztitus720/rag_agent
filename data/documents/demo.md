# RAG Agent Demo

本项目用于构建智能知识库问答系统。

## RAG
基本流程包括文档加载、文本切分、Embedding、向量检索以及基于上下文的生成。

## Agent
Agent 会先判断问题是否需要访问知识库；如果需要，则调用 RAG Tool 获取上下文，再让 LLM 生成回答。

## Engineering
系统使用 FastAPI 提供 REST API，ChromaDB 保存向量数据，并支持 Docker 化运行。
