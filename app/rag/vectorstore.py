import chromadb
from sentence_transformers import CrossEncoder

from app.config import settings
from app.rag.embeddings import embed_documents, embed_query


client = chromadb.PersistentClient(path=settings.chroma_dir)

collection = client.get_or_create_collection(
    name="knowledge_base"
)


# --------------------------------------------------
# BGE Reranker
# --------------------------------------------------

_reranker = None


def get_reranker():
    global _reranker

    if _reranker is None:
        print("Loading BGE Reranker...")

        _reranker = CrossEncoder(
            "BAAI/bge-reranker-v2-m3"
        )

        print("BGE Reranker loaded successfully.")

    return _reranker


# --------------------------------------------------
# Document Ingestion
# --------------------------------------------------

def add_documents(texts, metadatas):
    if not texts:
        return 0

    embeddings = embed_documents(texts)

    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(texts)


# --------------------------------------------------
# Dense Retrieval
# --------------------------------------------------

def dense_search(query, top_k=10):
    r = collection.query(
        query_embeddings=[embed_query(query)],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    docs = r.get("documents", [[]])[0]
    metas = r.get("metadatas", [[]])[0]
    distances = r.get("distances", [[]])[0]

    return [
        {
            "text": d,
            "metadata": m,
            "distance": dist,
        }
        for d, m, dist in zip(
            docs,
            metas,
            distances,
        )
    ]


# --------------------------------------------------
# Reranked Retrieval
# --------------------------------------------------

def search(
    query,
    top_k=3,
    candidate_k=10,
):
    """
    Dense Retrieval + BGE Reranker

    1. ChromaDB retrieves candidate_k documents.
    2. BGE Reranker scores all candidates.
    3. Return top_k most relevant documents.
    """

    candidates = dense_search(
        query,
        top_k=candidate_k,
    )

    if not candidates:
        return []

    reranker = get_reranker()

    pairs = [
        (query, item["text"])
        for item in candidates
    ]

    scores = reranker.predict(pairs)

    for item, score in zip(
        candidates,
        scores,
    ):
        item["reranker_score"] = float(score)

    candidates.sort(
        key=lambda x: x["reranker_score"],
        reverse=True,
    )

    return candidates[:top_k]