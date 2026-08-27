import json

from sentence_transformers import CrossEncoder

from app.rag.embeddings import embed_query
from app.rag.vectorstore import collection


DATASET_PATH = "tests/evaluation_dataset.json"

TOP_K = 10


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def retrieve(query, top_k=TOP_K):
    result = collection.query(
        query_embeddings=[embed_query(query)],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    return [
        {
            "chunk_index": metadata.get("chunk_index"),
            "source": metadata.get("source"),
            "text": document,
            "distance": distance,
        }
        for document, metadata, distance in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        )
    ]


def hit_at_k(results, relevant_chunks, k):
    retrieved = [
        item["chunk_index"]
        for item in results[:k]
    ]

    return int(
        any(chunk in relevant_chunks for chunk in retrieved)
    )


def reciprocal_rank(results, relevant_chunks):
    for rank, item in enumerate(results, start=1):
        if item["chunk_index"] in relevant_chunks:
            return 1.0 / rank

    return 0.0


def rerank(reranker, query, results):
    pairs = [
        (query, item["text"])
        for item in results
    ]

    scores = reranker.predict(pairs)

    for item, score in zip(results, scores):
        item["reranker_score"] = float(score)

    return sorted(
        results,
        key=lambda x: x["reranker_score"],
        reverse=True,
    )


def evaluate():
    dataset = load_dataset()

    print("=" * 70)
    print("RAG Retrieval Evaluation")
    print("=" * 70)

    print("\nLoading BGE Reranker...")

    reranker = CrossEncoder(
        "BAAI/bge-reranker-v2-m3"
    )

    print("Reranker loaded successfully.\n")

    dense_results_all = []
    reranked_results_all = []

    for index, item in enumerate(dataset, start=1):

        query = item["query"]
        relevant_chunks = item["relevant_chunks"]

        print("=" * 70)
        print(f"Query {index}")
        print("=" * 70)

        print(f"Query: {query}")
        print(f"Relevant chunks: {relevant_chunks}")

        # Dense Retrieval
        dense_results = retrieve(query)

        # Reranking
        reranked_results = rerank(
            reranker,
            query,
            dense_results.copy(),
        )

        dense_results_all.append(
            (dense_results, relevant_chunks)
        )

        reranked_results_all.append(
            (reranked_results, relevant_chunks)
        )

        dense_rank = None
        reranker_rank = None

        for rank, result in enumerate(
            dense_results,
            start=1,
        ):
            if result["chunk_index"] in relevant_chunks:
                dense_rank = rank
                break

        for rank, result in enumerate(
            reranked_results,
            start=1,
        ):
            if result["chunk_index"] in relevant_chunks:
                reranker_rank = rank
                break

        print(
            f"Dense first relevant rank: "
            f"{dense_rank}"
        )

        print(
            f"Reranker first relevant rank: "
            f"{reranker_rank}"
        )

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    def calculate_metrics(all_results):

        hit1 = sum(
            hit_at_k(results, relevant, 1)
            for results, relevant in all_results
        )

        hit3 = sum(
            hit_at_k(results, relevant, 3)
            for results, relevant in all_results
        )

        hit5 = sum(
            hit_at_k(results, relevant, 5)
            for results, relevant in all_results
        )

        mrr = sum(
            reciprocal_rank(results, relevant)
            for results, relevant in all_results
        )

        total = len(all_results)

        return {
            "Hit@1": hit1 / total,
            "Hit@3": hit3 / total,
            "Hit@5": hit5 / total,
            "MRR": mrr / total,
        }

    dense_metrics = calculate_metrics(
        dense_results_all
    )

    reranker_metrics = calculate_metrics(
        reranked_results_all
    )

    # --------------------------------------------------
    # Final Results
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL EVALUATION RESULTS")
    print("=" * 70)

    print(
        "\nMetric          Dense       + Reranker"
    )

    print("-" * 70)

    for metric in [
        "Hit@1",
        "Hit@3",
        "Hit@5",
        "MRR",
    ]:
        print(
            f"{metric:<15}"
            f"{dense_metrics[metric]:.4f}"
            f"       "
            f"{reranker_metrics[metric]:.4f}"
        )


if __name__ == "__main__":
    evaluate()