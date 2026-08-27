from sentence_transformers import CrossEncoder

from app.rag.embeddings import embed_query
from app.rag.vectorstore import collection


QUERIES = [
    "邹哲的多六足机器人项目使用了哪些技术？",
    "邹哲是如何实现六足机器人稳定行走的？",
    "邹哲有哪些项目涉及计算机视觉？",
    "邹哲有哪些后端开发经验？",
    "邹哲在哪些项目中使用了 Python？",
]


def dense_retrieval(query, top_k=5):
    result = collection.query(
        query_embeddings=[embed_query(query)],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    candidates = []

    for rank, (doc, metadata, distance) in enumerate(
        zip(documents, metadatas, distances),
        start=1,
    ):
        candidates.append(
            {
                "original_rank": rank,
                "text": doc,
                "metadata": metadata,
                "distance": distance,
            }
        )

    return candidates


def get_chunk_name(item):
    metadata = item["metadata"]

    source = metadata.get("source", "")
    chunk_index = metadata.get("chunk_index", "")

    if source.endswith(".pdf"):
        return f"PDF chunk {chunk_index}"

    return f"{source} chunk {chunk_index}"


def main():
    print("=" * 70)
    print("Multi-Query Reranker Evaluation")
    print("=" * 70)

    print("\nLoading BGE Reranker...")

    reranker = CrossEncoder(
        "BAAI/bge-reranker-v2-m3"
    )

    print("Reranker loaded successfully.")

    all_results = []

    for query_index, query in enumerate(QUERIES, start=1):

        print("\n" + "=" * 70)
        print(f"QUERY {query_index}")
        print("=" * 70)

        print(f"\nQuery: {query}")

        # --------------------------------------------------
        # 1. Dense Retrieval
        # --------------------------------------------------

        candidates = dense_retrieval(query, top_k=5)

        print("\nDense Retrieval:")
        print("-" * 70)

        for item in candidates:
            print(
                f"#{item['original_rank']} "
                f"{get_chunk_name(item)} "
                f"| distance={item['distance']:.6f}"
            )

        # --------------------------------------------------
        # 2. BGE Reranker
        # --------------------------------------------------

        pairs = [
            (query, item["text"])
            for item in candidates
        ]

        scores = reranker.predict(pairs)

        for item, score in zip(candidates, scores):
            item["reranker_score"] = float(score)

        reranked = sorted(
            candidates,
            key=lambda x: x["reranker_score"],
            reverse=True,
        )

        print("\nBGE Reranker:")
        print("-" * 70)

        for rank, item in enumerate(reranked, start=1):
            print(
                f"#{rank} "
                f"{get_chunk_name(item)} "
                f"| score={item['reranker_score']:.6f} "
                f"| original=#{item['original_rank']}"
            )

        # --------------------------------------------------
        # 3. Calculate movement
        # --------------------------------------------------

        for rank, item in enumerate(reranked, start=1):
            item["reranker_rank"] = rank
            item["rank_change"] = (
                item["original_rank"] - rank
            )

        # --------------------------------------------------
        # 4. Save result
        # --------------------------------------------------

        all_results.append(
            {
                "query": query,
                "dense_top1": candidates[0],
                "reranker_top1": reranked[0],
            }
        )

        print("\nTop-1 Comparison:")
        print("-" * 70)

        print(
            f"Dense Top-1: "
            f"{get_chunk_name(candidates[0])}"
        )

        print(
            f"Reranker Top-1: "
            f"{get_chunk_name(reranked[0])}"
        )

    # ------------------------------------------------------
    # Final Summary
    # ------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    for i, result in enumerate(all_results, start=1):

        dense = result["dense_top1"]
        reranked = result["reranker_top1"]

        print(f"\nQuery {i}:")
        print(f"  {result['query']}")

        print(
            f"  Dense Top-1: "
            f"{get_chunk_name(dense)}"
        )

        print(
            f"  Reranker Top-1: "
            f"{get_chunk_name(reranked)}"
        )

        print(
            f"  Reranker Score: "
            f"{reranked['reranker_score']:.6f}"
        )


if __name__ == "__main__":
    main()