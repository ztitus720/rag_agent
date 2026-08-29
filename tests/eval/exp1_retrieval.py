# -*- coding: utf-8 -*-
"""
实验 1（P0）：检索质量评测 —— Dense vs Dense + BGE Reranker

- 评测集：tests/eval/datasets/retrieval_queries.json（26 条，分 fact / paraphrase / multi 三类）
- 指标：Hit@1 / Hit@3 / Hit@5 / Recall@3 / Recall@5 / MRR
- 语料：每次从 data/documents 重新切分并写入内存向量库，结果可复现

输出：tests/eval/results/exp1_retrieval.json
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# 本实验只用本地已缓存的模型，禁止联网探测（国内直连 HF 会卡住）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import common  # noqa: E402

common.setup_console()

CANDIDATE_K = 10


def main():
    print("=" * 74)
    print("实验 1 / 检索质量评测：Dense vs Dense + BGE Reranker")
    print("=" * 74)

    corpus = common.load_corpus()
    print(f"\n语料：{len(corpus)} 个 chunk")
    for c in corpus:
        print(f"  {c['id']:<12} len={len(c['text']):<5} {c['source']}")

    from app.config import settings
    print(f"\nEmbedding 模型：{settings.embedding_model}")

    t0 = time.time()
    collection = common.build_ephemeral_collection(corpus)
    print(f"索引构建完成，用时 {time.time() - t0:.2f}s")

    print("\n加载 BGE Reranker (BAAI/bge-reranker-v2-m3)...")
    from sentence_transformers import CrossEncoder
    t0 = time.time()
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
    print(f"Reranker 加载完成，用时 {time.time() - t0:.2f}s")

    from app.rag.embeddings import embed_query

    dataset = common.load_dataset("retrieval_queries.json")
    queries = dataset["queries"]
    k = min(CANDIDATE_K, len(corpus))

    dense_all, rerank_all, details = [], [], []
    dense_latency, rerank_latency = [], []

    for item in queries:
        q, relevant = item["query"], item["relevant"]

        t0 = time.time()
        r = collection.query(query_embeddings=[embed_query(q)], n_results=k,
                             include=["documents", "metadatas", "distances"])
        dense_latency.append(time.time() - t0)

        ids = list(r["ids"][0])
        docs = list(r["documents"][0])
        dists = list(r["distances"][0])

        t0 = time.time()
        scores = reranker.predict([(q, d) for d in docs])
        rerank_latency.append(time.time() - t0)

        order = sorted(range(len(ids)), key=lambda i: float(scores[i]), reverse=True)
        reranked_ids = [ids[i] for i in order]

        dense_all.append({"ranked": ids, "relevant": relevant})
        rerank_all.append({"ranked": reranked_ids, "relevant": relevant})

        d_rank = common.first_relevant_rank(ids, relevant)
        r_rank = common.first_relevant_rank(reranked_ids, relevant)

        details.append({
            "id": item["id"], "category": item["category"], "query": q,
            "relevant": relevant,
            "dense_top3": ids[:3], "reranked_top3": reranked_ids[:3],
            "dense_first_rank": d_rank, "reranked_first_rank": r_rank,
            "rank_change": (d_rank - r_rank) if (d_rank and r_rank) else None,
            "dense_top1_distance": round(float(dists[0]), 6),
            "reranked_top1_score": round(float(scores[order[0]]), 6),
        })

        flag = "  "
        if d_rank and r_rank:
            if r_rank < d_rank:
                flag = "↑ "
            elif r_rank > d_rank:
                flag = "↓ "
        print(f"{flag}{item['id']} [{item['category']:<10}] dense#{d_rank} -> rerank#{r_rank}  {q}")

    dense_metrics = common.aggregate(dense_all)
    rerank_metrics = common.aggregate(rerank_all)

    # 分类别拆解
    by_category = {}
    for cat in ("fact", "paraphrase", "multi"):
        idx = [i for i, q in enumerate(queries) if q["category"] == cat]
        if not idx:
            continue
        by_category[cat] = {
            "dense": common.aggregate([dense_all[i] for i in idx]),
            "reranked": common.aggregate([rerank_all[i] for i in idx]),
        }

    print("\n" + "=" * 74)
    print("总体结果")
    print("=" * 74)
    rows = []
    for m in ("Hit@1", "Hit@3", "Hit@5", "Recall@3", "Recall@5", "MRR"):
        d, r = dense_metrics[m], rerank_metrics[m]
        rows.append([m, f"{d:.4f}", f"{r:.4f}", f"{r - d:+.4f}"])
    common.print_table(rows, ["指标", "Dense", "+Reranker", "Δ"])

    print("\n分类别 Hit@1 / MRR")
    rows = []
    for cat, v in by_category.items():
        rows.append([cat, v["dense"]["n"],
                     f"{v['dense']['Hit@1']:.4f}", f"{v['reranked']['Hit@1']:.4f}",
                     f"{v['dense']['MRR']:.4f}", f"{v['reranked']['MRR']:.4f}"])
    common.print_table(rows, ["类别", "n", "Dense H@1", "RR H@1", "Dense MRR", "RR MRR"])

    improved = sum(1 for d in details if d["rank_change"] and d["rank_change"] > 0)
    worsened = sum(1 for d in details if d["rank_change"] and d["rank_change"] < 0)
    print(f"\nReranker 使 {improved} 条 query 的首个相关 chunk 排名上升，{worsened} 条下降，"
          f"{len(details) - improved - worsened} 条不变。")

    common.save_results("exp1_retrieval.json", {
        "experiment": "retrieval_quality",
        "embedding_model": settings.embedding_model,
        "reranker_model": "BAAI/bge-reranker-v2-m3",
        "corpus_size": len(corpus),
        "corpus": [{"id": c["id"], "source": c["source"], "chars": len(c["text"])} for c in corpus],
        "n_queries": len(queries),
        "candidate_k": k,
        "dense": dense_metrics,
        "reranked": rerank_metrics,
        "by_category": by_category,
        "rank_improved": improved,
        "rank_worsened": worsened,
        "avg_dense_latency_s": round(sum(dense_latency) / len(dense_latency), 4),
        "avg_rerank_latency_s": round(sum(rerank_latency) / len(rerank_latency), 4),
        "details": details,
    })


if __name__ == "__main__":
    main()
