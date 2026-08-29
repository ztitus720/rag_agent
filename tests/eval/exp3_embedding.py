# -*- coding: utf-8 -*-
"""
实验 3（P1）：Embedding 模型横向对比

对比 all-MiniLM-L6-v2（英文为主，384 维）与 bge-small-zh-v1.5（中文专用，512 维）
在同一份中文评测集上的纯向量检索表现（不带 Reranker，隔离出 embedding 本身的贡献）。

bge 系列官方建议检索场景给 query 加指令前缀，这里两种写法都测，用来说明
"换模型不是换个名字，配套的用法也要跟着改"。

需要联网下载 bge-small-zh-v1.5（约 95MB）。国内默认走 hf-mirror.com；
下载失败会跳过该模型并如实记录，不会让整场评测失败。

输出：tests/eval/results/exp3_embedding.json
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# 本实验需要联网拉模型，因此不设 OFFLINE；国内默认走镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

import numpy as np  # noqa: E402

import common  # noqa: E402

common.setup_console()

BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

MODELS = [
    {"key": "all-MiniLM-L6-v2", "name": "sentence-transformers/all-MiniLM-L6-v2",
     "query_prefix": "", "note": "当前线上使用；英文语料训练，多语言能力有限"},
    {"key": "bge-small-zh-v1.5", "name": "BAAI/bge-small-zh-v1.5",
     "query_prefix": "", "note": "中文专用，未加指令前缀"},
    {"key": "bge-small-zh-v1.5 (+instruction)", "name": "BAAI/bge-small-zh-v1.5",
     "query_prefix": BGE_QUERY_PREFIX, "note": "中文专用，按官方建议给 query 加检索指令前缀"},
]


def rank_ids(model, corpus, query, prefix):
    doc_vecs = model["doc_vecs"]
    q = model["st"].encode([prefix + query], normalize_embeddings=True)[0]
    sims = doc_vecs @ q
    order = np.argsort(-sims)
    return [corpus[i]["id"] for i in order], float(sims[order[0]])


def main():
    print("=" * 74)
    print("实验 3 / Embedding 模型横向对比（纯向量检索，无 Reranker）")
    print("=" * 74)

    from sentence_transformers import SentenceTransformer

    corpus = common.load_corpus()
    texts = [c["text"] for c in corpus]
    dataset = common.load_dataset("retrieval_queries.json")
    queries = dataset["queries"]

    print(f"\n语料 {len(corpus)} 个 chunk，评测集 {len(queries)} 条 query")

    loaded = {}
    results = {}

    for cfg in MODELS:
        print(f"\n--- {cfg['key']} ---")

        if cfg["name"] not in loaded:
            try:
                t0 = time.time()
                st = SentenceTransformer(cfg["name"])
                print(f"模型加载完成，用时 {time.time() - t0:.1f}s")
            except Exception as e:
                print(f"[跳过] 加载失败：{type(e).__name__}: {e}")
                results[cfg["key"]] = {"status": "skipped",
                                       "reason": f"{type(e).__name__}: {e}"}
                continue

            t0 = time.time()
            doc_vecs = np.asarray(st.encode(texts, normalize_embeddings=True))
            encode_time = time.time() - t0
            loaded[cfg["name"]] = {"st": st, "doc_vecs": doc_vecs,
                                   "dim": int(doc_vecs.shape[1]),
                                   "encode_time": encode_time}
            print(f"向量维度 {doc_vecs.shape[1]}，编码 {len(texts)} 个 chunk 用时 {encode_time:.2f}s")

        model = loaded[cfg["name"]]

        per_query, details = [], []
        t0 = time.time()
        for item in queries:
            ranked, top_sim = rank_ids(model, corpus, item["query"], cfg["query_prefix"])
            per_query.append({"ranked": ranked, "relevant": item["relevant"]})
            details.append({
                "id": item["id"], "category": item["category"],
                "first_relevant_rank": common.first_relevant_rank(ranked, item["relevant"]),
                "top1": ranked[0], "top1_sim": round(top_sim, 4),
            })
        query_time = (time.time() - t0) / len(queries)

        metrics = common.aggregate(per_query)
        by_category = {}
        for cat in ("fact", "paraphrase", "multi"):
            idx = [i for i, q in enumerate(queries) if q["category"] == cat]
            if idx:
                by_category[cat] = common.aggregate([per_query[i] for i in idx])

        results[cfg["key"]] = {
            "status": "ok", "model": cfg["name"], "note": cfg["note"],
            "query_prefix": cfg["query_prefix"], "dim": model["dim"],
            "corpus_encode_time_s": round(model["encode_time"], 3),
            "avg_query_time_s": round(query_time, 4),
            "metrics": metrics, "by_category": by_category, "details": details,
        }

        print(f"Hit@1={metrics['Hit@1']:.4f}  Hit@3={metrics['Hit@3']:.4f}  MRR={metrics['MRR']:.4f}")

    ok = {k: v for k, v in results.items() if v.get("status") == "ok"}

    if ok:
        print("\n" + "=" * 74)
        print("对比结果（纯 Dense 检索）")
        print("=" * 74)
        common.print_table(
            [[k, v["dim"],
              f"{v['metrics']['Hit@1']:.4f}", f"{v['metrics']['Hit@3']:.4f}",
              f"{v['metrics']['MRR']:.4f}",
              f"{v['by_category'].get('paraphrase', {}).get('Hit@1', 0):.4f}",
              f"{v['avg_query_time_s'] * 1000:.1f}ms"] for k, v in ok.items()],
            ["模型", "维度", "Hit@1", "Hit@3", "MRR", "改写类 Hit@1", "单次编码"])

    skipped = {k: v for k, v in results.items() if v.get("status") == "skipped"}
    if skipped:
        print("\n未完成的模型：")
        for k, v in skipped.items():
            print(f"  {k}: {v['reason']}")

    common.save_results("exp3_embedding.json", {
        "experiment": "embedding_comparison",
        "corpus_size": len(corpus),
        "n_queries": len(queries),
        "hf_endpoint": os.environ.get("HF_ENDPOINT", ""),
        "models": results,
    })


if __name__ == "__main__":
    main()
