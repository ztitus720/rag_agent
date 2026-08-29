# -*- coding: utf-8 -*-
"""
实验 4（P2）：不可回答问题的拒答率（抗幻觉）

RAG 真正的风险不是"答得不够好"，而是"知识库里没有却编一个出来"。
本实验用 8 条知识库中确实没有答案的问题跑完整 RAG 链路（检索 -> 生成），
再用 LLM 作为裁判判断回答属于「明确说明信息不可得」还是「给出了具体断言」。

对照组：把同样的 8 条问题直接问 LLM（不走 RAG、不给 context），
用来说明"拒答能力来自 prompt 约束 + 检索上下文"，而不是模型本来就保守。

输出：tests/eval/results/exp4_unanswerable.json
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import common  # noqa: E402

common.setup_console()

TOP_K = 4

REFUSAL_KEYWORDS = [
    "未提", "没有提", "不包含", "未包含", "未提供", "没有提供", "无法确定",
    "无法回答", "没有相关", "无相关", "未涉及", "没有涉及", "未说明", "没有说明",
    "信息不足", "不可用", "not available", "does not contain", "no information",
    "not provided", "not mentioned",
]

JUDGE_PROMPT = """你是一个严格的评审。下面是一个问答系统对某个问题的回答。

问题：{question}

回答：{answer}

请判断这个回答属于以下哪一类，只输出一个词：

REFUSE —— 明确说明所提供的资料中没有这个信息 / 无法回答，且没有给出具体的事实性断言。
ANSWER —— 给出了具体的事实性断言（数字、名称、经历、结论等），无论对错。
PARTIAL —— 说明资料不足，但同时又给出了具体的推测性断言。

只输出 REFUSE、ANSWER 或 PARTIAL 之一。"""


def heuristic_refuse(answer):
    low = answer.lower()
    return any(k.lower() in low for k in REFUSAL_KEYWORDS)


def judge(question, answer):
    from app.llm.client import chat
    try:
        r = chat([{"role": "system",
                   "content": JUDGE_PROMPT.format(question=question, answer=answer)}],
                 temperature=0).strip().upper()
    except Exception as e:
        return f"JUDGE_ERROR: {type(e).__name__}"
    for label in ("REFUSE", "PARTIAL", "ANSWER"):
        if label in r:
            return label
    return "UNKNOWN"


def main():
    print("=" * 74)
    print("实验 4 / 不可回答问题的拒答率")
    print("=" * 74)

    SYSTEM_PROMPT = common.load_prompt_constants()["SYSTEM_PROMPT"]
    format_context = common.format_context

    from app.llm.client import chat
    from app.rag.embeddings import embed_query

    corpus = common.load_corpus()
    collection = common.build_ephemeral_collection(corpus)
    print(f"\n语料 {len(corpus)} 个 chunk，检索 top-{TOP_K}")

    dataset = common.load_dataset("unanswerable_queries.json")
    queries = dataset["queries"]

    details = []
    rag_refuse = 0
    baseline_refuse = 0

    for item in queries:
        q = item["query"]

        r = collection.query(query_embeddings=[embed_query(q)],
                             n_results=min(TOP_K, len(corpus)),
                             include=["documents", "metadatas"])
        results = [{"text": d, "metadata": m}
                   for d, m in zip(r["documents"][0], r["metadatas"][0])]
        context = format_context(results)

        # --- RAG 链路 ---
        try:
            rag_answer = chat([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content":
                 f"User question:\n{q}\n\nSelected tool:\nRAG\n\n"
                 f"Knowledge base context:\n{context}\n\n"
                 f"Tool result:\nNo tool result.\n\nProvide the final answer to the user.\n"},
            ])
        except Exception as e:
            rag_answer = f"[ERROR] {type(e).__name__}: {e}"

        # --- 对照组：不给上下文，直接问 ---
        try:
            base_answer = chat([{"role": "user", "content": q}])
        except Exception as e:
            base_answer = f"[ERROR] {type(e).__name__}: {e}"

        rag_verdict = judge(q, rag_answer)
        base_verdict = judge(q, base_answer)

        rag_refuse += int(rag_verdict == "REFUSE")
        baseline_refuse += int(base_verdict == "REFUSE")

        details.append({
            "id": item["id"], "query": q, "plausible": item.get("plausible", False),
            "retrieved": [x["metadata"].get("alias", "") + "::" +
                          str(x["metadata"].get("chunk_index", "")) for x in results],
            "rag_answer": rag_answer,
            "rag_verdict": rag_verdict,
            "rag_keyword_refuse": heuristic_refuse(rag_answer),
            "baseline_answer": base_answer,
            "baseline_verdict": base_verdict,
        })

        print(f"\n{item['id']}  {q}")
        print(f"  RAG      [{rag_verdict:<7}] {rag_answer[:110].replace(chr(10), ' ')}")
        print(f"  无上下文 [{base_verdict:<7}] {base_answer[:110].replace(chr(10), ' ')}")

    n = len(queries)

    print("\n" + "=" * 74)
    print(f"RAG 链路拒答率：   {rag_refuse}/{n} = {rag_refuse / n:.2%}")
    print(f"无上下文对照组：   {baseline_refuse}/{n} = {baseline_refuse / n:.2%}")
    print("=" * 74)

    kw = sum(1 for d in details if d["rag_keyword_refuse"])
    print(f"\n（关键词启发式判定的拒答数：{kw}/{n}，与 LLM 裁判对照用）")

    leaks = [d for d in details if d["rag_verdict"] in ("ANSWER", "PARTIAL")]
    if leaks:
        print("\n未拒答的样本（需要人工复核是否为幻觉）：")
        for d in leaks:
            print(f"  {d['id']} [{d['rag_verdict']}] {d['query']}")

    common.save_results("exp4_unanswerable.json", {
        "experiment": "unanswerable_refusal",
        "n_queries": n,
        "top_k": TOP_K,
        "rag_refusal_rate": rag_refuse / n,
        "baseline_refusal_rate": baseline_refuse / n,
        "keyword_refusal_count": kw,
        "details": details,
    })


if __name__ == "__main__":
    main()
