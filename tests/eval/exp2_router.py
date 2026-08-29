# -*- coding: utf-8 -*-
"""
实验 2（P0）：Router 分类准确率

Agent 的核心是路由；路由错了后面全错。本实验单独把 route_node 拉出来测：
- 20 条 query，覆盖 RAG / CALCULATOR / WEB_SEARCH / DIRECT 四条路
- 指标：整体准确率、各类 precision/recall、混淆矩阵、平均延迟
- 每条跑 N_RUNS 次，用来观察 LLM 路由的稳定性（同一 query 是否每次都路由到同一条路）

输出：tests/eval/results/exp2_router.json
"""

import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import common  # noqa: E402

common.setup_console()

LABELS = ["RAG", "CALCULATOR", "WEB_SEARCH", "DIRECT"]
N_RUNS = int(os.getenv("ROUTER_RUNS", "3"))   # 稳定性：同一 query 重复次数


ROUTER_PROMPT = None


def classify(question):
    """复用 app 里真实的 ROUTER_PROMPT 与解析逻辑，避免测的和跑的不是一套。"""
    from app.llm.client import chat

    raw = chat([{"role": "system",
                 "content": ROUTER_PROMPT.format(question=question)}]).strip().upper()

    if raw.startswith("RAG"):
        return "RAG", raw
    if raw.startswith("CALCULATOR"):
        return "CALCULATOR", raw
    if raw.startswith("WEB_SEARCH"):
        return "WEB_SEARCH", raw
    return "DIRECT", raw


def main():
    global ROUTER_PROMPT

    print("=" * 74)
    print("实验 2 / Router 分类准确率")
    print("=" * 74)

    ROUTER_PROMPT = common.load_prompt_constants()["ROUTER_PROMPT"]

    from app.config import settings
    print(f"\n路由模型：{settings.openai_model}    每条 query 重复 {N_RUNS} 次")

    dataset = common.load_dataset("router_queries.json")
    queries = dataset["queries"]

    details = []
    latencies = []
    confusion = defaultdict(Counter)
    correct = 0
    unstable = 0

    for item in queries:
        preds, raws = [], []
        for _ in range(N_RUNS):
            t0 = time.time()
            try:
                pred, raw = classify(item["query"])
            except Exception as e:            # 网络/额度问题不要中断整场评测
                pred, raw = "ERROR", f"{type(e).__name__}: {e}"
            latencies.append(time.time() - t0)
            preds.append(pred)
            raws.append(raw)

        majority = Counter(preds).most_common(1)[0][0]
        is_stable = len(set(preds)) == 1
        ok = majority == item["label"]

        correct += int(ok)
        unstable += int(not is_stable)
        confusion[item["label"]][majority] += 1

        details.append({
            "id": item["id"], "query": item["query"], "expected": item["label"],
            "predicted": majority, "runs": preds, "stable": is_stable,
            "correct": ok, "hard": item.get("hard", False),
            "raw_first": raws[0][:80],
        })

        mark = "OK  " if ok else "FAIL"
        stab = "" if is_stable else f"  (不稳定: {preds})"
        print(f"{mark} {item['id']} 期望={item['label']:<11} 实际={majority:<11} {item['query']}{stab}")

    n = len(queries)
    accuracy = correct / n

    # 各类 precision / recall
    per_label = {}
    for label in LABELS:
        tp = sum(1 for d in details if d["expected"] == label and d["predicted"] == label)
        fp = sum(1 for d in details if d["expected"] != label and d["predicted"] == label)
        fn = sum(1 for d in details if d["expected"] == label and d["predicted"] != label)
        per_label[label] = {
            "support": sum(1 for d in details if d["expected"] == label),
            "precision": tp / (tp + fp) if (tp + fp) else 0.0,
            "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        }

    print("\n" + "=" * 74)
    print(f"整体准确率：{correct}/{n} = {accuracy:.2%}")
    print(f"路由稳定性：{n - unstable}/{n} 条 query 在 {N_RUNS} 次重复中结果完全一致")
    print("=" * 74)

    print("\n各类别表现")
    common.print_table(
        [[l, per_label[l]["support"], f"{per_label[l]['precision']:.2f}",
          f"{per_label[l]['recall']:.2f}"] for l in LABELS],
        ["路由", "样本数", "precision", "recall"])

    print("\n混淆矩阵（行=期望，列=实际）")
    cols = LABELS + ["ERROR"]
    rows = [[l] + [confusion[l].get(c, 0) for c in cols] for l in LABELS]
    common.print_table(rows, ["期望\\实际"] + cols)

    errors = [d for d in details if not d["correct"]]
    if errors:
        print("\n错误样本：")
        for d in errors:
            print(f"  {d['id']} {d['expected']} -> {d['predicted']} | {d['query']}")

    common.save_results("exp2_router.json", {
        "experiment": "router_accuracy",
        "router_model": settings.openai_model,
        "n_queries": n,
        "runs_per_query": N_RUNS,
        "accuracy": accuracy,
        "correct": correct,
        "stability": (n - unstable) / n,
        "per_label": per_label,
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "avg_latency_s": round(sum(latencies) / len(latencies), 3),
        "details": details,
    })


if __name__ == "__main__":
    main()
