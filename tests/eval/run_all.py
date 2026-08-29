# -*- coding: utf-8 -*-
"""
一键跑完全部评测实验。

用法（在项目根目录 D:\\RAG-Agent 下，先激活 .venv）：

    python tests/eval/run_all.py              # 全部
    python tests/eval/run_all.py 1 2          # 只跑实验 1 和 2

每个实验在独立子进程中运行，互不影响：
    实验 1  检索质量 Dense vs Reranker   （本地模型，无需联网，不花 token）
    实验 2  Router 分类准确率            （需要 .env 里的 LLM key，约 60 次调用）
    实验 3  Embedding 横向对比           （需要联网下载 bge-small-zh，约 95MB）
    实验 4  不可回答问题拒答率           （需要 LLM key，约 32 次调用）

结果写入 tests/eval/results/*.json 与 summary.json
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import common  # noqa: E402

common.setup_console()

EXPERIMENTS = [
    ("1", "exp1_retrieval.py", "检索质量：Dense vs BGE Reranker", "exp1_retrieval.json"),
    ("2", "exp2_router.py", "Router 分类准确率", "exp2_router.json"),
    ("3", "exp3_embedding.py", "Embedding 横向对比", "exp3_embedding.json"),
    ("4", "exp4_unanswerable.py", "不可回答问题拒答率", "exp4_unanswerable.json"),
]


def run(script):
    """
    跑一个实验子进程，同时把输出打到屏幕并存进 results/log_<script>.txt。
    子进程崩了的时候，日志文件就是排查现场。
    """
    print("\n" + "#" * 74)
    print(f"# {script}")
    print("#" * 74 + "\n", flush=True)

    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = common.RESULTS_DIR / f"log_{Path(script).stem}.txt"

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"      # 防止 Windows GBK 控制台把中文输出打崩
    env["PYTHONUNBUFFERED"] = "1"

    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(HERE / script)],
        cwd=str(ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    elapsed = time.time() - t0

    output = proc.stdout.decode("utf-8", errors="replace")
    print(output, flush=True)
    log_path.write_text(output, encoding="utf-8")

    tail = [l for l in output.strip().splitlines() if l.strip()][-15:]
    return proc.returncode, elapsed, str(log_path.name), tail


REQUIRED = [
    ("pydantic_settings", "pydantic-settings"),
    ("dotenv", "python-dotenv"),
    ("pypdf", "pypdf"),
    ("numpy", "numpy"),
    ("chromadb", "chromadb"),
    ("sentence_transformers", "sentence-transformers"),
    ("openai", "openai"),
]


def preflight():
    """
    先确认解释器环境是对的。上一次全部实验在 0.2 秒内退出，
    最常见的原因就是没用项目 .venv 里的 python，依赖一个都不在。
    """
    import importlib.util

    print("=" * 74)
    print("环境自检")
    print("=" * 74)
    print(f"解释器：{sys.executable}")
    print(f"项目根：{ROOT}")

    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    print(f"虚拟环境：{'是' if in_venv else '否 —— 注意，当前不在 venv 里'}")

    missing = [pip_name for mod, pip_name in REQUIRED
               if importlib.util.find_spec(mod) is None]

    env_file = ROOT / ".env"
    print(f".env：{'找到' if env_file.exists() else '缺失（实验 2/4 需要 LLM key）'}")

    if missing:
        print("\n缺少依赖：" + ", ".join(missing))
        print("\n多半是没用项目虚拟环境里的 python。请改用：")
        print(f'    {ROOT}\\.venv\\Scripts\\python.exe tests\\eval\\run_all.py')
        print("\n或者先激活环境再跑：")
        print(f'    {ROOT}\\.venv\\Scripts\\activate')
        print("    python tests\\eval\\run_all.py")
        return False

    print("依赖：齐全")
    return True


def main():
    wanted = set(sys.argv[1:]) or {e[0] for e in EXPERIMENTS}

    if not preflight():
        sys.exit(1)

    status = {}
    for key, script, title, out in EXPERIMENTS:
        if key not in wanted:
            continue
        code, elapsed, log_name, tail = run(script)
        status[title] = {
            "script": script,
            "exit_code": code,
            "elapsed_s": round(elapsed, 1),
            "result_file": out if (common.RESULTS_DIR / out).exists() else None,
            "log_file": log_name,
        }
        if code != 0:
            status[title]["output_tail"] = tail

    # 汇总
    summary = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": status,
               "headline": {}}

    def load(name):
        p = common.RESULTS_DIR / name
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    e1 = load("exp1_retrieval.json")
    if e1:
        summary["headline"]["retrieval"] = {
            "n_queries": e1["n_queries"], "corpus_size": e1["corpus_size"],
            "dense_hit1": e1["dense"]["Hit@1"], "reranked_hit1": e1["reranked"]["Hit@1"],
            "dense_mrr": e1["dense"]["MRR"], "reranked_mrr": e1["reranked"]["MRR"],
        }
    e2 = load("exp2_router.json")
    if e2:
        summary["headline"]["router"] = {
            "n_queries": e2["n_queries"], "accuracy": e2["accuracy"],
            "stability": e2["stability"], "avg_latency_s": e2["avg_latency_s"],
        }
    e3 = load("exp3_embedding.json")
    if e3:
        summary["headline"]["embedding"] = {
            k: (v["metrics"]["Hit@1"] if v.get("status") == "ok" else v.get("status"))
            for k, v in e3["models"].items()}
    e4 = load("exp4_unanswerable.json")
    if e4:
        summary["headline"]["refusal"] = {
            "n_queries": e4["n_queries"],
            "rag_refusal_rate": e4["rag_refusal_rate"],
            "baseline_refusal_rate": e4["baseline_refusal_rate"],
        }

    common.save_results("summary.json", summary)

    print("\n" + "=" * 74)
    print("全部实验完成")
    print("=" * 74)
    for title, s in status.items():
        mark = "OK  " if s["exit_code"] == 0 else f"FAIL({s['exit_code']})"
        print(f"{mark}  {title:<34} {s['elapsed_s']:>7.1f}s  -> {s['result_file'] or s['log_file']}")
    print(f"\n结果目录：{common.RESULTS_DIR}")
    print("把 tests/eval/results/ 下的 json 发回给我，我来写进 README。")


if __name__ == "__main__":
    main()
