# -*- coding: utf-8 -*-
"""
评测公共模块：路径引导、语料构建、指标计算、结果落盘。

设计要点：
1. 每次评测都从 data/documents 重新切分并写入一个临时（内存）向量库，
   不依赖 data/chroma 的历史状态 —— 避免重复 ingest 造成的重复 chunk 污染指标。
2. chunk 用 "source_alias::chunk_index" 唯一标识，避免不同文件的同号 chunk 互相误判。
"""

import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DATASET_DIR = Path(__file__).resolve().parent / "datasets"


def setup_console():
    """
    Windows 默认 GBK(cp936)，输出被重定向到管道/文件时打印中文会抛
    UnicodeEncodeError 直接把脚本干掉。这里强制把 stdout/stderr 换成 UTF-8，
    并且 errors='replace'，保证任何情况下都不会因为"打印"而崩。
    """
    for stream in ("stdout", "stderr"):
        s = getattr(sys, stream, None)
        if s is None:
            continue
        enc = (getattr(s, "encoding", "") or "").lower().replace("-", "")
        if enc.startswith("utf8"):
            continue
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
            continue
        except Exception:
            pass
        try:
            setattr(sys, stream, io.TextIOWrapper(
                s.buffer, encoding="utf-8", errors="replace", line_buffering=True))
        except Exception:
            pass


def source_alias(path: Path) -> str:
    """把文件名映射成评测集里用的短别名。"""
    if path.suffix.lower() == ".pdf":
        return "resume"
    return path.stem


def load_prompt_constants():
    """
    从 app/agent/graph.py 里直接取 ROUTER_PROMPT / SYSTEM_PROMPT 的字面量。

    用 ast 解析而不是 import，是为了绕开 app.agent.tools 在 import 时就实例化
    TavilyClient 的副作用 —— 缺 TAVILY_API_KEY 时那一行会直接抛错，
    评测不该因为一个用不到的工具挂掉。取到的仍是线上同一份 prompt。
    """
    import ast

    src = (ROOT / "app" / "agent" / "graph.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out[target.id] = node.value.value
    return out


def format_context(results):
    """与 app/agent/tools.py 的 format_context 保持一致（同样理由：避免 import 副作用）。"""
    if not results:
        return "No relevant documents were found."
    return "\n\n".join(
        f"[Source {i}: {x['metadata'].get('source', 'unknown')}, "
        f"chunk {x['metadata'].get('chunk_index', '?')}]\n{x['text']}"
        for i, x in enumerate(results, 1)
    )


def load_corpus():
    """用项目自己的 loader + splitter 重建语料，返回 chunk 列表。"""
    from app.config import settings
    from app.rag.loader import load_file, SUPPORTED
    from app.rag.splitter import split_text

    root = Path(settings.document_dir)
    if not root.is_absolute():
        root = ROOT / root

    corpus = []
    for path in sorted(root.iterdir()):
        if not (path.is_file() and path.suffix.lower() in SUPPORTED):
            continue
        alias = source_alias(path)
        for i, chunk in enumerate(split_text(load_file(path))):
            corpus.append({
                "id": f"{alias}::{i}",
                "source": path.name,
                "alias": alias,
                "chunk_index": i,
                "text": chunk,
            })
    return corpus


def build_ephemeral_collection(corpus, embed_fn=None):
    """把语料写进一个进程内的临时 Chroma collection（不落盘、不污染生产库）。"""
    import chromadb

    if embed_fn is None:
        from app.rag.embeddings import embed_documents as embed_fn

    client = chromadb.EphemeralClient()
    collection = client.create_collection(name="eval_kb")
    collection.add(
        ids=[c["id"] for c in corpus],
        documents=[c["text"] for c in corpus],
        embeddings=embed_fn([c["text"] for c in corpus]),
        metadatas=[{"alias": c["alias"], "chunk_index": c["chunk_index"],
                    "source": c["source"]} for c in corpus],
    )
    return collection


# --------------------------------------------------
# 指标
# --------------------------------------------------

def hit_at_k(ranked_ids, relevant, k):
    return int(any(i in relevant for i in ranked_ids[:k]))


def recall_at_k(ranked_ids, relevant, k):
    if not relevant:
        return 0.0
    return len(set(ranked_ids[:k]) & set(relevant)) / len(relevant)


def mrr(ranked_ids, relevant):
    for rank, i in enumerate(ranked_ids, start=1):
        if i in relevant:
            return 1.0 / rank
    return 0.0


def first_relevant_rank(ranked_ids, relevant):
    for rank, i in enumerate(ranked_ids, start=1):
        if i in relevant:
            return rank
    return None


def aggregate(per_query):
    """per_query: [{'ranked': [...], 'relevant': [...]}] -> 指标字典"""
    n = len(per_query)
    if n == 0:
        return {}
    return {
        "Hit@1": sum(hit_at_k(q["ranked"], q["relevant"], 1) for q in per_query) / n,
        "Hit@3": sum(hit_at_k(q["ranked"], q["relevant"], 3) for q in per_query) / n,
        "Hit@5": sum(hit_at_k(q["ranked"], q["relevant"], 5) for q in per_query) / n,
        "Recall@3": sum(recall_at_k(q["ranked"], q["relevant"], 3) for q in per_query) / n,
        "Recall@5": sum(recall_at_k(q["ranked"], q["relevant"], 5) for q in per_query) / n,
        "MRR": sum(mrr(q["ranked"], q["relevant"]) for q in per_query) / n,
        "n": n,
    }


# --------------------------------------------------
# IO
# --------------------------------------------------

def load_dataset(name):
    with open(DATASET_DIR / name, "r", encoding="utf-8") as f:
        return json.load(f)


def save_results(name, payload):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload.setdefault("generated_at", time.strftime("%Y-%m-%d %H:%M:%S"))
    path = RESULTS_DIR / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n[saved] {path}")
    return path


def print_table(rows, headers):
    widths = [max(len(str(r[i])) for r in [headers] + rows) + 2
              for i in range(len(headers))]
    line = "".join(str(h).ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("".join(str(c).ljust(w) for c, w in zip(r, widths)))
