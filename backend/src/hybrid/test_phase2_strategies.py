#!/usr/bin/env python3
"""二期策略索引层测试（5.1-5.3）- 分步版本

先测试 standard + parent_child（无需 LLM），再测试 summary + hypothetical（需 LLM）。
测试文档：自指学口播文稿_第三版.pdf
"""

import os
import sys
import shutil
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, "..", ".env"))

from langchain_core.documents import Document
from hybrid.document_store import DocumentStore, make_derivative_id, parse_derivative_id
from hybrid.channels.vector import VectorChannel
from hybrid.channels.fts import FTSChannel
from hybrid.channels.graph import GraphChannel
from hybrid.strategies.standard import StandardStrategy
from hybrid.strategies.summary import SummaryStrategy
from hybrid.strategies.parent_child import ParentChildStrategy
from hybrid.strategies.hypothetical import HypotheticalStrategy


def load_pdf(path):
    import pdfplumber
    docs = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                docs.append(Document(page_content=text.strip(),
                                     metadata={"source": os.path.basename(path), "page": i + 1}))
    return docs


def reset_data(db_path):
    print("\n[准备] 重置测试数据...")
    if os.path.exists(db_path):
        os.remove(db_path)
    chroma_dir = os.path.join(BASE_DIR, "chroma_db_hybrid")
    if os.path.exists(chroma_dir):
        shutil.rmtree(chroma_dir)
    graph_path = os.path.join(BASE_DIR, "data", "knowledge_graph.pkl")
    if os.path.exists(graph_path):
        os.remove(graph_path)
    print("  → 数据重置完成")


def verify_strategy_db(store, strategy_name):
    import sqlite3
    print(f"\n  [{strategy_name}] 数据库验证:")
    with sqlite3.connect(store.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT COUNT(*) as cnt FROM chunks").fetchone()
        print(f"    chunks: {row['cnt']} 条")
        rows = conn.execute(
            "SELECT COUNT(*) as cnt FROM chunk_derivatives WHERE strategy=?",
            (strategy_name,)
        ).fetchone()
        deriv_count = rows['cnt']
        print(f"    derivatives (strategy='{strategy_name}'): {deriv_count} 条")
        rows = conn.execute(
            """SELECT channel, COUNT(*) as cnt FROM index_status
               WHERE strategy=? AND indexed=1 GROUP BY channel""",
            (strategy_name,)
        ).fetchall()
        indexed_count = sum(r['cnt'] for r in rows)
        for r in rows:
            print(f"    indexed ({r['channel']}): {r['cnt']} 条")
        if deriv_count > 0:
            sample = conn.execute(
                "SELECT derivative_id, content FROM chunk_derivatives WHERE strategy=? LIMIT 1",
                (strategy_name,)
            ).fetchone()
            tc, cid, seq = parse_derivative_id(sample["derivative_id"])
            preview = sample["content"][:80].replace("\n", " ")
            print(f"    抽样: type={tc}, chunk={cid}, seq={seq}, content={preview}...")
    # standard 策略不创建 derivatives，用 index_status 验证
    if strategy_name == "standard":
        return indexed_count > 0
    return deriv_count > 0


def test_strategy(name, strategy, chunks, store, vector, fts, graph):
    print(f"\n{'='*60}")
    print(f"测试策略: {name}")
    print(f"{'='*60}")
    print(f"[Build] 构建索引...")
    count = strategy.build(chunks, store, vector=vector, fts=fts, graph=graph)
    print(f"  → 索引数量: {count}")
    stored = verify_strategy_db(store, name)

    print(f"[Search] 检索测试...")
    for query in ["自指", "哥德尔"]:
        results = strategy.search(query, store, vector=vector, fts=fts, graph=graph, k=2)
        print(f"  \"{query}\": 命中 {len(results)} 条")
        for r in results[:1]:
            preview = r.get("content", "")[:60].replace("\n", " ")
            print(f"    score={r.get('score', 0):.4f}, parent={r.get('parent_chunk_id')}, {preview}...")
    return stored


def main():
    pdf_path = "/home/ubuntu/.openclaw/workspace/RAG教学/docs/自指学口播文稿_第三版.pdf"
    db_path = "/home/ubuntu/.openclaw/workspace/RAG教学/backend/data/rag_data_phase2.db"

    print("=" * 60)
    print("二期策略索引层测试（5.1-5.3）")
    print("=" * 60)

    reset_data(db_path)

    print("\n[1/5] 初始化 DocumentStore...")
    store = DocumentStore(db_path=db_path, chunk_size=2000, chunk_overlap=200)

    print("\n[2/5] 加载 PDF...")
    docs = load_pdf(pdf_path)
    print(f"  → {len(docs)} 页, {sum(len(d.page_content) for d in docs)} 字")

    print("\n[3/5] 分块保存...")
    chunks = store.split_and_save(docs)
    print(f"  → {len(chunks)} 个父块")

    print("\n[4/5] 初始化三通道...")
    vector = VectorChannel()
    fts = FTSChannel(db_path=db_path)
    graph = GraphChannel()
    print("  → Vector, FTS, Graph 就绪")

    print("\n[5/5] 测试四种策略...")
    results = {}

    # 1. standard（无需 LLM）
    results["standard"] = test_strategy("standard", StandardStrategy(), chunks, store, vector, fts, graph)

    # 2. parent_child（无需 LLM）
    results["parent_child"] = test_strategy("parent_child", ParentChildStrategy(), chunks, store, vector, fts, graph)

    # 3. summary（需 LLM）- 带重试
    print("\n[summary] 需要 LLM，尝试调用...")
    for attempt in range(3):
        try:
            results["summary"] = test_strategy("summary", SummaryStrategy(), chunks, store, vector, fts, graph)
            break
        except Exception as e:
            print(f"  尝试 {attempt+1}/3 失败: {e}")
            if attempt < 2:
                wait = 30
                print(f"  等待 {wait} 秒后重试...")
                time.sleep(wait)
            else:
                print("  → summary 策略最终失败")
                results["summary"] = False

    # 4. hypothetical（需 LLM）- 带重试
    print("\n[hypothetical] 需要 LLM，尝试调用...")
    for attempt in range(3):
        try:
            results["hypothetical"] = test_strategy("hypothetical", HypotheticalStrategy(n=3), chunks, store, vector, fts, graph)
            break
        except Exception as e:
            print(f"  尝试 {attempt+1}/3 失败: {e}")
            if attempt < 2:
                wait = 30
                print(f"  等待 {wait} 秒后重试...")
                time.sleep(wait)
            else:
                print("  → hypothetical 策略最终失败")
                results["hypothetical"] = False

    # 汇总
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    for name, ok in results.items():
        print(f"  {name}: {'✅ 通过' if ok else '❌ 失败'}")

    stats = store.get_stats()
    print(f"\n数据库: chunks={stats['chunks']}, derivatives={stats['derivatives']}, indexed={stats['indexed']}")

    all_ok = all(results.values())
    print(f"\n{'✅ 全部通过' if all_ok else '⚠️ 部分失败'}")
    return all_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
