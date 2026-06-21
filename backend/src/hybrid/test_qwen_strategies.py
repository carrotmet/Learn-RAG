#!/usr/bin/env python3
"""二期策略索引层测试（5.1-5.3）- 使用 Qwen 模型

测试内容：
1. standard + parent_child（无需 LLM）
2. summary（LLM 生成摘要）
3. hypothetical（LLM 生成假设问题）

使用模型: qwen/qwen3.5-flash-02-23
"""

import os
import sys
import shutil
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, "..", ".env"))

print(f"[模型] DEFAULT_MODEL={os.getenv('DEFAULT_MODEL')}")

from langchain_core.documents import Document
from hybrid.document_store import DocumentStore, parse_derivative_id
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
                docs.append(Document(
                    page_content=text.strip(),
                    metadata={"source": os.path.basename(path), "page": i + 1}
                ))
    return docs


# 配置
pdf_path = "/home/ubuntu/.openclaw/workspace/RAG教学/docs/自指学口播文稿_第三版.pdf"
db_path = "/home/ubuntu/.openclaw/workspace/RAG教学/backend/data/rag_data_phase2.db"

# 重置数据
print("\n[准备] 重置测试数据...")
if os.path.exists(db_path):
    os.remove(db_path)
for p in [
    os.path.join(BASE_DIR, "chroma_db_hybrid"),
    os.path.join(BASE_DIR, "data", "knowledge_graph.pkl")
]:
    if os.path.isdir(p):
        shutil.rmtree(p)
    elif os.path.exists(p):
        os.remove(p)
print("  → 完成")

# 初始化
print("\n[1/4] 初始化...")
store = DocumentStore(db_path=db_path, chunk_size=2000, chunk_overlap=200)

print("\n[2/4] 加载 PDF...")
docs = load_pdf(pdf_path)
print(f"  → {len(docs)} 页, {sum(len(d.page_content) for d in docs)} 字")

print("\n[3/4] 分块保存...")
chunks = store.split_and_save(docs)
print(f"  → {len(chunks)} 个父块")

print("\n[4/4] 初始化三通道...")
vector = VectorChannel()
fts = FTSChannel(db_path=db_path)
graph = GraphChannel()
print("  → 完成")

# 测试 standard
print("\n" + "=" * 60)
print("策略: standard")
print("=" * 60)
strategy = StandardStrategy()
count = strategy.build(chunks, store, vector=vector, fts=fts, graph=graph)
print(f"  → 索引: {count}")

# 测试 parent_child
print("\n" + "=" * 60)
print("策略: parent_child")
print("=" * 60)
strategy = ParentChildStrategy()
count = strategy.build(chunks, store, vector=vector, fts=fts, graph=graph)
print(f"  → 索引: {count}")

# 测试 summary（使用 LLM）
print("\n" + "=" * 60)
print("策略: summary (LLM 生成摘要)")
print("=" * 60)
strategy = SummaryStrategy()
t0 = time.time()
count = strategy.build(chunks, store, vector=vector, fts=fts, graph=graph)
t1 = time.time()
print(f"  → 索引: {count}, 耗时: {t1-t0:.1f} 秒")

# 验证摘要质量
print("\n  [摘要质量检查]")
import sqlite3
with sqlite3.connect(store.db_path) as conn:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT chunk_id, content FROM chunk_derivatives WHERE strategy='summary' LIMIT 3"
    ).fetchall()
    for r in rows:
        print(f"    chunk {r['chunk_id']}: {r['content'][:100]}...")

# 测试 hypothetical（使用 LLM）
print("\n" + "=" * 60)
print("策略: hypothetical (LLM 生成假设问题)")
print("=" * 60)
strategy = HypotheticalStrategy(n=3)
t0 = time.time()
count = strategy.build(chunks, store, vector=vector, fts=fts, graph=graph)
t1 = time.time()
print(f"  → 索引: {count}, 耗时: {t1-t0:.1f} 秒")

# 验证问题质量
print("\n  [假设问题质量检查]")
with sqlite3.connect(store.db_path) as conn:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT chunk_id, content, metadata FROM chunk_derivatives 
           WHERE strategy='hypothetical' ORDER BY chunk_id, derivative_id LIMIT 6"""
    ).fetchall()
    current_chunk = None
    for r in rows:
        if r['chunk_id'] != current_chunk:
            current_chunk = r['chunk_id']
            print(f"    chunk {current_chunk}:")
        print(f"      Q: {r['content']}")

# 最终统计
print("\n" + "=" * 60)
print("测试结果汇总")
print("=" * 60)
stats = store.get_stats()
print(f"  chunks: {stats['chunks']}")
print(f"  derivatives: {stats['derivatives']}")
print(f"  indexed: {stats['indexed']}")

with sqlite3.connect(store.db_path) as conn:
    rows = conn.execute(
        "SELECT strategy, COUNT(*) as cnt FROM chunk_derivatives GROUP BY strategy"
    ).fetchall()
    print(f"\n  各策略派生数量:")
    for r in rows:
        print(f"    {r['strategy']}: {r['cnt']}")

print("\n✅ 全部策略测试完成")
