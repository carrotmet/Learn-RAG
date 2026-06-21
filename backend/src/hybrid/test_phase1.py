#!/usr/bin/env python3
"""二期第一阶段集成测试

测试内容：
1. PDF 导入 → 分块（父块 2000字）
2. 三通道索引（Vector/FTS5/Graph）
3. 检索验证
4. 数据表可见性检查

测试文档：/home/ubuntu/.openclaw/workspace/RAG教学/docs/自指学口播文稿_第三版.pdf
"""

import os
import sys

# 添加项目路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from langchain_core.documents import Document
from hybrid.document_store import DocumentStore, make_derivative_id, parse_derivative_id
from hybrid.channels.vector import VectorChannel
from hybrid.channels.fts import FTSChannel
from hybrid.channels.graph import GraphChannel


def load_pdf(path: str) -> list:
    """加载 PDF 文本"""
    try:
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
    except Exception as e:
        print(f"[错误] PDF 加载失败: {e}")
        sys.exit(1)


def test_phase1():
    """二期第一阶段测试主流程"""
    
    pdf_path = "/home/ubuntu/.openclaw/workspace/RAG教学/docs/自指学口播文稿_第三版.pdf"
    db_path = "/home/ubuntu/.openclaw/workspace/RAG教学/backend/data/rag_data.db"
    
    print("=" * 60)
    print("二期第一阶段测试：基础表 + DocumentStore + 三通道")
    print("=" * 60)
    
    # ── 1. 初始化 DocumentStore ────────────────────────────
    print("\n[1/7] 初始化 DocumentStore...")
    store = DocumentStore(db_path=db_path, chunk_size=2000, chunk_overlap=200)
    print(f"  → DocumentStore: {store}")
    
    # 清空已有数据（测试模式）
    print("  → 清空旧数据（测试模式）...")
    store.clear_all(confirm=True)
    
    # ── 2. 加载 PDF ────────────────────────────────────────
    print(f"\n[2/7] 加载 PDF: {os.path.basename(pdf_path)}")
    docs = load_pdf(pdf_path)
    print(f"  → 共 {len(docs)} 页")
    total_chars = sum(len(d.page_content) for d in docs)
    print(f"  → 总字数: {total_chars}")
    
    # ── 3. 分块并保存（父块） ─────────────────────────────
    print("\n[3/7] 文档分块（父块，chunk_size=2000）...")
    chunks = store.split_and_save(docs)
    print(f"  → 生成 {len(chunks)} 个父块")
    
    # 显示前3个块信息
    for i, c in enumerate(chunks[:3]):
        print(f"    chunk[{i}] id={c.chunk_id}, source={c.source}, page={c.page}, "
              f"chars={len(c.content)}")
        preview = c.content[:80].replace('\n', ' ')
        print(f"      预览: {preview}...")
    
    # ── 4. 初始化三通道（清空旧数据） ────────────────────
    print("\n[4/7] 初始化三通道...")
    
    # 清空 VectorChannel 旧集合
    try:
        import shutil
        chroma_dir = os.path.join(BASE_DIR, "src", "chroma_db_hybrid")
        if os.path.exists(chroma_dir):
            shutil.rmtree(chroma_dir)
            print("  → 已清空旧 ChromaDB")
    except Exception as e:
        print(f"  → 清空 ChromaDB 警告: {e}")
    
    # 清空 Graph 旧数据
    graph_path = os.path.join(BASE_DIR, "src", "data", "knowledge_graph.pkl")
    if os.path.exists(graph_path):
        os.remove(graph_path)
        print("  → 已清空旧 Graph")
    
    vector_ch = VectorChannel()
    fts_ch = FTSChannel(db_path=db_path)
    graph_ch = GraphChannel()
    
    # 清空 FTS5
    fts_ch.clear()
    print("  → 已清空 FTS5")
    
    print("  → VectorChannel (ChromaDB) 就绪")
    print("  → FTSChannel (SQLite FTS5) 就绪")
    print("  → GraphChannel (NetworkX) 就绪")
    
    # ── 5. 构建索引（Standard 策略） ───────────────────────
    print("\n[5/7] 构建索引（Standard 策略，三通道并行）...")
    
    items = []
    for c in chunks:
        meta = {
            "parent_chunk_id": c.chunk_id,
            "derivative_id": c.chunk_id,  # standard 策略无派生，用 chunk_id
            "strategy": "standard",
            "source": c.source,
            "page": c.page,
        }
        items.append({"content": c.content, "metadata": meta})
    
    # Vector
    print("  → 写入 VectorChannel...")
    vector_ch.add(items)
    vector_count = vector_ch.count()
    print(f"    完成: {vector_count} 条")
    
    # FTS5
    print("  → 写入 FTSChannel...")
    fts_ch.add(items)
    fts_count = fts_ch.count()
    print(f"    完成: {fts_count} 条")
    
    # Graph
    print("  → 写入 GraphChannel...")
    graph_ch.add(items)
    graph_stats = graph_ch.stats()
    print(f"    完成: {graph_stats}")
    
    # 标记索引状态
    for c in chunks:
        deriv_id = c.chunk_id  # standard 策略
        store.mark_indexed(deriv_id, "standard", "vector")
        store.mark_indexed(deriv_id, "standard", "fts")
        store.mark_indexed(deriv_id, "standard", "graph")
    
    # ── 6. 数据表可见性检查 ────────────────────────────────
    print("\n[6/7] 数据表可见性检查...")
    stats = store.get_stats()
    print(f"  → chunks 表: {stats['chunks']} 条")
    print(f"  → chunk_derivatives 表: {stats['derivatives']} 条")
    print(f"  → index_status 表: {stats['indexed']} 条已索引")
    
    # 验证 chunks 表内容
    print("\n  → chunks 表内容抽样（前3条）:")
    sample = store.get_all_chunks(limit=3)
    for c in sample:
        print(f"    chunk_id={c.chunk_id}, source={c.source}, page={c.page}, chars={len(c.content)}")
    
    # ── 7. 检索测试 ─────────────────────────────────────────
    print("\n[7/7] 检索测试...")
    
    # Graph 调试：打印一些关键词
    print("\n  [Graph 关键词抽样]:")
    graph_stats = graph_ch.stats()
    keyword_nodes = [n for n, d in graph_ch.G.nodes(data=True) if d.get("type") == "keyword"]
    print(f"    共 {len(keyword_nodes)} 个关键词")
    print(f"    前20个: {keyword_nodes[:20]}")
    
    test_queries = [
        "自指",
        "哥德尔",
        "大语言模型",
        "罗素",
    ]
    
    for query in test_queries:
        print(f"\n  [查询] \"{query}\"")
        
        # Vector 检索
        v_results = vector_ch.search(query, k=2)
        if v_results:
            print(f"    [Vector] 命中 {len(v_results)} 条")
            for r in v_results:
                print(f"      score={r['score']}, parent={r['parent_chunk_id']}, "
                      f"content={r['content'][:60]}...")
        else:
            print(f"    [Vector] 无结果")
        
        # FTS 检索
        f_results = fts_ch.search(query, k=2)
        if f_results:
            print(f"    [FTS5] 命中 {len(f_results)} 条")
            for r in f_results:
                print(f"      score={r['score']}, parent={r['parent_chunk_id']}, "
                      f"content={r['content'][:60]}...")
        else:
            print(f"    [FTS5] 无结果")
        
        # Graph 检索
        g_results = graph_ch.search(query, k=2)
        if g_results:
            print(f"    [Graph] 命中 {len(g_results)} 条")
            for r in g_results:
                print(f"      score={r['score']}, parent={r['parent_chunk_id']}, "
                      f"keywords={r.get('matched_keywords', [])}")
        else:
            print(f"    [Graph] 无结果")
    
    # ── 总结 ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    print(f"\n数据汇总:")
    print(f"  - 父块数量: {len(chunks)}")
    print(f"  - Vector 索引: {vector_count} 条")
    print(f"  - FTS5 索引: {fts_count} 条")
    print(f"  - Graph 节点: {graph_stats['nodes']} 个")
    print(f"  - Graph 边: {graph_stats['edges']} 条")
    print(f"\n数据表位置:")
    print(f"  - SQLite: {db_path}")
    print(f"  - ChromaDB: {vector_ch.persist_dir}")
    print(f"  - Graph: {graph_ch.path}")
    
    return True


if __name__ == "__main__":
    success = test_phase1()
    sys.exit(0 if success else 1)
