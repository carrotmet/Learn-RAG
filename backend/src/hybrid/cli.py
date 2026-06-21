#!/usr/bin/env python3
"""HybridRAG CLI 工具

用法示例:
    # 索引文档（使用全部策略）
    python -m hybrid.cli index docs/文档.pdf --db backend/data/rag.db

    # 仅使用标准策略
    python -m hybrid.cli index docs/文档.pdf --strategies standard

    # 使用标准+摘要策略
    python -m hybrid.cli index docs/文档.pdf --strategies standard,summary

    # 重置后索引
    python -m hybrid.cli index docs/文档.pdf --reset

    # 查看状态
    python -m hybrid.cli status --db backend/data/rag.db

    # 检索测试
    python -m hybrid.cli search "自指" --db backend/data/rag.db
"""

import os
import sys
import argparse

# 添加项目路径（支持从任意目录运行）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# __file__ 是 hybrid/cli.py，上两级是 backend/src，再上两级是项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend", "src"))

DEFAULT_DB = os.path.join(PROJECT_ROOT, "backend", "data", "rag_data.db")
DEFAULT_STRATEGIES = "standard,summary,parent_child,hypothetical"


def _resolve_pdf_path(pdf_path: str) -> str:
    """解析 PDF 路径：支持相对路径、绝对路径和项目根目录相对路径"""
    # 1. 绝对路径直接返回
    if os.path.isabs(pdf_path):
        return pdf_path

    # 2. 当前目录相对路径
    if os.path.exists(pdf_path):
        return os.path.abspath(pdf_path)

    # 3. 尝试从项目根目录解析
    project_relative = os.path.join(PROJECT_ROOT, pdf_path)
    if os.path.exists(project_relative):
        return os.path.abspath(project_relative)

    return pdf_path  # 返回原路径，让后续报错提示更友好

def _load_pdf(path: str):
    """加载 PDF 为 Document 列表"""
    import pdfplumber
    from langchain_core.documents import Document

    path = _resolve_pdf_path(path)
    if not os.path.exists(path):
        cwd = os.getcwd()
        print(f"❌ 文件不存在: {path}")
        print(f"   当前工作目录: {cwd}")
        print(f"   项目根目录:   {PROJECT_ROOT}")
        print(f"\n   提示: 可以使用以下方式之一指定路径:")
        print(f"     1) 绝对路径: /home/ubuntu/.../doc.pdf")
        print(f"     2) 相对项目根目录: docs/doc.pdf")
        print(f"     3) 相对当前目录: ./docs/doc.pdf")
        sys.exit(1)

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


def _init_components(db_path: str):
    """初始化存储和通道（所有组件使用统一的数据目录）"""
    from hybrid.document_store import DocumentStore
    from hybrid.channels.vector import VectorChannel
    from hybrid.channels.fts import FTSChannel
    from hybrid.channels.graph import GraphChannel
    from hybrid.config import DEFAULT_CHROMA_DIR, DEFAULT_GRAPH_PATH

    store = DocumentStore(db_path=db_path, chunk_size=2000, chunk_overlap=200)

    # 基于 db_path 推导向量/图谱路径，确保与 SQLite 在同一目录
    data_dir = os.path.dirname(db_path)
    vector_dir = os.path.join(data_dir, "chroma_db_hybrid")
    graph_path = os.path.join(data_dir, "knowledge_graph.pkl")

    vector = VectorChannel(persist_dir=vector_dir)
    fts = FTSChannel(db_path=db_path)
    graph = GraphChannel(path=graph_path)
    return store, vector, fts, graph


def _reset_data(db_path: str):
    """重置所有数据（SQLite + ChromaDB + Graph）"""
    import shutil

    print("[准备] 重置数据...")

    data_dir = os.path.dirname(db_path)
    vector_dir = os.path.join(data_dir, "chroma_db_hybrid")
    graph_path = os.path.join(data_dir, "knowledge_graph.pkl")

    # 删除 SQLite
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"  → 删除数据库: {db_path}")

    # 清理 ChromaDB
    if os.path.exists(vector_dir):
        shutil.rmtree(vector_dir)
        print(f"  → 清空 ChromaDB: {vector_dir}")

    # 清理 Graph
    if os.path.exists(graph_path):
        os.remove(graph_path)
        print(f"  → 清空 Graph: {graph_path}")

    print("  → 重置完成")


def cmd_index(args):
    """索引命令"""
    from hybrid.registry import Registry
    import time

    pdf_path = args.pdf
    db_path = args.db or DEFAULT_DB
    strategies = [s.strip() for s in args.strategies.split(",")]

    print("=" * 60)
    print("HybridRAG 文档索引")
    print("=" * 60)
    print(f"PDF: {pdf_path}")
    print(f"DB:  {db_path}")
    print(f"策略: {', '.join(strategies)}")

    # 加载环境变量中的模型
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    print(f"模型: {os.getenv('DEFAULT_MODEL', '未配置')}")

    # 重置
    if args.reset:
        _reset_data(db_path)

    # 初始化
    print("\n[1/4] 初始化组件...")
    store, vector, fts, graph = _init_components(db_path)
    print("  → DocumentStore, Vector, FTS, Graph 就绪")

    # 加载 PDF
    print(f"\n[2/4] 加载 PDF...")
    docs = _load_pdf(pdf_path)
    total_chars = sum(len(d.page_content) for d in docs)
    print(f"  → {len(docs)} 页, {total_chars} 字")

    # 分块
    print(f"\n[3/4] 分块保存...")
    chunks = store.split_and_save(docs)
    print(f"  → {len(chunks)} 个父块 (chunk_id: {chunks[0].chunk_id}~{chunks[-1].chunk_id})")

    # 策略索引
    print(f"\n[4/4] 构建策略索引...")
    for name in strategies:
        if name not in Registry.list_strategies():
            print(f"  ⚠️ 未知策略: {name}，跳过")
            continue

        strategy = Registry.get(name)
        print(f"\n  [{name}] {strategy.__class__.__name__}...")
        t0 = time.time()

        try:
            count = strategy.build(chunks, store, vector=vector, fts=fts, graph=graph)
            t1 = time.time()
            print(f"    → 索引: {count} 条, 耗时: {t1-t0:.1f}s")
            ch_names = "/".join(strategy.supported_channels)
            print(f"    → 通道: {ch_names}")
        except Exception as e:
            print(f"    ❌ 失败: {e}")

    # 汇总
    print("\n" + "=" * 60)
    print("索引完成")
    print("=" * 60)
    stats = store.get_stats()
    print(f"  chunks:       {stats['chunks']}")
    print(f"  derivatives:  {stats['derivatives']}")
    print(f"  indexed:      {stats['indexed']}")
    print(f"  ChromaDB:     {vector.count()}")
    print(f"  FTS5:         {fts.count()}")
    print(f"  Graph:        {graph.stats()}")

    if args.verbose:
        _show_detail_stats(store, db_path)


def cmd_status(args):
    """状态命令"""
    import sqlite3
    from hybrid.document_store import DocumentStore

    db_path = args.db or DEFAULT_DB

    print("=" * 60)
    print("HybridRAG 状态")
    print("=" * 60)

    if not os.path.exists(db_path):
        print(f"❌ 数据库不存在: {db_path}")
        return

    store = DocumentStore(db_path=db_path)
    stats = store.get_stats()
    print(f"  DB: {db_path}")
    print(f"  chunks:      {stats['chunks']}")
    print(f"  derivatives: {stats['derivatives']}")
    print(f"  indexed:     {stats['indexed']}")

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row

        print("\n  各策略派生:")
        rows = conn.execute("""
            SELECT strategy, COUNT(*) as cnt, COUNT(DISTINCT chunk_id) as chunk_count
            FROM chunk_derivatives GROUP BY strategy
        """).fetchall()
        for r in rows:
            print(f"    {r['strategy']:15s}: {r['cnt']:3d} 条, {r['chunk_count']:2d} 个 chunk")

        print("\n  各策略索引状态:")
        rows = conn.execute("""
            SELECT strategy, channel, COUNT(*) as cnt
            FROM index_status WHERE indexed = 1
            GROUP BY strategy, channel
            ORDER BY strategy, channel
        """).fetchall()
        for r in rows:
            print(f"    {r['strategy']:15s} / {r['channel']:7s}: {r['cnt']:3d} 条")


def cmd_search(args):
    """检索命令"""
    from hybrid.document_store import DocumentStore
    from hybrid.registry import Registry

    query = args.query
    db_path = args.db or DEFAULT_DB
    k = args.k
    strategy_name = args.strategy

    print("=" * 60)
    print(f"检索: '{query}'")
    print("=" * 60)

    store, vector, fts, graph = _init_components(db_path)

    if strategy_name:
        if strategy_name not in Registry.list_strategies():
            print(f"❌ 未知策略: {strategy_name}")
            return
        strategy = Registry.get(strategy_name)
        print(f"\n[{strategy_name}]...")
        results = strategy.search(query, store, vector=vector, fts=fts, graph=graph, k=k)
        _print_results(results, k)
    else:
        for name in Registry.list_strategies():
            strategy = Registry.get(name)
            print(f"\n[{name}]...")
            results = strategy.search(query, store, vector=vector, fts=fts, graph=graph, k=k)
            _print_results(results, k)


def _print_results(results, k):
    """打印检索结果"""
    if not results:
        print("  无结果")
        return

    print(f"  命中 {len(results)} 条 (top {k}):")
    for i, r in enumerate(results[:k]):
        content = r.get("content", "")[:80].replace("\n", " ")
        score = r.get("score", 0)
        channel = r.get("channel", "?")
        print(f"  [{i+1}] score={score:.4f}, channel={channel}")
        print(f"      {content}...")


def _show_detail_stats(store, db_path):
    """显示详细统计"""
    import sqlite3
    from hybrid.document_store import parse_derivative_id

    print("\n  详细统计:")
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row

        for strategy in ["summary", "hypothetical"]:
            rows = conn.execute("""
                SELECT derivative_id, content FROM chunk_derivatives
                WHERE strategy = ? ORDER BY derivative_id LIMIT 2
            """, (strategy,)).fetchall()
            if rows:
                print(f"\n  [{strategy}] 抽样:")
                for r in rows:
                    tc, cid, seq = parse_derivative_id(r["derivative_id"])
                    preview = r["content"][:60].replace("\n", " ")
                    print(f"    type={tc}, chunk={cid}, seq={seq}: {preview}...")


def main():
    parser = argparse.ArgumentParser(
        prog="hybrid.cli",
        description="HybridRAG 文档索引与检索 CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s index doc.pdf                    # 使用全部策略索引
  %(prog)s index doc.pdf --strategies standard,summary
  %(prog)s index doc.pdf --reset            # 重置后索引
  %(prog)s status                           # 查看状态
  %(prog)s search "自指"                    # 检索测试
        """
    )
    subparsers = parser.add_subparsers(dest="command", help="命令")

    # index 命令
    index_parser = subparsers.add_parser("index", help="索引 PDF 文档")
    index_parser.add_argument("pdf", help="PDF 文件路径")
    index_parser.add_argument("--db", default=DEFAULT_DB, help=f"数据库路径 (默认: {DEFAULT_DB})")
    index_parser.add_argument("--strategies", default=DEFAULT_STRATEGIES,
                              help=f"策略列表，逗号分隔 (默认: {DEFAULT_STRATEGIES})")
    index_parser.add_argument("--reset", action="store_true", help="重置后重新索引")
    index_parser.add_argument("-v", "--verbose", action="store_true", help="显示详细输出")
    index_parser.set_defaults(func=cmd_index)

    # status 命令
    status_parser = subparsers.add_parser("status", help="查看索引状态")
    status_parser.add_argument("--db", default=DEFAULT_DB, help=f"数据库路径 (默认: {DEFAULT_DB})")
    status_parser.set_defaults(func=cmd_status)

    # search 命令
    search_parser = subparsers.add_parser("search", help="检索测试")
    search_parser.add_argument("query", help="查询字符串")
    search_parser.add_argument("--db", default=DEFAULT_DB, help=f"数据库路径 (默认: {DEFAULT_DB})")
    search_parser.add_argument("-k", type=int, default=3, help="返回结果数 (默认: 3)")
    search_parser.add_argument("--strategy", help="指定策略检索 (默认: 全部)")
    search_parser.set_defaults(func=cmd_search)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
