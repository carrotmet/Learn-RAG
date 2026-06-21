"""HybridRAG 全局配置：统一路径常量

解决数据库路径分裂问题：
- 各通道类（Vector/FTS/Graph）和 DocumentStore 默认使用此处定义的路径
- 确保所有数据都落在 backend/data/ 下，而不是分散到 backend/src/

路径约定（基于 hybrid/__init__.py 的位置推导）：
- __file__ = .../backend/src/hybrid/config.py
- dirname×3 → .../backend/（项目 backend 目录）
"""

import os

# 计算 backend/ 目录（hybrid/config.py → hybrid/ → src/ → backend/）
BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 默认数据路径（全部放在 backend/data/ 下，避免分散）
DEFAULT_DB_PATH = os.path.join(BACKEND_ROOT, "data", "rag_data.db")
DEFAULT_CHROMA_DIR = os.path.join(BACKEND_ROOT, "data", "chroma_db_hybrid")
DEFAULT_GRAPH_PATH = os.path.join(BACKEND_ROOT, "data", "knowledge_graph.pkl")

# 导出，供各组件使用
__all__ = [
    "BACKEND_ROOT",
    "DEFAULT_DB_PATH",
    "DEFAULT_CHROMA_DIR",
    "DEFAULT_GRAPH_PATH",
]
