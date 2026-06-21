"""HybridRAG 核心模块

包含：
- DocumentStore: 统一 Chunk 存储层
- channels: 三通道实现（Vector/FTS/Graph）
- strategies: 索引策略（Standard/Summary/Parent-Child/HyDE）
- retrieval: 检索组件（Intent/Enrich/Multi-Recall/Fusion）
- registry: 策略注册中心
"""

from hybrid.document_store import DocumentStore, Chunk, make_derivative_id, parse_derivative_id
from hybrid.registry import Registry

__all__ = [
    "DocumentStore",
    "Chunk",
    "make_derivative_id",
    "parse_derivative_id",
    "Registry",
]
