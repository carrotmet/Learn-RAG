"""标准索引策略

直接将父块内容原样索引到各通道，不做任何派生处理。
"""

from typing import List, Dict
from hybrid.strategies.base import IndexStrategy


class StandardStrategy(IndexStrategy):
    """标准策略：直接索引父块原文
    
    - build: 将每个父块内容写入 vector/fts/graph
    - search: 从各通道检索，返回原始结果
    """

    name = "standard"
    supported_channels = ["vector", "fts", "graph"]

    def build(self, chunks, doc_store, vector=None, fts=None, graph=None) -> int:
        """构建标准索引
        
        Args:
            chunks: Chunk 列表
            doc_store: DocumentStore
            vector: VectorChannel（可选）
            fts: FTSChannel（可选）
            graph: GraphChannel（可选）
        
        Returns:
            索引的父块数量
        """
        items = []
        for c in chunks:
            meta = {
                "parent_chunk_id": c.chunk_id,
                "derivative_id": c.chunk_id,  # standard 无派生，derivative_id = chunk_id
                "strategy": "standard",
                "source": c.source,
                "page": c.page,
            }
            items.append({"content": c.content, "metadata": meta})

        # 根据 supported_channels 过滤通道，防止误写入
        channels = self._filter_channels(vector, fts, graph)
        for ch_name, ch in channels.items():
            ch.add(items)

        # 标记索引状态（仅标记支持的通道）
        for c in chunks:
            deriv_id = c.chunk_id
            if "vector" in channels:
                doc_store.mark_indexed(deriv_id, "standard", "vector")
            if "fts" in channels:
                doc_store.mark_indexed(deriv_id, "standard", "fts")
            if "graph" in channels:
                doc_store.mark_indexed(deriv_id, "standard", "graph")

        return len(chunks)

    def search(self, query: str, doc_store,
               vector=None, fts=None, graph=None, k: int = 5) -> List[Dict]:
        """标准策略检索
        
        从所有可用通道检索，直接返回结果（不做额外处理）。
        """
        results = []

        if vector:
            for r in vector.search(query, k=k):
                r["strategy"] = "standard"
                results.append(r)

        if fts:
            for r in fts.search(query, k=k):
                r["strategy"] = "standard"
                results.append(r)

        if graph:
            for r in graph.search(query, k=k):
                r["strategy"] = "standard"
                results.append(r)

        return results
