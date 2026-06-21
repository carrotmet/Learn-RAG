"""策略基类定义

所有索引策略必须继承 IndexStrategy，实现 build 和 search 方法。

策略与通道的映射关系（设计意图）：
- standard:     vector + fts + graph  (原文，全通道)
- summary:      vector + fts           (摘要，语义+关键词)
- parent_child: vector + fts           (子块，语义+关键词)
- hypothetical: vector                 (假设问题，仅语义)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, ClassVar


class IndexStrategy(ABC):
    """索引策略抽象基类
    
    设计原则：
    - build: 接收父块列表，构建索引（只写入 supported_channels 声明的通道）
    - search: 接收查询，从各通道检索，返回结果列表
    """

    # 子类覆盖：声明本策略支持写入哪些通道
    supported_channels: ClassVar[List[str]] = ["vector", "fts", "graph"]

    @property
    @abstractmethod
    def name(self) -> str:
        """策略名称（唯一标识）"""
        pass

    def _filter_channels(self, vector=None, fts=None, graph=None) -> Dict:
        """根据 supported_channels 过滤通道，防止误写入"""
        channels = {}
        if "vector" in self.supported_channels and vector is not None:
            channels["vector"] = vector
        if "fts" in self.supported_channels and fts is not None:
            channels["fts"] = fts
        if "graph" in self.supported_channels and graph is not None:
            channels["graph"] = graph
        return channels

    @abstractmethod
    def build(self, chunks: List, doc_store,
              vector=None, fts=None, graph=None) -> int:
        """构建索引
        
        Args:
            chunks: Chunk 列表（父块）
            doc_store: DocumentStore 实例
            vector: VectorChannel 实例（可选）
            fts: FTSChannel 实例（可选）
            graph: GraphChannel 实例（可选）
        
        Returns:
            索引的文档/派生数量
        
        Note:
            实现时通过 self._filter_channels(vector, fts, graph) 获取允许写入的通道。
            即使调用方传入了 graph，若本策略不支持 graph，也不会写入。
        """
        pass

    @abstractmethod
    def search(self, query: str, doc_store,
               vector=None, fts=None, graph=None,
               k: int = 5) -> List[Dict]:
        """检索
        
        Args:
            query: 查询字符串
            doc_store: DocumentStore 实例
            vector: VectorChannel 实例（可选）
            fts: FTSChannel 实例（可选）
            graph: GraphChannel 实例（可选）
            k: 返回结果数量
        
        Returns:
            检索结果列表，每条包含:
            - content: 内容文本
            - score: 相似度分数
            - parent_chunk_id: 父块 ID
            - derivative_id: 派生 ID
            - channel: 来源通道
            - context: 可选的上下文（如摘要策略的【摘要】+【原文】）
        """
        pass
