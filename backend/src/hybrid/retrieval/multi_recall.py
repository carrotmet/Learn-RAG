"""多路召回器：并行执行多种策略/通道的召回

核心设计：
- 根据意图识别结果，选择对应的策略组和通道权重
- 每个策略独立在各通道上执行检索
- 结果按通道聚合，返回给融合层
"""

from typing import List, Dict, Optional
from hybrid.registry import Registry
from hybrid.document_store import DocumentStore
from hybrid.channels.vector import VectorChannel
from hybrid.channels.fts import FTSChannel
from hybrid.channels.graph import GraphChannel


class MultiRecall:
    """多路召回器
    
    支持：
    - 多策略并行召回（如 standard + summary + parent_child）
    - 多通道并行召回（vector + fts + graph）
    - 按意图配置动态选择策略和权重
    """

    def __init__(self, doc_store: DocumentStore,
                 vector: VectorChannel = None,
                 fts: FTSChannel = None,
                 graph: GraphChannel = None):
        self.doc_store = doc_store
        self.vector = vector or VectorChannel()
        self.fts = fts or FTSChannel(db_path=doc_store.db_path)
        self.graph = graph or GraphChannel()

    def recall(self, query: str,
                 strategies: List[str] = None,
                 channels: List[str] = None,
                 k: int = 5) -> Dict[str, List[Dict]]:
        """执行多路召回
        
        Args:
            query: 查询字符串
            strategies: 策略列表（如 ["standard", "summary"]）
            channels: 通道列表（如 ["vector", "fts"]）
            k: 每路召回数量
        
        Returns:
            {channel_name: [result_dict, ...], ...}
            例如 {"vector": [...], "fts": [...], "graph": [...]}
        """
        strategies = strategies or ["standard"]
        channels = channels or ["vector", "fts", "graph"]

        # 构建通道实例映射
        channel_map = {}
        if "vector" in channels:
            channel_map["vector"] = self.vector
        if "fts" in channels:
            channel_map["fts"] = self.fts
        if "graph" in channels:
            channel_map["graph"] = self.graph

        # 按通道聚合结果
        all_results: Dict[str, List[Dict]] = {ch: [] for ch in channels}

        for strategy_name in strategies:
            if strategy_name not in Registry.list_strategies():
                continue
            strategy = Registry.get(strategy_name)

            # 构建 kwargs（只传入该策略支持的通道）
            kwargs = {}
            for ch_name, ch_obj in channel_map.items():
                if ch_name in strategy.supported_channels:
                    kwargs[ch_name] = ch_obj

            if not kwargs:
                continue

            # 执行策略检索
            try:
                results = strategy.search(query, self.doc_store, k=k, **kwargs)
                for r in results:
                    ch = r.get("channel", "unknown")
                    if ch in all_results:
                        all_results[ch].append(r)
            except Exception as e:
                # 单策略失败不影响其他策略
                print(f"[MultiRecall] 策略 {strategy_name} 召回失败: {e}")
                continue

        return all_results

    def recall_by_config(self, query: str, config: Dict) -> Dict[str, List[Dict]]:
        """根据意图识别配置执行召回
        
        Args:
            query: 查询字符串
            config: 意图识别的召回配置，格式:
                {
                    "strategies": ["standard", "summary"],
                    "weights": {"vector": 0.5, "fts": 0.3, "graph": 0.2},
                    "mode": "hybrid",
                    "k": 5
                }
        
        Returns:
            多路召回结果
        """
        strategies = config.get("strategies", ["standard"])
        mode = config.get("mode", "hybrid")
        k = config.get("k", 5)

        # 根据 mode 确定通道
        if mode == "vector":
            channels = ["vector"]
        elif mode == "fts":
            channels = ["fts"]
        elif mode == "graph":
            channels = ["graph"]
        else:  # hybrid
            channels = ["vector", "fts", "graph"]

        return self.recall(query, strategies=strategies, channels=channels, k=k)
