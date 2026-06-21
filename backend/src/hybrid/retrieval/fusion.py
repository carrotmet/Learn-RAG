"""融合层：RRF（Reciprocal Rank Fusion）多路召回结果融合

核心设计：
- 接收多路召回结果（按通道聚合）
- 使用 RRF 公式融合排名：score = sum(1 / (k + rank))
- 支持按通道权重加权
- 去重：同一 parent_chunk_id 只保留最高分
"""

from collections import defaultdict
from typing import List, Dict


class Fusion:
    """RRF 融合器
    
    RRF 公式：score = sum(weight[channel] * (1 / (k + rank)))
    
    其中 k=60 是经验值，防止排名靠前项的分数差异过大。
    """

    def __init__(self, weights: Dict[str, float] = None, k: int = 60):
        """
        Args:
            weights: 通道权重，如 {"vector": 0.5, "fts": 0.3, "graph": 0.2}
            k: RRF 常数，默认 60
        """
        self.weights = weights or {"vector": 0.5, "fts": 0.3, "graph": 0.2}
        self.k = k

    def rrf(self, result_dict: Dict[str, List[Dict]], top_k: int = 5) -> List[Dict]:
        """RRF 融合多路召回结果
        
        Args:
            result_dict: 多路召回结果，格式:
                {
                    "vector": [{"parent_chunk_id": 1000001, "score": 0.8, ...}, ...],
                    "fts": [...],
                    "graph": [...]
                }
            top_k: 返回融合后的 top_k 结果
        
        Returns:
            融合排序后的结果列表，每条包含:
            - parent_chunk_id: 父块 ID
            - fusion_score: RRF 融合分数
            - hit_channels: 命中通道列表
            - channel_scores: 各通道原始分数
            - 原始结果字段（content, score, channel 等）
        """
        # 记录每个 parent_chunk_id 在各通道的排名和分数
        channel_ranks = defaultdict(dict)  # pid -> {channel: rank}
        channel_scores = defaultdict(dict)  # pid -> {channel: score}
        info = {}  # pid -> 最佳结果（用于提取 content 等）
        hit_channels = defaultdict(list)  # pid -> [channel_name, ...]

        for channel_name, results in result_dict.items():
            w = self.weights.get(channel_name, 0)
            if w == 0 or not results:
                continue

            for rank, r in enumerate(results):
                pid = r.get("parent_chunk_id")
                if pid is None:
                    continue

                channel_ranks[pid][channel_name] = rank
                channel_scores[pid][channel_name] = r.get("score", 0)
                hit_channels[pid].append(channel_name)

                # 保留最完整的信息（优先保留有 context 的）
                if pid not in info or r.get("context"):
                    info[pid] = r

        # 计算 RRF 分数
        rrf_scores = {}
        for pid, ranks in channel_ranks.items():
            score = 0.0
            for ch, rank in ranks.items():
                w = self.weights.get(ch, 0)
                score += w * (1.0 / (self.k + rank + 1))
            rrf_scores[pid] = score

        # 按 RRF 分数排序
        sorted_pids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # 组装结果
        output = []
        for pid, score in sorted_pids[:top_k]:
            base = info.get(pid, {})
            result = {
                "parent_chunk_id": pid,
                "fusion_score": round(score, 6),
                "hit_channels": hit_channels.get(pid, []),
                "channel_scores": channel_scores.get(pid, {}),
                "content": base.get("content", ""),
                "context": base.get("context", base.get("content", "")),
                "channel": base.get("channel", "unknown"),
                "score": base.get("score", 0),
            }
            output.append(result)

        return output

    @staticmethod
    def deduplicate_by_parent(results: List[Dict]) -> List[Dict]:
        """按 parent_chunk_id 去重，保留最高分
        
        Args:
            results: 检索结果列表
        
        Returns:
            去重后的结果列表
        """
        best = {}
        for r in results:
            pid = r.get("parent_chunk_id")
            if pid is None:
                continue
            if pid not in best or r.get("score", 0) > best[pid].get("score", 0):
                best[pid] = r
        return list(best.values())
