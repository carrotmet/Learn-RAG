"""图通道：基于 NetworkX 的知识图谱

二期简化版：
- 节点：chunk 节点 + 关键词节点
- 边：chunk 与关键词的关联
- 检索：基于查询关键词的图遍历

三期可扩展为完整的实体关系抽取（使用 LLM）。
"""

import os
import pickle
import re
from collections import Counter
from typing import List, Dict, Set
import networkx as nx


class GraphChannel:
    """图谱检索通道
    
    简化实现：关键词共现图谱
    - chunk 节点：包含 chunk 内容摘要
    - keyword 节点：从 chunk 提取的关键词
    - 边：chunk 与其关键词的关联（带权重）
    """

    # 停用词（简化版，中文）
    STOPWORDS = set("""的 了 和 是 在 有 我 他 她 它 们 这 那 之 与 或 但 而 及 等 对 为 以 可 将 并 让 从 被 把 给 向 到 上 下 中 内 外 里 间 边 头 面 方 部 种 类 个 些 一 二 三 四 五 六 七 八 九 十 百 千 万 亿 零 几 多 很 非常 比较 最 更 太 特别 已经 正在 曾经 经常 总是 有时 偶尔 不 没 无 非 勿 别 不要 不能 不会 不可 不得 不必 不用 应该 应当 该 应 须 必须 一定 肯定 可能 也许 大概 或许 恐怕 难道 究竟 到底 什么 谁 哪 哪里 什么时候 为什么 怎么 怎样 如何 多少 几 怎样 什么 如何 谁 哪里 哪个 哪些 几时 多少""")

    def __init__(self, path: str = None):
        if path is None:
            base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            path = os.path.join(base, "data", "knowledge_graph.pkl")
        self.path = path
        self.G = self._load()

    def _load(self) -> nx.Graph:
        """加载图谱（如果不存在则创建空图）"""
        if os.path.exists(self.path):
            try:
                with open(self.path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"[GraphChannel] 加载图谱失败: {e}，创建新图")
        return nx.Graph()

    def save(self):
        """持久化图谱"""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump(self.G, f)

    # ── 建图 ────────────────────────────────────────────────

    def _extract_keywords(self, text: str, topk: int = 8) -> List[str]:
        """简单关键词抽取（基于词频，过滤停用词）"""
        # 提取中文词（2-6字）
        words = re.findall(r'[\u4e00-\u9fa5]{2,6}', text)
        # 过滤停用词和数字
        filtered = [w for w in words if w not in self.STOPWORDS and not w.isdigit()]
        # 统计词频，取 topk
        counter = Counter(filtered)
        return [w for w, _ in counter.most_common(topk)]

    def add(self, items: List[Dict]):
        """添加 chunk 到图谱
        
        Args:
            items: [{"content": str, "metadata": {parent_chunk_id, derivative_id}}, ...]
        """
        for i in items:
            meta = i["metadata"]
            chunk_id = meta.get("parent_chunk_id")
            content = i["content"]
            
            # chunk 节点
            node_id = f"c{chunk_id}"
            self.G.add_node(node_id, type="chunk", content=content[:200])
            
            # 提取关键词
            keywords = self._extract_keywords(content)
            for kw in keywords:
                self.G.add_node(kw, type="keyword")
                # 边权重 = 1（简单计数）
                if self.G.has_edge(node_id, kw):
                    self.G[node_id][kw]["weight"] += 1
                else:
                    self.G.add_edge(node_id, kw, weight=1)
        
        self.save()

    # ── 检索 ────────────────────────────────────────────────

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """基于关键词的图检索
        
        策略：
        1. 从 query 提取关键词
        2. 找到与这些关键词相连的 chunk 节点
        3. 按关联度排序返回
        """
        query_keywords = self._extract_keywords(query, topk=5)
        if not query_keywords:
            return []

        # 收集相关 chunk 节点
        chunk_scores: Dict[str, float] = {}
        for kw in query_keywords:
            if kw not in self.G:
                continue
            for neighbor in self.G.neighbors(kw):
                if self.G.nodes[neighbor].get("type") == "chunk":
                    weight = self.G[neighbor][kw].get("weight", 1)
                    chunk_scores[neighbor] = chunk_scores.get(neighbor, 0) + weight

        if not chunk_scores:
            return []

        # 归一化分数
        max_score = max(chunk_scores.values())
        sorted_chunks = sorted(
            chunk_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]

        results = []
        for node_id, score in sorted_chunks:
            chunk_id = int(node_id[1:])  # 去掉 "c" 前缀
            node_data = self.G.nodes[node_id]
            results.append({
                "content": node_data.get("content", ""),
                "score": round(score / max_score, 4) if max_score > 0 else 0,
                "parent_chunk_id": chunk_id,
                "derivative_id": chunk_id,  # 图检索直接关联父块
                "channel": "graph",
                "matched_keywords": [
                    kw for kw in query_keywords
                    if kw in self.G and node_id in self.G.neighbors(kw)
                ],
            })
        return results

    def clear(self):
        """清空图谱"""
        self.G = nx.Graph()
        self.save()

    def stats(self) -> Dict:
        """图谱统计"""
        nodes_by_type = {}
        for n, d in self.G.nodes(data=True):
            t = d.get("type", "unknown")
            nodes_by_type[t] = nodes_by_type.get(t, 0) + 1
        return {
            "nodes": self.G.number_of_nodes(),
            "edges": self.G.number_of_edges(),
            "nodes_by_type": nodes_by_type,
        }
