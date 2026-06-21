"""假设性问题索引策略（HyDE - Hypothetical Document Embeddings）

为每个父块生成假设性用户问题，索引问题文本。
检索时用用户查询匹配问题，返回对应父块原文。

适用于口语化查询、词汇鸿沟场景。
"""

import re
from typing import List, Dict
from hybrid.strategies.base import IndexStrategy
from hybrid.document_store import make_derivative_id
from agent.llm import OpenRouterLLM


class HypotheticalStrategy(IndexStrategy):
    """HyDE 策略：为每个父块生成假设性问题，索引问题，返回原文
    
    - build: 调用 LLM 为每个父块生成 n 个假设问题，保存并索引到 vector
    - search: 用查询匹配问题，返回对应父块原文
    
    Note:
        仅写入 vector 通道。假设问题是问句形式，与 FTS 关键词匹配的
        查询模式不同；也不适合图检索（问题不是知识实体）。
    """

    name = "hypothetical"
    supported_channels = ["vector"]

    def __init__(self, n: int = 3, llm=None):
        self.n = n
        self.llm = llm or OpenRouterLLM()

    def build(self, chunks, doc_store, vector=None, fts=None, graph=None) -> int:
        """构建 HyDE 索引
        
        为每个父块生成 n 个假设性问题，保存并索引。
        """
        total_questions = 0
        for c in chunks:
            # 检查是否已存在
            existing = doc_store.get_derivatives(c.chunk_id, "hypothetical")
            if existing and len(existing) >= self.n:
                # 已生成足够的问题，直接索引
                questions = [d["content"] for d in existing[:self.n]]
            else:
                # 调用 LLM 生成假设问题
                prompt = f"""基于以下文本，生成 {self.n} 个用户可能会问的中文问题。
每个问题独占一行，只输出问题，不要编号和额外说明。

---
{c.content}
---

问题："""
                try:
                    raw = self.llm.generate(
                        prompt,
                        system="你是一个问题生成助手，请基于给定文本生成用户可能提出的问题。",
                        temperature=0.7,
                    )
                    # 解析问题（每行一个）
                    questions = []
                    for line in raw.strip().split("\n"):
                        line = line.strip()
                        # 去掉编号前缀
                        if line and not line.startswith("问题"):
                            # 去掉 "1." "2)" 等前缀
                            line = re.sub(r"^\s*[\d一二三四五六七八九十]+[\.、)\]）]\s*", "", line)
                            if line:
                                questions.append(line)
                        elif line.startswith("问题"):
                            line = line.lstrip("问题：").lstrip("问题:").strip()
                            if line:
                                questions.append(line)

                    # 限制数量
                    questions = questions[:self.n]
                except Exception as e:
                    print(f"[HypotheticalStrategy] LLM 生成问题失败 (chunk_id={c.chunk_id}): {e}")
                    # fallback: 生成简单问题
                    questions = [f"这段文字的主要内容是什么？"] * self.n

                # 保存到 derivatives
                for i, q in enumerate(questions):
                    deriv_id = make_derivative_id(2, c.chunk_id, i + 1)  # 类型2=hyde
                    doc_store.save_derivative(
                        chunk_id=c.chunk_id,
                        strategy="hypothetical",
                        dtype="hyde",
                        derivative_id=deriv_id,
                        content=q,
                        metadata={"seq": i + 1, "source": c.source, "page": c.page}
                    )

            # 索引问题到 vector 通道
            channels = self._filter_channels(vector, fts, graph)
            for i, q in enumerate(questions):
                deriv_id = make_derivative_id(2, c.chunk_id, i + 1)
                meta = {
                    "parent_chunk_id": c.chunk_id,
                    "derivative_id": deriv_id,
                    "strategy": "hypothetical",
                    "source": c.source,
                    "page": c.page,
                }
                items = [{"content": q, "metadata": meta}]
                
                for ch_name, ch in channels.items():
                    ch.add(items)

                # 标记索引状态（仅标记支持的通道）
                if "vector" in channels:
                    doc_store.mark_indexed(deriv_id, "hypothetical", "vector")

                total_questions += 1

        return total_questions

    def search(self, query: str, doc_store,
               vector=None, fts=None, graph=None, k: int = 5) -> List[Dict]:
        """HyDE 策略检索
        
        用查询匹配假设问题，去重后返回对应父块原文。
        """
        results = []
        seen_parents = set()

        # 从 vector 检索
        if vector:
            for r in vector.search(query, k=k * 3):
                pid = r["parent_chunk_id"]
                if pid in seen_parents:
                    continue
                seen_parents.add(pid)

                parent = doc_store.get_chunk(pid)
                if parent:
                    # 附加匹配到的问题
                    derivs = doc_store.get_derivatives(pid, "hypothetical")
                    matched_questions = [d["content"] for d in derivs] if derivs else []

                    results.append({
                        **r,
                        "strategy": "hypothetical",
                        "chunk_id": pid,
                        "content": parent.content,
                        "context": parent.content,
                        "matched_questions": matched_questions,
                    })
                if len(results) >= k:
                    break

        # 从 fts 检索（补充）
        if fts and len(results) < k:
            for r in fts.search(query, k=k * 3):
                pid = r["parent_chunk_id"]
                if pid in seen_parents:
                    continue
                seen_parents.add(pid)

                parent = doc_store.get_chunk(pid)
                if parent:
                    derivs = doc_store.get_derivatives(pid, "hypothetical")
                    matched_questions = [d["content"] for d in derivs] if derivs else []

                    results.append({
                        **r,
                        "strategy": "hypothetical",
                        "chunk_id": pid,
                        "content": parent.content,
                        "context": parent.content,
                        "matched_questions": matched_questions,
                    })
                if len(results) >= k:
                    break

        # 从 graph 检索（补充）
        if graph and len(results) < k:
            for r in graph.search(query, k=k * 3):
                pid = r["parent_chunk_id"]
                if pid in seen_parents:
                    continue
                seen_parents.add(pid)

                parent = doc_store.get_chunk(pid)
                if parent:
                    derivs = doc_store.get_derivatives(pid, "hypothetical")
                    matched_questions = [d["content"] for d in derivs] if derivs else []

                    results.append({
                        **r,
                        "strategy": "hypothetical",
                        "chunk_id": pid,
                        "content": parent.content,
                        "context": parent.content,
                        "matched_questions": matched_questions,
                    })
                if len(results) >= k:
                    break

        return results[:k]
