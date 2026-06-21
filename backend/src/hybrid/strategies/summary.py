"""摘要索引策略

为每个父块生成 LLM 摘要，索引摘要内容。
检索时返回【摘要】+【原文】的上下文。
"""

import os
from typing import List, Dict
from hybrid.strategies.base import IndexStrategy
from hybrid.document_store import make_derivative_id
from agent.llm import OpenRouterLLM


class SummaryStrategy(IndexStrategy):
    """摘要策略：为每个父块生成摘要，索引摘要
    
    - build: 调用 LLM 生成摘要，保存到 chunk_derivatives，索引到 vector+fts
    - search: 检索摘要，返回【摘要】+【原文】的上下文
    
    Note:
        不写入 graph 通道。摘要是对原文的压缩，关键词密度降低，
        图检索需要完整原文的实体关系，因此摘要不适合图谱索引。
    """

    name = "summary"
    supported_channels = ["vector", "fts"]

    def __init__(self, llm=None):
        self.llm = llm or OpenRouterLLM()

    def build(self, chunks, doc_store, vector=None, fts=None, graph=None) -> int:
        """构建摘要索引
        
        为每个父块生成摘要（如果尚未生成），然后索引到各通道。
        """
        count = 0
        for c in chunks:
            deriv_id = make_derivative_id(0, c.chunk_id, 0)  # 类型0=summary

            # 检查是否已存在摘要
            existing = doc_store.get_derivatives(c.chunk_id, "summary")
            if existing:
                summary = existing[0]["content"]
            else:
                # 调用 LLM 生成摘要
                prompt = f"""请为以下文本生成一段简洁的中文摘要，100字以内，保留核心观点：

---
{c.content}
---

摘要："""
                try:
                    summary = self.llm.generate(
                        prompt,
                        system="你是一个专业的文本摘要助手，请用中文生成精炼的摘要。",
                        temperature=0.3,
                    )
                    # 清理可能的额外输出
                    summary = summary.strip()
                    if summary.startswith("摘要"):
                        summary = summary.lstrip("摘要：").lstrip("摘要:").strip()
                except Exception as e:
                    print(f"[SummaryStrategy] LLM 生成摘要失败 (chunk_id={c.chunk_id}): {e}")
                    # fallback: 取前100字作为摘要
                    summary = c.content[:100] + "..." if len(c.content) > 100 else c.content

                # 保存到 derivatives 表
                doc_store.save_derivative(
                    chunk_id=c.chunk_id,
                    strategy="summary",
                    dtype="summary",
                    derivative_id=deriv_id,
                    content=summary,
                    metadata={"seq": 0, "source": c.source, "page": c.page}
                )

            # 构建索引项
            meta = {
                "parent_chunk_id": c.chunk_id,
                "derivative_id": deriv_id,
                "strategy": "summary",
                "source": c.source,
                "page": c.page,
            }
            items = [{"content": summary, "metadata": meta}]

            # 根据 supported_channels 过滤通道，只写入 vector 和 fts
            channels = self._filter_channels(vector, fts, graph)
            for ch_name, ch in channels.items():
                ch.add(items)

            # 标记索引状态（仅标记支持的通道）
            if "vector" in channels:
                doc_store.mark_indexed(deriv_id, "summary", "vector")
            if "fts" in channels:
                doc_store.mark_indexed(deriv_id, "summary", "fts")

            count += 1

        return count

    def search(self, query: str, doc_store,
               vector=None, fts=None, graph=None, k: int = 5) -> List[Dict]:
        """摘要策略检索
        
        检索摘要内容，返回结果时附加【摘要】+【原文】上下文。
        同一父块只返回一次（去重）。
        """
        results = []
        seen_parents = set()

        # 从 vector 检索
        if vector:
            for r in vector.search(query, k=k * 2):
                pid = r["parent_chunk_id"]
                if pid in seen_parents:
                    continue
                seen_parents.add(pid)

                parent = doc_store.get_chunk(pid)
                derivs = doc_store.get_derivatives(pid, "summary")
                summary = derivs[0]["content"] if derivs else ""

                context = f"【摘要】{summary}\n\n【原文】{parent.content if parent else ''}"
                results.append({
                    **r,
                    "strategy": "summary",
                    "chunk_id": pid,
                    "content": parent.content if parent else "",
                    "context": context,
                })
                if len(results) >= k:
                    break

        # 从 fts 检索（补充）
        if fts and len(results) < k:
            for r in fts.search(query, k=k * 2):
                pid = r["parent_chunk_id"]
                if pid in seen_parents:
                    continue
                seen_parents.add(pid)

                parent = doc_store.get_chunk(pid)
                derivs = doc_store.get_derivatives(pid, "summary")
                summary = derivs[0]["content"] if derivs else ""

                context = f"【摘要】{summary}\n\n【原文】{parent.content if parent else ''}"
                results.append({
                    **r,
                    "strategy": "summary",
                    "chunk_id": pid,
                    "content": parent.content if parent else "",
                    "context": context,
                })
                if len(results) >= k:
                    break

        # 从 graph 检索（补充）
        if graph and len(results) < k:
            for r in graph.search(query, k=k * 2):
                pid = r["parent_chunk_id"]
                if pid in seen_parents:
                    continue
                seen_parents.add(pid)

                parent = doc_store.get_chunk(pid)
                derivs = doc_store.get_derivatives(pid, "summary")
                summary = derivs[0]["content"] if derivs else ""

                context = f"【摘要】{summary}\n\n【原文】{parent.content if parent else ''}"
                results.append({
                    **r,
                    "strategy": "summary",
                    "chunk_id": pid,
                    "content": parent.content if parent else "",
                    "context": context,
                })
                if len(results) >= k:
                    break

        return results[:k]
