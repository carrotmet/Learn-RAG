"""父子索引策略（Parent-Child）

父块 = 原始 chunk（2000字）
子块 = 父块内部二次切分（300字）

索引子块，检索子块，返回父块全文。
适用于细节定位检索。
"""

from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from hybrid.strategies.base import IndexStrategy
from hybrid.document_store import make_derivative_id


class ParentChildStrategy(IndexStrategy):
    """父子策略：父块二次切分为子块，索引子块，返回父块
    
    - build: 将每个父块切分为 300 字子块，保存到 chunk_derivatives，索引子块到 vector+fts
    - search: 检索子块，去重后返回对应父块全文
    
    Note:
        不写入 graph 通道。子块是原文的片段，关键词被切分后片段化，
        图检索需要完整的实体和关系上下文，子块不适合直接建图。
    """

    name = "parent_child"
    supported_channels = ["vector", "fts"]

    def __init__(self, child_size: int = 300, child_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size,
            chunk_overlap=child_overlap,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
        )

    def build(self, chunks, doc_store, vector=None, fts=None, graph=None) -> int:
        """构建父子索引
        
        将每个父块切分为子块，保存并索引。
        """
        total_children = 0
        for parent in chunks:
            # 切分子块
            child_docs = self.splitter.split_documents(
                [Document(page_content=parent.content)]
            )

            for i, child in enumerate(child_docs):
                deriv_id = make_derivative_id(1, parent.chunk_id, i + 1)  # 类型1=child, seq从1开始

                # 保存子块到 derivatives
                doc_store.save_derivative(
                    chunk_id=parent.chunk_id,
                    strategy="parent_child",
                    dtype="child",
                    derivative_id=deriv_id,
                    content=child.page_content,
                    metadata={"seq": i + 1, "source": parent.source, "page": parent.page}
                )

                # 构建索引项
                meta = {
                    "parent_chunk_id": parent.chunk_id,
                    "derivative_id": deriv_id,
                    "strategy": "parent_child",
                    "source": parent.source,
                    "page": parent.page,
                }
                items = [{"content": child.page_content, "metadata": meta}]

                # 根据 supported_channels 过滤通道，只写入 vector 和 fts
                channels = self._filter_channels(vector, fts, graph)
                for ch_name, ch in channels.items():
                    ch.add(items)

                # 标记索引状态（仅标记支持的通道）
                if "vector" in channels:
                    doc_store.mark_indexed(deriv_id, "parent_child", "vector")
                if "fts" in channels:
                    doc_store.mark_indexed(deriv_id, "parent_child", "fts")

                total_children += 1

        return total_children

    def search(self, query: str, doc_store,
               vector=None, fts=None, graph=None, k: int = 5) -> List[Dict]:
        """父子策略检索
        
        检索子块，按父块去重，返回父块全文。
        """
        results = []
        seen_parents = set()

        # 从 vector 检索子块
        if vector:
            for r in vector.search(query, k=k * 3):
                pid = r["parent_chunk_id"]
                if pid in seen_parents:
                    continue
                seen_parents.add(pid)

                parent = doc_store.get_chunk(pid)
                if parent:
                    results.append({
                        **r,
                        "strategy": "parent_child",
                        "chunk_id": pid,
                        "content": parent.content,
                        "context": parent.content,
                    })
                if len(results) >= k:
                    break

        # 从 fts 检索子块（补充）
        if fts and len(results) < k:
            for r in fts.search(query, k=k * 3):
                pid = r["parent_chunk_id"]
                if pid in seen_parents:
                    continue
                seen_parents.add(pid)

                parent = doc_store.get_chunk(pid)
                if parent:
                    results.append({
                        **r,
                        "strategy": "parent_child",
                        "chunk_id": pid,
                        "content": parent.content,
                        "context": parent.content,
                    })
                if len(results) >= k:
                    break

        # 从 graph 检索子块（补充）
        if graph and len(results) < k:
            for r in graph.search(query, k=k * 3):
                pid = r["parent_chunk_id"]
                if pid in seen_parents:
                    continue
                seen_parents.add(pid)

                parent = doc_store.get_chunk(pid)
                if parent:
                    results.append({
                        **r,
                        "strategy": "parent_child",
                        "chunk_id": pid,
                        "content": parent.content,
                        "context": parent.content,
                    })
                if len(results) >= k:
                    break

        return results[:k]
