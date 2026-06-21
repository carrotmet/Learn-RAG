"""向量通道：基于 ChromaDB + OpenRouter 嵌入"""

import os
from typing import List, Dict
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from hybrid.config import DEFAULT_CHROMA_DIR


class VectorChannel:
    """向量检索通道
    
    复用第一阶段 OpenRouter 嵌入模型，支持持久化存储。
    """

    def __init__(self, persist_dir: str = None, collection_name: str = "hybrid_docs"):
        if persist_dir is None:
            persist_dir = DEFAULT_CHROMA_DIR
        
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        
        # 动态导入 embedding（复用第一阶段的逻辑）
        self.embedding = self._load_embedding()
        self.db = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embedding,
            collection_name=collection_name,
        )

    def _load_embedding(self):
        """加载嵌入模型（复用第一阶段逻辑）"""
        # 尝试使用 OpenRouter 在线嵌入
        try:
            from agent.vector_store import OpenRouterEmbeddings
            return OpenRouterEmbeddings()
        except Exception:
            pass
        
        # fallback: 使用 FakeEmbeddings（测试用）
        try:
            from agent.vector_store import FakeEmbeddings
            print("[VectorChannel] 使用 FakeEmbeddings（测试模式）")
            return FakeEmbeddings(dim=2048)
        except Exception:
            pass
        
        # 终极 fallback
        from langchain_community.embeddings import FakeEmbeddings
        print("[VectorChannel] 使用默认 FakeEmbeddings")
        return FakeEmbeddings(size=2048)

    def add(self, items: List[Dict]) -> List[str]:
        """添加文档到向量库
        
        Args:
            items: [{"content": str, "metadata": {parent_chunk_id, derivative_id, ...}}, ...]
        
        Returns:
            添加的文档 ID 列表
        """
        docs = [
            Document(page_content=i["content"], metadata=i["metadata"])
            for i in items
        ]
        return self.db.add_documents(docs)

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """向量相似度检索
        
        Returns:
            [{"content": str, "score": float, "parent_chunk_id": int, 
              "derivative_id": int, "channel": "vector"}, ...]
        """
        results = self.db.similarity_search_with_score(query, k=k)
        output = []
        for doc, score in results:
            # score 是距离（越小越近），转换为相似度（越大越近）
            similarity = max(0, 1 - score)
            output.append({
                "content": doc.page_content,
                "score": round(similarity, 4),
                "parent_chunk_id": doc.metadata.get("parent_chunk_id"),
                "derivative_id": doc.metadata.get("derivative_id"),
                "channel": "vector",
                "metadata": doc.metadata,
            })
        return output

    def delete_collection(self):
        """删除整个集合（调试用）"""
        self.db.delete_collection()

    def count(self) -> int:
        """集合中文档数量"""
        return self.db._collection.count()
