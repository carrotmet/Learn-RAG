import os
from typing import List
from langchain_core.documents import Document

# 加载 .env 文件（如果存在）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class FakeEmbeddings:
    """本地假嵌入模型，用于无网络环境测试"""
    def __init__(self, dim: int = 2048):
        self.dim = dim
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        import hashlib
        return [
            [float(int(hashlib.md5((t + str(i)).encode()).hexdigest(), 16) % 10000) / 10000.0 for i in range(self.dim)]
            for t in texts
        ]
    
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class OpenRouterEmbeddings:
    """直接调用 OpenRouter 嵌入 API，绕过 langchain-openai 兼容性问题"""
    def __init__(self, model: str, api_key: str, api_base: str = "https://openrouter.ai/api/v1"):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        import requests
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })
    
    def _embed(self, texts: List[str]) -> List[List[float]]:
        import requests
        url = f"{self.api_base}/embeddings"
        response = self.session.post(url, json={
            "model": self.model,
            "input": texts,
        })
        response.raise_for_status()
        data = response.json()
        # OpenRouter 返回格式: {"data": [{"embedding": [...], "index": 0}]}
        embeddings = sorted(data["data"], key=lambda x: x.get("index", 0))
        return [item["embedding"] for item in embeddings]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # OpenRouter 有批量限制，分批处理
        batch_size = 100
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            results.extend(self._embed(batch))
        return results
    
    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]


class VectorStore:
    """简单的向量数据库封装：索引 + 检索（嵌入模型可配置）"""

    def __init__(self, persist_dir: str = None):
        self.persist_dir = persist_dir or os.getenv("CHROMA_DB_PATH", "./chroma_db")
        
        # 嵌入模型配置（优先级：环境变量 > 配置文件 > 默认）
        embedding_provider = os.getenv("EMBEDDING_PROVIDER", "fake").lower()
        embedding_model = os.getenv("EMBEDDING_MODEL", "fake")
        
        if embedding_provider == "openrouter":
            # OpenRouter 在线嵌入模型（使用自定义封装，绕过 langchain-openai 兼容性问题）
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("使用 OpenRouter 嵌入需要设置 OPENROUTER_API_KEY")
            self.embedding = OpenRouterEmbeddings(
                model=embedding_model,
                api_key=api_key,
                api_base="https://openrouter.ai/api/v1",
            )
            print(f"[VectorStore] 使用 OpenRouter 嵌入: {embedding_model}")
        elif embedding_provider == "huggingface":
            # HuggingFace 本地模型（需预下载到服务器）
            from langchain_huggingface import HuggingFaceEmbeddings
            self.embedding = HuggingFaceEmbeddings(model_name=embedding_model)
            print(f"[VectorStore] 使用 HuggingFace 嵌入: {embedding_model}")
        elif embedding_provider == "openai":
            # OpenAI 官方嵌入
            from langchain_openai import OpenAIEmbeddings
            self.embedding = OpenAIEmbeddings(model=embedding_model)
            print(f"[VectorStore] 使用 OpenAI 嵌入: {embedding_model}")
        else:
            # 回退到假嵌入（无网络环境测试）
            self.embedding = FakeEmbeddings()
            print(f"[VectorStore] 使用 FakeEmbeddings（本地哈希嵌入）")
        
        from langchain_community.vectorstores import Chroma
        self.db = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embedding,
        )
        
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )

    def index_file(self, file_path: str) -> None:
        """索引单个文件（支持 txt / pdf）"""
        if file_path.endswith(".pdf"):
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(file_path)
        else:
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(file_path, encoding="utf-8")

        docs = loader.load()
        chunks = self.splitter.split_documents(docs)
        self.db.add_documents(chunks)
        print(f"[索引完成] {file_path} -> {len(chunks)} 个片段")

    def index_directory(self, dir_path: str) -> None:
        """索引整个目录"""
        for root, _, files in os.walk(dir_path):
            for f in files:
                if f.endswith((".txt", ".pdf", ".md")):
                    self.index_file(os.path.join(root, f))

    def search(self, query: str, k: int = 4) -> list:
        """检索与问题最相关的文档片段"""
        return self.db.similarity_search(query, k=k)
