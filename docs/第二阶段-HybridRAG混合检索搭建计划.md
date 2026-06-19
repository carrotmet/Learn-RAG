# HybridRAG 第二阶段 — 混合检索搭建计划（v6.0 最终版）

> 基于第一阶段教学项目结构，升级混合检索与多路召回，引入两层意图识别（纯路由表 + 多选混合）与 Enrich（问题完善），至 Enrich 为止构成二期主体架构。

---

## 1. 与第一阶段的衔接

### 1.1 第一阶段项目结构（以 SKILL.md 为准）

```
backend/src/
├── agent/
│   ├── graph.py          # RAG 图定义（index → retrieve → generate）
│   ├── llm.py            # OpenRouter LLM 封装
│   ├── vector_store.py   # ChromaDB 向量存储（含 FakeEmbeddings/OpenRouterEmbeddings）
│   ├── app.py            # FastAPI（上传/采集/导出/测试集搭建/评估）
│   └── state.py          # RAGState
├── data_collection/
│   ├── sqlite_store.py   # SQLite 存储层（conversations/retrieval_logs/llm_calls...）
│   ├── uploader.py       # 离线上传解析器
│   ├── exporter.py       # 数据导出器
│   └── config.py         # 采集配置
├── testset/              # 测试集搭建
├── evaluation/           # RAGAS 评估
└── feedback/             # 可视化报告
```

### 1.2 第二阶段改动范围

**保留不变**：`agent/llm.py`、`data_collection/*`（除 sqlite_store 扩展表）、`testset/`、`evaluation/`、`feedback/`

**改造**：
- `agent/graph.py`：接入 Enrich → Intent → 多路召回 → Fusion → Generate 新链路
- `agent/vector_store.py`：不再自己分块，改为消费 Chunk
- `agent/app.py`：新增 `/api/hybrid/*` 接口
- `agent/state.py`：新增 enrich/intent/strategy/mode 字段
- `data_collection/sqlite_store.py`：扩展 chunks / chunk_derivatives / index_status 三张表
- `.env`：新增 `INTENT_MODEL`（意图识别专用模型，用户可配置）

**新增目录**：`backend/src/hybrid/`

### 1.3 环境变量配置

```bash
# backend/.env（二期新增）

# 意图识别模型（用户可修改，复用 OpenRouterLLM）
INTENT_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free
# 也可配置为：
# INTENT_MODEL=openai/gpt-4o-mini
# INTENT_MODEL=meta-llama/llama-3.3-70b-instruct:free

# 意图路由表配置路径（可选，默认内置）
INTENT_CONFIG=config/intents.yaml

# 一阶段已有配置保留
DEFAULT_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free
FALLBACK_MODELS=meta-llama/llama-3.3-70b-instruct:free,openai/gpt-oss-20b:free
EMBEDDING_PROVIDER=openrouter
EMBEDDING_MODEL=nvidia/llama-nemotron-embed-vl-1b-v2:free
CHROMA_DB_PATH=./chroma_db
```

---

## 2. 数据层：纯数字 ID 设计

### 2.1 设计原则

- **唯一性**：单调自增，不依赖哈希，不依赖字符串拼接
- **检索便捷性**：ID 本身可解析，通过数学运算或固定位数即可提取类型和关联信息
- **无安全性要求**：不加密，不混淆

### 2.2 chunk_id（父块）

```sql
chunk_id BIGINT PRIMARY KEY AUTOINCREMENT
-- 从 1000001 开始自增
-- 示例: 1000001, 1000002, 1000003, ...
```

### 2.3 derivative_id（派生内容）

固定 14 位纯数字，格式：`{类型位(1位)}{chunk_id(10位)}{序号(3位)}`

| 类型 | 编码 | derivative_id 示例 | 说明 |
|---|---|---|---|
| Summary | 0 | `01000001000` | 类型0 + chunk=1000001 + seq=000 |
| ChildChunk | 1 | `11000001001` | 类型1 + chunk=1000001 + seq=001 |
| HyDEQuestion | 2 | `21000001001` | 类型2 + chunk=1000001 + seq=001 |
| Custom | 3 | `31000001001` | 类型3 + chunk=1000001 + seq=001 |

**解析规则**（Python）：
```python
def parse_derivative_id(did: int) -> tuple:
    """返回 (type_code, chunk_id, seq)"""
    s = str(did).zfill(14)
    return int(s[0]), int(s[1:11]), int(s[11:14])

def make_derivative_id(type_code: int, chunk_id: int, seq: int = 0) -> int:
    return int(f"{type_code}{chunk_id:010d}{seq:03d}")
```

**关联规则**：
- `chunk_derivatives.chunk_id` 外键 → `chunks.chunk_id`
- 检索时，子块命中后从 metadata 读 `parent_chunk_id`（= chunk_id），直接去 chunks 表取父块内容
- 无需 JOIN chunk_derivatives 即可找到父块

### 2.4 表结构

```sql
-- 已有表保留：conversations, retrieval_logs, llm_calls, user_feedback, raw_data, processed_data, testset_versions

-- 新增：chunks 主表
CREATE TABLE chunks (
    chunk_id     BIGINT PRIMARY KEY AUTOINCREMENT,
    content      TEXT NOT NULL,
    source       TEXT NOT NULL,
    page         INTEGER DEFAULT 0,
    chunk_index  INTEGER,
    created_at   TEXT DEFAULT CURRENT_TIMESTAMP
);

-- 新增：chunk_derivatives 派生表
CREATE TABLE chunk_derivatives (
    derivative_id   BIGINT PRIMARY KEY,     -- 14位纯数字，规则见 §2.3
    chunk_id        BIGINT NOT NULL,        -- 外键 → chunks.chunk_id
    strategy        TEXT NOT NULL,          -- "standard"/"summary"/"parent_child"/"hypothetical"
    derivative_type TEXT NOT NULL,          -- "summary"/"child"/"hyde"/"custom"
    content         TEXT NOT NULL,          -- 派生内容
    metadata        TEXT,                   -- JSON
    created_at      TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id)
);

-- 新增：index_status 索引状态表
CREATE TABLE index_status (
    derivative_id  BIGINT NOT NULL,
    strategy       TEXT NOT NULL,
    channel        TEXT NOT NULL,           -- "vector"/"fts"/"graph"
    indexed        BOOLEAN DEFAULT 0,
    channel_doc_id TEXT,
    PRIMARY KEY (derivative_id, strategy, channel)
);
```

---

## 3. DocumentStore

```python
# backend/src/hybrid/document_store.py
import sqlite3
from typing import List, Dict, Optional
from dataclasses import dataclass
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class Chunk:
    chunk_id: int      # BIGINT
    content: str
    source: str
    page: int = 0
    chunk_index: int = 0


class DocumentStore:
    """统一 Chunk 存储：一次分块（2000字），全策略复用"""

    def __init__(self, db_path: str = "data/rag_data.db", chunk_size: int = 2000):
        self.db_path = db_path
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=200
        )
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS chunks (...);
                CREATE TABLE IF NOT EXISTS chunk_derivatives (...);
                CREATE TABLE IF NOT EXISTS index_status (...);
            """)

    def split_and_save(self, docs: List[Document]) -> List[Chunk]:
        """文档 → 分块 → 存入 chunks 表"""
        raw = self.splitter.split_documents(docs)
        chunks = []
        with sqlite3.connect(self.db_path) as conn:
            for i, doc in enumerate(raw):
                c = Chunk(0, doc.page_content, 
                         doc.metadata.get("source", ""),
                         doc.metadata.get("page", 0), i)
                cur = conn.execute(
                    "INSERT INTO chunks (content, source, page, chunk_index) VALUES (?,?,?,?)",
                    (c.content, c.source, c.page, c.chunk_index)
                )
                c.chunk_id = cur.lastrowid  # 获取自增 ID
                chunks.append(c)
        return chunks

    def save_derivative(self, chunk_id: int, strategy: str, dtype: str,
                        derivative_id: int, content: str, metadata: dict = None):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO chunk_derivatives
                (derivative_id, chunk_id, strategy, derivative_type, content, metadata)
                VALUES (?,?,?,?,?,?)
            """, (derivative_id, chunk_id, strategy, dtype, content,
                  json.dumps(metadata) if metadata else None))

    def get_chunk(self, chunk_id: int) -> Optional[Chunk]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone()
            return Chunk(**dict(row)) if row else None

    def get_derivatives(self, chunk_id: int, strategy: str) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM chunk_derivatives WHERE chunk_id=? AND strategy=?
            """, (chunk_id, strategy)).fetchall()
            return [dict(r) for r in rows]
```

---

## 4. 三通道实现

### 4.1 向量通道（ChromaDB）

```python
# backend/src/hybrid/channels/vector.py
from typing import List, Dict
from langchain_core.documents import Document

class VectorChannel:
    def __init__(self, persist_dir: str = "./chroma_db"):
        # embedding 初始化同第一阶段...
        from langchain_community.vectorstores import Chroma
        self.db = Chroma(persist_directory=persist_dir, embedding_function=self.embedding)

    def add(self, items: List[Dict]) -> List[str]:
        """items: [{"content": str, "metadata": {parent_chunk_id, derivative_id}}, ...]"""
        docs = [Document(page_content=i["content"], metadata=i["metadata"]) for i in items]
        return self.db.add_documents(docs)

    def search(self, query: str, k: int = 5) -> List[Dict]:
        results = self.db.similarity_search_with_score(query, k=k)
        return [
            {
                "content": doc.page_content,
                "score": 1 - score,
                "parent_chunk_id": doc.metadata.get("parent_chunk_id"),
                "derivative_id": doc.metadata.get("derivative_id"),
                "channel": "vector",
            }
            for doc, score in results
        ]
```

### 4.2 全文通道（FTS5）

```python
# backend/src/hybrid/channels/fts.py
import sqlite3

class FTSChannel:
    def __init__(self, db_path: str = "data/rag_data.db"):
        self.db_path = db_path
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS doc_fts USING fts5(
                    content, derivative_id, parent_chunk_id,
                    tokenize='porter unicode61'
                )
            """)

    def add(self, items: List[Dict]):
        with sqlite3.connect(self.db_path) as conn:
            for i in items:
                conn.execute("""
                    INSERT INTO doc_fts (content, derivative_id, parent_chunk_id)
                    VALUES (?,?,?)
                """, (i["content"], i["metadata"]["derivative_id"], i["metadata"]["parent_chunk_id"]))

    def search(self, query: str, k: int = 5) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT content, derivative_id, parent_chunk_id, bm25(doc_fts) as bm25
                FROM doc_fts WHERE doc_fts MATCH ? ORDER BY bm25(doc_fts) ASC LIMIT ?
            """, (query, k)).fetchall()
            max_s = max(r["bm25"] for r in rows) if rows else 1
            return [
                {
                    "content": r["content"],
                    "score": max(0, 1 - r["bm25"] / max(max_s, 1)),
                    "parent_chunk_id": r["parent_chunk_id"],
                    "derivative_id": r["derivative_id"],
                    "channel": "fts",
                }
                for r in rows
            ]
```

### 4.3 图通道（NetworkX）

```python
# backend/src/hybrid/channels/graph.py
import networkx as nx, pickle, os

class GraphChannel:
    def __init__(self, path: str = "data/knowledge_graph.pkl"):
        self.path = path
        self.G = self._load()

    def _load(self):
        return pickle.load(open(self.path, "rb")) if os.path.exists(self.path) else nx.Graph()

    def save(self):
        pickle.dump(self.G, open(self.path, "wb"))

    def add_entities(self, chunk_id: int, content: str):
        """从 chunk 抽取实体关系，构建图谱"""
        node = f"c{chunk_id}"  # 图节点 ID
        self.G.add_node(node, type="chunk", content=content[:200])
        # 实体抽取 + 建边...
        self.save()

    def search(self, query: str, k: int = 5) -> List[Dict]:
        # 基于 query 实体 BFS...
        return [{"parent_chunk_id": ..., "score": ..., "channel": "graph"}]
```

---

## 5. 索引策略层

### 5.1 策略接口

```python
# backend/src/hybrid/strategies/base.py
from abc import ABC, abstractmethod
from typing import List, Dict

class IndexStrategy(ABC):
    @property
    @abstractmethod
    def name(self) -> str: pass

    @abstractmethod
    def build(self, chunks: List[Chunk], doc_store: DocumentStore,
              vector: VectorChannel = None, fts: FTSChannel = None, graph: GraphChannel = None) -> int:
        pass

    @abstractmethod
    def search(self, query: str, doc_store: DocumentStore,
               vector: VectorChannel = None, fts: FTSChannel = None, graph: GraphChannel = None,
               k: int = 5) -> List[Dict]:
        pass
```

### 5.2 四种策略实现

```python
# backend/src/hybrid/strategies/standard.py
class StandardStrategy(IndexStrategy):
    name = "standard"

    def build(self, chunks, doc_store, vector=None, fts=None, graph=None):
        items = []
        for c in chunks:
            meta = {"parent_chunk_id": c.chunk_id, "derivative_id": c.chunk_id, "strategy": "standard"}
            items.append({"content": c.content, "metadata": meta})
        if vector: vector.add(items)
        if fts: fts.add(items)
        if graph:
            for c in chunks: graph.add_entities(c.chunk_id, c.content)
        return len(chunks)

    def search(self, query, doc_store, vector=None, fts=None, graph=None, k=5):
        results = []
        if vector: results.extend(vector.search(query, k))
        if fts: results.extend(fts.search(query, k))
        return results


# backend/src/hybrid/strategies/summary.py
from agent.llm import OpenRouterLLM

class SummaryStrategy(IndexStrategy):
    name = "summary"

    def __init__(self, llm=None):
        self.llm = llm or OpenRouterLLM()

    def build(self, chunks, doc_store, vector=None, fts=None, graph=None):
        from hybrid.document_store import make_derivative_id
        for c in chunks:
            deriv_id = make_derivative_id(0, c.chunk_id, 0)  # 类型0=summary
            existing = doc_store.get_derivatives(c.chunk_id, "summary")
            if existing:
                summary = existing[0]["content"]
            else:
                prompt = f"为以下文本生成摘要（100字内）：\n\n{c.content}\n\n摘要："
                summary = self.llm.generate(prompt)
                doc_store.save_derivative(c.chunk_id, "summary", "summary", deriv_id, summary)
            meta = {"parent_chunk_id": c.chunk_id, "derivative_id": deriv_id, "strategy": "summary"}
            if vector: vector.add([{"content": summary, "metadata": meta}])
            if fts: fts.add([{"content": summary, "metadata": meta}])
        return len(chunks)

    def search(self, query, doc_store, vector=None, fts=None, graph=None, k=5):
        results = []
        if vector:
            for r in vector.search(query, k):
                parent = doc_store.get_chunk(r["parent_chunk_id"])
                deriv = doc_store.get_derivatives(r["parent_chunk_id"], "summary")
                summary = deriv[0]["content"] if deriv else ""
                results.append({**r, "context": f"【摘要】{summary}\n\n【原文】{parent.content}"})
        return results


# backend/src/hybrid/strategies/parent_child.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class ParentChildStrategy(IndexStrategy):
    """父块=原始chunk(2000字)，子块=内部二次切分(300字)，检索子块返回父块"""
    name = "parent_child"

    def __init__(self, child_size: int = 300):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=child_size, chunk_overlap=50)

    def build(self, chunks, doc_store, vector=None, fts=None, graph=None):
        from hybrid.document_store import make_derivative_id
        for parent in chunks:
            child_docs = self.splitter.split_documents([Document(page_content=parent.content)])
            for i, child in enumerate(child_docs):
                deriv_id = make_derivative_id(1, parent.chunk_id, i + 1)  # 类型1=child, seq从1开始
                doc_store.save_derivative(parent.chunk_id, "parent_child", "child", deriv_id, child.page_content, {"seq": i + 1})
                meta = {"parent_chunk_id": parent.chunk_id, "derivative_id": deriv_id, "strategy": "parent_child"}
                if vector: vector.add([{"content": child.page_content, "metadata": meta}])
                if fts: fts.add([{"content": child.page_content, "metadata": meta}])
        return sum(len(self.splitter.split_documents([Document(page_content=c.content)])) for c in chunks)

    def search(self, query, doc_store, vector=None, fts=None, graph=None, k=5):
        results, seen = [], set()
        if vector:
            for r in vector.search(query, k=k * 2):
                pid = r["parent_chunk_id"]
                if pid in seen:
                    continue
                seen.add(pid)
                parent = doc_store.get_chunk(pid)
                results.append({**r, "chunk_id": pid, "content": parent.content, "context": parent.content})
                if len(results) >= k:
                    break
        return results


# backend/src/hybrid/strategies/hypothetical.py
class HypotheticalStrategy(IndexStrategy):
    """HyDE：为每个chunk生成假设性问题，索引问题，返回原文"""
    name = "hypothetical"

    def __init__(self, n: int = 3, llm=None):
        self.n = n
        self.llm = llm or OpenRouterLLM()

    def build(self, chunks, doc_store, vector=None, fts=None, graph=None):
        from hybrid.document_store import make_derivative_id
        for c in chunks:
            prompt = f"基于以下文本，生成{self.n}个用户可能会问的问题：\n\n{c.content}\n\n问题："
            questions = self.llm.generate(prompt).strip().split("\n")[:self.n]
            for i, q in enumerate(questions):
                deriv_id = make_derivative_id(2, c.chunk_id, i + 1)  # 类型2=hyde
                doc_store.save_derivative(c.chunk_id, "hypothetical", "hyde", deriv_id, q, {"seq": i + 1})
                meta = {"parent_chunk_id": c.chunk_id, "derivative_id": deriv_id, "strategy": "hypothetical"}
                if vector: vector.add([{"content": q, "metadata": meta}])
                if fts: fts.add([{"content": q, "metadata": meta}])
        return len(chunks) * self.n

    def search(self, query, doc_store, vector=None, fts=None, graph=None, k=5):
        results, seen = [], set()
        if vector:
            for r in vector.search(query, k=k * 2):
                pid = r["parent_chunk_id"]
                if pid in seen:
                    continue
                seen.add(pid)
                parent = doc_store.get_chunk(pid)
                results.append({**r, "chunk_id": pid, "content": parent.content, "context": parent.content})
                if len(results) >= k:
                    break
        return results
```

### 5.3 策略注册

```python
# backend/src/hybrid/registry.py
class Registry:
    _strategies = {}
    _instances = {}

    @classmethod
    def register(cls, name, klass):
        cls._strategies[name] = klass

    @classmethod
    def get(cls, name):
        if name not in cls._instances:
            cls._instances[name] = cls._strategies[name]()
        return cls._instances[name]

# 初始化
from hybrid.strategies.standard import StandardStrategy
from hybrid.strategies.summary import SummaryStrategy
from hybrid.strategies.parent_child import ParentChildStrategy
from hybrid.strategies.hypothetical import HypotheticalStrategy

Registry.register("standard", StandardStrategy)
Registry.register("summary", SummaryStrategy)
Registry.register("parent_child", ParentChildStrategy)
Registry.register("hypothetical", HypotheticalStrategy)
```

---

## 6. 多路召回

### 6.1 多路召回设计

HybridRAG 的核心不仅是三通道并行，更在于**不同召回路径针对不同查询特征**：

| 召回路 | 技术 | 适用查询特征 | 召回内容 |
|---|---|---|---|
| 语义召回 | 向量检索 + 标准/摘要策略 | 概念理解、语义相似 | 语义相关 chunk |
| 关键词召回 | FTS5 + 标准策略 | 精准术语、人名、专有名词 | 关键词匹配 chunk |
| 关系召回 | 图检索 + 标准策略 | 实体关联、多跳推理 | 关联实体所在 chunk |
| 摘要召回 | 向量/FTS + 摘要策略 | 高层语义、概述性查询 | 摘要相关 chunk |
| 子块召回 | 向量/FTS + 父子策略 | 细节定位、精确匹配 | 子块命中 → 父块返回 |
| HyDE 召回 | 向量 + 假设性问题策略 | 口语化、词汇鸿沟 | 问题匹配 → 原文返回 |

**多路召回的执行方式**：
- 并行执行所有可用召回路
- 各路独立返回结果（含 parent_chunk_id 和 channel 标记）
- 融合层按 RRF 或加权融合

### 6.2 多路召回实现（支持多策略并行）

```python
# backend/src/hybrid/retrieval/multi_recall.py
from typing import List, Dict
from hybrid.registry import Registry
from hybrid.channels.vector import VectorChannel
from hybrid.channels.fts import FTSChannel
from hybrid.channels.graph import GraphChannel
from hybrid.document_store import DocumentStore

class MultiRecall:
    """多路召回器：并行执行多种召回策略，支持多策略混合"""

    def __init__(self, doc_store: DocumentStore):
        self.doc_store = doc_store
        self.vector = VectorChannel()
        self.fts = FTSChannel()
        self.graph = GraphChannel()

    def recall(self, query: str, strategies: List[str] = None, 
               mode: str = "hybrid", k: int = 5) -> Dict[str, List[Dict]]:
        """
        执行多路召回，支持多策略并行。
        
        strategies: 索引策略列表（如 ["standard", "summary"]）
        mode: 检索通道（决定用什么技术检索）
        """
        strategies = strategies or ["standard"]
        
        # 根据 mode 启用通道
        channels = {}
        if mode in ("vector", "hybrid"):
            channels["vector"] = self.vector
        if mode in ("fts", "hybrid"):
            channels["fts"] = self.fts
        if mode in ("graph", "hybrid"):
            channels["graph"] = self.graph
        
        # 多策略并行召回，结果按通道聚合
        all_results = {}
        for strategy_name in strategies:
            strategy_impl = Registry.get(strategy_name)
            for ch_name, ch in channels.items():
                kwargs = {ch_name: ch for ch_name in channels}
                items = strategy_impl.search(query, self.doc_store, k=k, **kwargs)
                if ch_name not in all_results:
                    all_results[ch_name] = []
                all_results[ch_name].extend(items)
        
        return all_results
```

---

## 7. 意图识别（纯路由表设计）

### 7.1 设计原则

- **意图识别只贴标签，不与下游耦合**：LLM 只输出 `path` + `confidence` + `reasoning`，不提取实体、不填充参数、不决定后续动作
- **两层架构**：第一层判定方向（`retrieve` / `skill` / `tool`），第二层判定子类型（`retrieve.keyword` / `retrieve.concept` 等）
- **多选混合**：LLM 返回 `intents` 数组，一个查询可同时命中多个子意图（如 `retrieve.keyword` + `retrieve.concept`）
- **可扩展**：三期引入 `skill.sql`、`tool.report` 等，只需在路由表注册新路径，无需改动识别逻辑
- **复用一阶段 LLM**：使用 `OpenRouterLLM`，通过环境变量 `INTENT_MODEL` 配置模型，用户可在 `.env` 修改

### 7.2 纯路由表（意图注册表）

```yaml
# config/intents.yaml — 意图纯路由表，只保留"是什么"和"属于哪类"
intents:
  # 第一层：retrieve（需要检索）
  retrieve.keyword:
    description: "基于关键词的精确检索，适合已知实体名、术语、编号"
    tags: [retrieve, search, exact-match]
    
  retrieve.concept:
    description: "概念定义检索，回答'是什么'类问题"
    tags: [retrieve, concept, definition]
    
  retrieve.relation:
    description: "关系图谱检索，查找实体间关联、区别、对比"
    tags: [retrieve, relation, graph]
    
  retrieve.detail:
    description: "细节定位检索，查找具体步骤、方法、精确事实"
    tags: [retrieve, detail, precision]
    
  # 第二层扩展预留（三期）
  skill.global:
    description: "全局技能调用，跨项目通用能力"
    tags: [skill, global, cross-project]
    
  skill.sql:
    description: "结构化数据查询，通过 SQL 检索"
    tags: [skill, sql, structured]
    
  tool.report:
    description: "报表生成工具"
    tags: [tool, report, output]
```

**倒排索引**（供大模型 function call 快速过滤）：
```yaml
index:
  by_tag:
    retrieve: [retrieve.keyword, retrieve.concept, retrieve.relation, retrieve.detail]
    skill: [skill.global, skill.sql]
    tool: [tool.report]
    
  by_keyword:
    "查": [retrieve.keyword, retrieve.concept]
    "找": [retrieve.keyword]
    "是什么": [retrieve.concept]
    "关系": [retrieve.relation]
    "区别": [retrieve.relation]
    "步骤": [retrieve.detail]
    "生成": [tool.report]
```

### 7.3 大模型交互方法

**复用一阶段 `OpenRouterLLM`**，通过 `INTENT_MODEL` 环境变量配置：

```python
# .env
INTENT_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free
# 用户可修改为其他模型，如：
# INTENT_MODEL=openai/gpt-4o-mini
```

**Prompt 设计**：
```text
你是意图识别助手。请从以下意图列表中选择最匹配的意图。

可用意图：
- retrieve.keyword: 基于关键词的精确检索，适合已知实体名 [tags: retrieve, search]
- retrieve.concept: 概念定义检索，回答'是什么'类问题 [tags: retrieve, concept]
- retrieve.relation: 关系图谱检索，查找实体间关联 [tags: retrieve, relation]
- retrieve.detail: 细节定位检索，查找具体步骤、方法 [tags: retrieve, detail]

如果意图不明确，请先调用 search_intents 工具缩小范围。
返回格式：intents 数组，每个元素包含 path, confidence, reasoning。
注意：你只需要识别意图，不需要提取任何实体或参数。子意图可多选。
```

**大模型返回格式**：
```json
{
  "intents": [
    {
      "path": "retrieve.concept",
      "confidence": "high",
      "reasoning": "用户询问概念定义"
    },
    {
      "path": "retrieve.keyword",
      "confidence": "medium",
      "reasoning": "问题中包含具体术语"
    }
  ],
  "primary_intent": "retrieve.concept"
}
```

### 7.4 实现代码

```python
# backend/src/hybrid/retrieval/intent.py
import os
import json
import yaml
from typing import List, Dict
from agent.llm import OpenRouterLLM

class IntentRouter:
    """
    意图识别器：纯路由表设计，只贴标签，不与下游耦合。
    
    复用一阶段 OpenRouterLLM，模型通过 INTENT_MODEL 环境变量配置。
    """

    def __init__(self, llm=None):
        # ★ 复用一阶段 LLM，但使用 INTENT_MODEL 指定的模型
        base_llm = llm or OpenRouterLLM()
        intent_model = os.getenv("INTENT_MODEL", base_llm.primary_model)
        self.llm = OpenRouterLLM(model_id=intent_model)
        
        # 加载意图路由表
        self.intents = self._load_intents()
        self.index = self._build_index()

    def _load_intents(self) -> Dict:
        """加载意图注册表（YAML 配置）"""
        config_path = os.getenv("INTENT_CONFIG", "config/intents.yaml")
        if os.path.exists(config_path):
            with open(config_path) as f:
                return yaml.safe_load(f).get("intents", {})
        # 默认内置意图
        return {
            "retrieve.keyword": {"description": "关键词精确检索", "tags": ["retrieve", "search"]},
            "retrieve.concept": {"description": "概念定义检索", "tags": ["retrieve", "concept"]},
            "retrieve.relation": {"description": "关系图谱检索", "tags": ["retrieve", "relation"]},
            "retrieve.detail": {"description": "细节定位检索", "tags": ["retrieve", "detail"]},
        }

    def _build_index(self) -> Dict:
        """构建倒排索引，用于 function call 快速过滤"""
        by_tag = {}
        by_keyword = {}
        for path, info in self.intents.items():
            for tag in info.get("tags", []):
                by_tag.setdefault(tag, []).append(path)
            # 从 description 提取关键词（简单分词）
            words = info.get("description", "").replace(",", "").replace("、", " ").split()
            for w in words:
                by_keyword.setdefault(w, []).append(path)
        return {"by_tag": by_tag, "by_keyword": by_keyword}

    def search_intents(self, tag: str = None, keyword: str = None, max_results: int = 10) -> List[Dict]:
        """基于倒排索引快速过滤候选意图（供大模型 function call 使用）"""
        results = set()
        if tag:
            results.update(self.index["by_tag"].get(tag, []))
        if keyword:
            results.update(self.index["by_keyword"].get(keyword, []))
        return [
            {"path": p, "description": self.intents[p]["description"]}
            for p in list(results)[:max_results]
        ]

    def recognize(self, query: str, history: list = None) -> Dict:
        """
        调用大模型识别意图。
        
        Returns: {
            "intents": [{"path": "retrieve.concept", "confidence": "high", "reasoning": "..."}, ...],
            "primary_intent": "retrieve.concept",
            "needs_retrieve": True,  # 第一层判定：是否需要检索
        }
        """
        # 构建 prompt
        intent_list = "\n".join(
            f"- {path}: {info['description']} [tags: {', '.join(info['tags'])}]"
            for path, info in self.intents.items()
        )
        
        prompt = f"""你是意图识别助手。请从以下意图列表中选择最匹配的意图。

可用意图：
{intent_list}

用户问题：{query}

请返回 JSON 格式：
{{
  "intents": [
    {{"path": "意图路径", "confidence": "high/medium/low", "reasoning": "选择理由"}}
  ],
  "primary_intent": "主意图路径"
}}

注意：
1. 子意图可多选（如同时命中 keyword 和 concept）
2. 只需要识别意图，不需要提取实体或参数
3. confidence 为 high 表示非常确定，medium 表示可能，low 表示不确定"""

        response = self.llm.generate(prompt, system="你是一个意图识别专家，只输出意图标签。")
        
        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            # LLM 返回非 JSON，fallback 到规则匹配
            result = self._rule_fallback(query)
        
        # 第一层判定：是否需要检索
        paths = [i["path"] for i in result.get("intents", [])]
        result["needs_retrieve"] = any(p.startswith("retrieve.") for p in paths)
        
        return result

    def _rule_fallback(self, query: str) -> Dict:
        """LLM 调用失败时的规则回退"""
        import re
        intents = []
        if re.search(r"(关系|关联|区别|对比)", query):
            intents.append({"path": "retrieve.relation", "confidence": "medium", "reasoning": "规则匹配"})
        if re.search(r"(步骤|流程|方法|具体|详细)", query):
            intents.append({"path": "retrieve.detail", "confidence": "medium", "reasoning": "规则匹配"})
        if re.search(r"(是什么|定义|解释)", query):
            intents.append({"path": "retrieve.concept", "confidence": "medium", "reasoning": "规则匹配"})
        if not intents:
            intents.append({"path": "retrieve.keyword", "confidence": "low", "reasoning": "默认回退"})
        return {"intents": intents, "primary_intent": intents[0]["path"]}

    @staticmethod
    def get_retrieve_config(intents: List[Dict]) -> Dict:
        """
        根据检索意图列表，生成召回配置（多意图混合）。
        
        多个检索子意图同时命中时，合并权重和策略。
        """
        # 子意图到配置的映射
        config_map = {
            "retrieve.keyword": {"strategies": ["standard"], "weights": {"vector": 0.2, "fts": 0.7, "graph": 0.1}},
            "retrieve.concept": {"strategies": ["summary", "standard"], "weights": {"vector": 0.7, "fts": 0.2, "graph": 0.1}},
            "retrieve.relation": {"strategies": ["standard"], "weights": {"vector": 0.3, "fts": 0.1, "graph": 0.6}},
            "retrieve.detail": {"strategies": ["parent_child", "standard"], "weights": {"vector": 0.5, "fts": 0.4, "graph": 0.1}},
        }
        
        # 合并多个意图的配置
        all_strategies = set()
        merged_weights = {"vector": 0, "fts": 0, "graph": 0}
        
        for intent in intents:
            path = intent["path"]
            if path not in config_map:
                continue
            cfg = config_map[path]
            all_strategies.update(cfg["strategies"])
            for k, v in cfg["weights"].items():
                merged_weights[k] = max(merged_weights[k], v)  # 取最大值合并
        
        # 归一化权重
        total = sum(merged_weights.values())
        if total > 0:
            merged_weights = {k: v / total for k, v in merged_weights.items()}
        
        return {
            "strategies": list(all_strategies),  # 多策略并行
            "weights": merged_weights,
            "mode": "hybrid",
        }
```

---

## 8. Enrich（问题完善）

### 8.1 设计定位

Enrich 是 HybridRAG 的**入口层**，负责：
1. **完整度判断**：用户问题是否完整？是否需要补充信息？
2. **问题改写**：将口语化/模糊的问题改写为结构化查询
3. **请求补充信息**：问题不完整时，主动向用户追问

**Enrich 是 HybridRAG 与 AgenticRAG 的桥梁**：一期 HybridRAG 实现基础 Enrich，二期 AgenticRAG 扩展为更复杂的对话状态管理和多轮推理。

### 8.2 完整度判断

```python
# backend/src/hybrid/retrieval/enrich.py
from agent.llm import OpenRouterLLM

class Enrich:
    def __init__(self, llm=None):
        self.llm = llm or OpenRouterLLM()

    def check_completeness(self, query: str, history: list = None) -> Dict:
        """
        判断问题是否完整。
        
        Returns: {
            "complete": bool,
            "reason": str,           # 判断理由
            "missing_info": list,    # 缺失的信息项
            "rewritten_query": str,  # 改写后的问题（如完整）
            "follow_up_question": str # 追问问题（如不完整）
        }
        """
        prompt = f"""请判断以下用户问题是否完整，是否需要补充信息才能准确回答。

用户问题：{query}

请按以下格式回答：
完整度：(完整/不完整)
理由：...
缺失信息：...
改写后的问题（如完整）：...
追问建议（如不完整）：...
"""
        response = self.llm.generate(prompt, system="你是一个问题分析助手。")
        # 解析 LLM 输出...
        return self._parse_response(response)
```

### 8.3 问题改写

```python
    def rewrite(self, query: str, history: list = None) -> str:
        """将口语化/模糊问题改写为结构化查询"""
        prompt = f"""请将以下用户问题改写为更清晰的检索查询，保留核心意图。

原始问题：{query}

改写要求：
- 保留关键实体和概念
- 去除口语化表达
- 补充隐含的上下文（如果有对话历史）
- 输出一条最简洁的检索查询

改写后："""
        return self.llm.generate(prompt, system="你是一个查询改写专家。")
```

### 8.4 Enrich 在链路中的位置

```
User Query
    │
    ▼
┌─────────────────┐
│   Enrich        │  ← 完整度判断
│  (问题完善)      │
└─────────────────┘
    │
    ├─ 不完整 → 返回追问 → User 补充 → 重新进入 Enrich
    │
    └─ 完整 → 改写查询 → Intent 识别 → 多路召回 → Fusion → Generate
```

---

## 9. 融合层

```python
# backend/src/hybrid/retrieval/fusion.py
from collections import defaultdict
from typing import List, Dict

class Fusion:
    def __init__(self, weights: dict = None):
        self.weights = weights or {"vector": 0.5, "fts": 0.3, "graph": 0.2}
        self.k = 60

    def rrf(self, result_dict: Dict[str, List[Dict]], top_k: int = 5) -> List[Dict]:
        """
        多路召回结果融合。
        
        result_dict: {"vector": [...], "fts": [...], "graph": [...]}
        """
        scores = defaultdict(float)
        info = {}
        channels = defaultdict(list)

        for channel_name, results in result_dict.items():
            w = self.weights.get(channel_name, 0)
            for rank, r in enumerate(results):
                pid = r["parent_chunk_id"]
                scores[pid] += w * (1.0 / (self.k + rank + 1))
                info[pid] = r
                channels[pid].append(channel_name)

        sorted_pids = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [
            {**info[pid], "fusion_score": score, "hit_channels": channels[pid]}
            for pid, score in sorted_pids[:top_k]
        ]
```

---

## 10. LangGraph 编排

### 10.1 状态定义

```python
# backend/src/agent/state.py（扩展）
from typing import TypedDict, Annotated, List, Dict
from langgraph.graph import add_messages

class RAGState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    
    # Enrich 层
    enrich_complete: bool           # 问题是否完整
    enrich_reason: str              # 完整度判断理由
    enrich_rewritten: str           # 改写后的问题
    enrich_follow_up: str           # 追问问题
    
    # Intent 层（纯路由表，两层架构，多选混合）
    intents: List[Dict]             # 识别的意图列表 [{path, confidence, reasoning}, ...]
    primary_intent: str             # 主意图路径
    needs_retrieve: bool            # 第一层判定：是否需要检索
    retrieve_strategies: List[str]  # 检索子策略列表（可多选混合）
    retrieve_weights: dict          # 合并后的通道权重
    retrieve_mode: str              # 检索通道模式
    
    # 召回层
    recall_results: dict            # 多路召回原始结果
    fused_results: list             # 融合后结果
    
    # 生成层
    answer: str
    retrieval_latency: int
```

### 10.2 图节点

```python
# backend/src/agent/graph.py（改造后）
from langgraph.graph import StateGraph, START, END
from hybrid.retrieval.enrich import Enrich
from hybrid.retrieval.intent import IntentRouter
from hybrid.retrieval.multi_recall import MultiRecall
from hybrid.retrieval.fusion import Fusion
from hybrid.document_store import DocumentStore

enricher = Enrich()
intent_router = IntentRouter()
recaller = MultiRecall(DocumentStore())

# 节点 1：Enrich
def enrich_node(state: RAGState):
    question = extract_last_question(state["messages"])
    result = enricher.check_completeness(question)
    return {
        "enrich_complete": result["complete"],
        "enrich_reason": result["reason"],
        "enrich_rewritten": result.get("rewritten_query", question),
        "enrich_follow_up": result.get("follow_up_question", ""),
        "question": question,
    }

# 节点 2：Intent 识别（纯路由表，两层判定）
def intent_node(state: RAGState):
    if not state["enrich_complete"]:
        return {"intents": [], "needs_retrieve": False}
    
    # 调用大模型识别意图（复用 OpenRouterLLM，模型可配置）
    result = intent_router.recognize(state["enrich_rewritten"])
    
    # 第一层判定：是否需要检索
    if not result.get("needs_retrieve"):
        return {
            "intents": result.get("intents", []),
            "primary_intent": result.get("primary_intent", ""),
            "needs_retrieve": False,
        }
    
    # 第二层：检索子策略配置（多意图混合）
    retrieve_intents = [i for i in result.get("intents", []) if i["path"].startswith("retrieve.")]
    config = IntentRouter.get_retrieve_config(retrieve_intents)
    
    return {
        "intents": result.get("intents", []),
        "primary_intent": result.get("primary_intent", ""),
        "needs_retrieve": True,
        "retrieve_strategies": config["strategies"],
        "retrieve_weights": config["weights"],
        "retrieve_mode": config["mode"],
    }

# 节点 3：多路召回（多策略并行）
def recall_node(state: RAGState):
    if not state.get("needs_retrieve"):
        return {"recall_results": {}}
    
    # 多策略并行召回
    all_results = {}
    for strategy_name in state["retrieve_strategies"]:
        results = recaller.recall(
            state["enrich_rewritten"],
            strategy=strategy_name,
            mode=state["retrieve_mode"],
            k=8,
        )
        # 合并各路结果
        for channel, items in results.items():
            if channel not in all_results:
                all_results[channel] = []
            all_results[channel].extend(items)
    
    return {"recall_results": all_results}

# 节点 4：融合
def fusion_node(state: RAGState):
    if not state.get("recall_results"):
        return {"fused_results": []}
    fused = Fusion(state["retrieve_weights"]).rrf(state["recall_results"], top_k=5)
    return {"fused_results": fused}

# 节点 5：生成
def generate_node(state: RAGState):
    if not state.get("enrich_complete"):
        # 问题不完整，返回追问
        return {"answer": state["enrich_follow_up"], "messages": [AIMessage(content=state["enrich_follow_up"])]}
    
    if not state.get("needs_retrieve"):
        # 不需要检索，直接调用 LLM 回答（如 skill/tool 意图）
        answer = OpenRouterLLM().generate(state["enrich_rewritten"])
        return {"answer": answer, "messages": [AIMessage(content=answer)]}
    
    # 正常生成...
    contexts = [r.get("context", r.get("content", "")) for r in state.get("fused_results", [])]
    prompt = f"基于以下文档回答问题：\n\n{'\n---\n'.join(contexts)}\n\n问题：{state['enrich_rewritten']}"
    answer = OpenRouterLLM().generate(prompt)
    return {"answer": answer, "messages": [AIMessage(content=answer)]}

# 构图
builder = StateGraph(RAGState)
builder.add_node("enrich", enrich_node)
builder.add_node("intent", intent_node)
builder.add_node("recall", recall_node)
builder.add_node("fusion", fusion_node)
builder.add_node("generate", generate_node)

builder.add_edge(START, "enrich")
builder.add_edge("enrich", "intent")
builder.add_edge("intent", "recall")
builder.add_edge("recall", "fusion")
builder.add_edge("fusion", "generate")
builder.add_edge("generate", END)

hybrid_graph = builder.compile(name="hybrid-rag")
```

---

## 11. API 接口

```python
# backend/src/agent/app.py（新增接口）

# 混合检索对话（流式）
@app.post("/runs/stream")
async def hybrid_stream(request: Request):
    body = await request.json()
    input_data = body.get("input", {})
    return {
        "input": {
            "messages": input_data.get("messages", []),
            # 可覆盖自动意图识别的配置
            "strategy": input_data.get("strategy"),      # 单策略覆盖
            "strategies": input_data.get("strategies"),  # 多策略覆盖
            "mode": input_data.get("mode"),              # "vector"/"fts"/"graph"/"hybrid"
            "weights": input_data.get("weights"),
        }
    }

# 获取意图识别结果（调试）
@app.post("/api/hybrid/intent")
async def detect_intent(data: dict = Body(...)):
    from hybrid.retrieval.intent import IntentRouter
    result = IntentRouter().recognize(data["query"])
    return {
        "intents": result.get("intents", []),
        "primary_intent": result.get("primary_intent"),
        "needs_retrieve": result.get("needs_retrieve"),
        "retrieve_config": IntentRouter.get_retrieve_config(result.get("intents", [])) if result.get("needs_retrieve") else None,
    }

# 问题改写（调试）
@app.post("/api/hybrid/enrich")
async def enrich_query(data: dict = Body(...)):
    from hybrid.retrieval.enrich import Enrich
    result = Enrich().check_completeness(data["query"])
    return result

# 多路召回调试
@app.post("/api/hybrid/recall")
async def debug_recall(data: dict = Body(...)):
    from hybrid.retrieval.multi_recall import MultiRecall
    from hybrid.document_store import DocumentStore
    results = MultiRecall(DocumentStore()).recall(
        data["query"], 
        strategies=data.get("strategies", ["standard"]), 
        mode=data.get("mode", "hybrid")
    )
    return {"recall_results": results}
```

---

## 12. 项目结构

```
backend/src/
├── agent/
│   ├── graph.py          # 改造：Enrich → Intent → Recall → Fusion → Generate
│   ├── llm.py            # 保留
│   ├── vector_store.py   # 改造：不再自己分块
│   ├── app.py            # 改造：新增 /api/hybrid/* 接口
│   └── state.py          # 改造：新增 enrich/intent/strategy 字段
│
├── data_collection/      # 保留
│   ├── sqlite_store.py   # 扩展：chunks / chunk_derivatives / index_status 表
│   ├── uploader.py
│   ├── exporter.py
│   └── config.py
│
├── testset/              # 保留
├── evaluation/           # 保留
├── feedback/             # 保留
│
└── hybrid/               # 新增
    ├── __init__.py
    ├── document_store.py       # chunks 主表 + 派生表读写 + 纯数字 ID 工具
    ├── registry.py             # 策略注册中心
    │
    ├── channels/
    │   ├── vector.py           # ChromaDB 向量通道
    │   ├── fts.py              # SQLite FTS5 通道
    │   └── graph.py            # NetworkX 图通道
    │
    ├── strategies/
    │   ├── base.py             # IndexStrategy 抽象类
    │   ├── standard.py         # 标准索引
    │   ├── summary.py          # 摘要索引
    │   ├── parent_child.py     # 父子索引
    │   └── hypothetical.py     # HyDE 索引
    │
    └── retrieval/
        ├── intent.py           # 意图识别 + 策略路由
        ├── enrich.py           # 完整度判断 + 问题改写
        ├── multi_recall.py     # 多路召回执行
        └── fusion.py           # RRF/加权融合
```

---

## 13. 实施里程碑

| 阶段 | 周期 | 内容 | 验证标准 |
|---|---|---|---|
| **Phase 1** | W1 | DocumentStore + 纯数字 ID + 三通道 | chunks 表可读写，ID 可解析 |
| **Phase 2** | W1 | 四种策略实现 | 可切换 standard/summary/parent_child/hyde |
| **Phase 3** | W1-2 | 多路召回 + 融合 | 并行召回，RRF 融合结果正确 |
| **Phase 4** | W2 | 意图识别 + Enrich | 意图分类准确，不完整问题返回追问 |
| **Phase 5** | W2-3 | LangGraph 编排贯通 | Enrich→Intent→Recall→Fusion→Generate 端到端 |
| **Phase 6** | W3 | API + 前端 | `/api/hybrid/*` 调试接口可用，前端展示意图标签 |
| **Phase 7** | W3-4 | 测试与文档 | 各策略 Recall@K 对比，操作手册更新 |

---

## 14. AgenticRAG 衔接展望（非本期内容）

以下能力不在 HybridRAG（二期）范围内，作为向 AgenticRAG（三期）过渡的规划。二期已预留接口，三期无需改动意图识别框架，只需在路由表注册新意图：

| 能力 | 说明 | 二期预留 | 三期实现 |
|---|---|---|---|
| **Rerank** | 精排模型（Cross-Encoder）对融合结果二次排序 | 融合层接口 | AgenticRAG |
| **SQL 检索** | 结构化数据查询，`skill.sql` 意图触发 SQL 生成 | `skill.sql` 意图已注册（路由表预留） | AgenticRAG |
| **工具引入** | 调用外部 API，`tool.report` 意图触发报表生成 | `tool.report` 意图已注册（路由表预留） | AgenticRAG |
| **Skill 引入** | 可插拔技能模块，`skill.global` / `skill.project` | 意图路径已预留 | AgenticRAG |
| **大视角问答** | 分步骤多次检索，收集所有必要信息后综合回答 | — | AgenticRAG |
| **Agent 架构** | ReAct / Plan-and-Execute 多步推理 | — | AgenticRAG |
| **外置记忆** | 长期对话记忆、用户偏好学习 | — | AgenticRAG |
| **死循环处理** | 错误反传大模型、检索失败降级、最大步数限制 | — | AgenticRAG |

**二期到三期的扩展方式**：
1. 在 `config/intents.yaml` 中注册新意图（如 `skill.sql`、`tool.report`）
2. `IntentRouter` 自动识别新意图，无需改代码
3. 下游根据 `path` 前缀路由到不同处理模块（`retrieve.*` → HybridRAG，`skill.*` → Skill 模块，`tool.*` → Tool 模块）

**配置扩展**：
```yaml
# config/intents.yaml（三期扩展示例）
intents:
  # 二期已有
  retrieve.keyword: {description: "关键词检索", tags: [retrieve, search]}
  retrieve.concept: {description: "概念检索", tags: [retrieve, concept]}
  retrieve.relation: {description: "关系检索", tags: [retrieve, relation]}
  retrieve.detail: {description: "细节检索", tags: [retrieve, detail]}
  
  # 三期新增（只需注册，无需改 IntentRouter 代码）
  skill.sql:
    description: "结构化数据查询，生成 SQL 检索"
    tags: [skill, sql, structured]
  skill.global:
    description: "全局技能调用"
    tags: [skill, global]
  tool.report:
    description: "报表生成工具"
    tags: [tool, report, output]
```

---

## 附录：ID 速查

| 类型 | 编码 | derivative_id 示例 | 解析结果 |
|---|---|---|---|
| Summary | 0 | `01000001000` | chunk=1000001, seq=0 |
| ChildChunk | 1 | `11000001003` | chunk=1000001, seq=3 |
| HyDE | 2 | `21000001002` | chunk=1000001, seq=2 |
| Custom | 3 | `31000001001` | chunk=1000001, seq=1 |

```python
# 解析
def parse(did: int):
    s = str(did).zfill(14)
    return int(s[0]), int(s[1:11]), int(s[11:14])

# 构造
def make(type_code: int, chunk_id: int, seq: int = 0) -> int:
    return int(f"{type_code}{chunk_id:010d}{seq:03d}")
```

---

> **最终版总结**：一期（纯向量 RAG）→ 二期（HybridRAG）= 混合检索（向量/FTS/图）× 索引策略（标准/摘要/父子/HyDE）+ 多路召回 + **两层意图识别（纯路由表：第一层 `retrieve`/`skill`/`tool` 判定方向，第二层 `retrieve.keyword`/`concept`/`relation`/`detail` 多选混合）**+ Enrich（问题完善）。意图识别复用一阶段 OpenRouterLLM，通过 `INTENT_MODEL` 环境变量可配置模型选型。纯数字 ID 设计（14位：类型1位+chunk_id10位+序号3位），与第一阶段结构无缝衔接。AgenticRAG 展望放在文末。
