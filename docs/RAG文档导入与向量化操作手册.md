# RAG 教学项目 — 文档导入与向量化操作手册

> 版本：v1.0 | 更新日期：2026-06-10 | 适用项目：RAG教学

---

## 目录

1. [快速开始](#1-快速开始)
2. [嵌入模型配置](#2-嵌入模型配置)
3. [分块策略配置](#3-分块策略配置)
4. [文档导入方法](#4-文档导入方法)
5. [检索参数配置](#5-检索参数配置)
6. [常见问题与排查](#6-常见问题与排查)
7. [配置速查表](#7-配置速查表)

---

## 1. 快速开始

### 1.1 当前环境状态

| 项目 | 配置 |
|------|------|
| 嵌入提供商 | `openrouter` |
| 嵌入模型 | `nvidia/llama-nemotron-embed-vl-1b-v2:free` |
| 向量维度 | 2048 维 |
| 向量数据库 | ChromaDB（本地持久化） |
| 分块大小 | 500 字符 |
| 分块重叠 | 50 字符 |
| 当前知识库文档数 | 34 条（来自《自指学口播文稿》PDF） |

### 1.2 验证服务状态

```bash
cd backend && source venv/bin/activate

# 检查知识库文档数量
curl -s http://106.54.2.239:5173/api/status | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'文档数: {d[\"documents_count\"]}')"
```

### 1.3 快速测试检索

```bash
# 测试语义检索
curl -s -X POST http://106.54.2.239:5173/runs/stream \
  -H "Content-Type: application/json" \
  -d '{
    "thread_id": null,
    "assistant_id": "agent",
    "input": {"messages": [{"type": "human", "content": "什么是自指学"}]},
    "stream_mode": ["messages", "values"]
  }'
```

---

## 2. 嵌入模型配置

### 2.1 配置文件位置

编辑 `backend/.env`：

```bash
# 嵌入模型提供商（四选一）
EMBEDDING_PROVIDER=openrouter

# 嵌入模型名称
EMBEDDING_MODEL=nvidia/llama-nemotron-embed-vl-1b-v2:free
```

### 2.2 支持的提供商

| 提供商 | 配置值 | 适用场景 | 是否需要联网 | 向量维度 |
|--------|--------|----------|-------------|----------|
| **OpenRouter** | `openrouter` | 免费/低成本在线嵌入 | ✅ 需要 | 依模型而定 |
| **HuggingFace** | `huggingface` | 本地部署，隐私优先 | ❌ 不需要 | 依模型而定 |
| **OpenAI** | `openai` | 高质量商业嵌入 | ✅ 需要 | 1536 / 3072 |
| **Fake（伪嵌入）** | `fake` | 离线测试/开发 | ❌ 不需要 | 384 |

### 2.3 推荐模型配置

#### OpenRouter 免费/低成本模型

```bash
# 当前使用（免费，2048维，多语言支持）
EMBEDDING_PROVIDER=openrouter
EMBEDDING_MODEL=nvidia/llama-nemotron-embed-vl-1b-v2:free

# 备选：OpenAI text-embedding-3-small（通过 OpenRouter）
EMBEDDING_MODEL=openai/text-embedding-3-small

# 备选：OpenAI text-embedding-3-large（更高质量）
EMBEDDING_MODEL=openai/text-embedding-3-large
```

#### HuggingFace 本地模型（需预下载）

```bash
# 需要先下载模型到服务器
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# 或中文优化模型
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
```

> **下载方法**：在能访问 HuggingFace 的机器上运行 `sentence-transformers` 下载脚本，将模型文件打包上传到服务器的 `backend/models/` 目录。

#### OpenAI 官方

```bash
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
# 需额外设置 OPENAI_API_KEY
```

### 2.4 切换模型的注意事项

> ⚠️ **重要**：不同模型的向量维度不同，**切换模型后必须重建知识库**！

```bash
# 1. 停止后端服务
lsof -ti:47569 | xargs kill -9

# 2. 删除旧向量数据库
rm -rf backend/chroma_db
mkdir -p backend/chroma_db

# 3. 修改 .env 中的模型配置
vim backend/.env

# 4. 重新启动后端
 cd backend && source venv/bin/activate && langgraph dev --host 0.0.0.0 --port 47569

# 5. 重新索引文档
python3 -c "
import sys; sys.path.insert(0, 'src')
from agent.vector_store import VectorStore
vs = VectorStore()
vs.index_file('docs/你的文档.pdf')
"
```

---

## 3. 分块策略配置

### 3.1 当前分块配置

编辑 `backend/src/agent/vector_store.py`：

```python
self.splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # 每个块的最大字符数
    chunk_overlap=50,    # 相邻块之间的重叠字符数
)
```

### 3.2 分块策略选择指南

| 文档类型 | 推荐 chunk_size | 推荐 chunk_overlap | 说明 |
|----------|----------------|-------------------|------|
| **学术论文** | 800-1200 | 100-200 | 保留完整论证段落 |
| **技术文档** | 500-800 | 50-100 | 平衡粒度与上下文 |
| **口播/演讲稿** | 400-600 | 50-80 | 口语化内容，段落较短 |
| **法律合同** | 1000-1500 | 200-300 | 需要完整条款上下文 |
| **代码文档** | 300-500 | 30-50 | 函数级粒度 |
| **FAQ/问答** | 200-400 | 20-40 | 问题-答案对独立成块 |

### 3.3 分块策略调整方法

```python
# 示例：调整为适合学术论文的配置
from langchain_text_splitters import RecursiveCharacterTextSplitter

self.splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # 增大块大小，保留更多上下文
    chunk_overlap=150,      # 增加重叠，避免边界信息丢失
    separators=["\n\n", "\n", "。", "；", " ", ""],  # 自定义分隔符优先级
)
```

### 3.4 高级分块策略

#### 按 Markdown 标题分块（适合结构化文档）

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
self.splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
```

#### 按字符数 + 语义边界分块

```python
from langchain_text_splitters import CharacterTextSplitter

self.splitter = CharacterTextSplitter(
    separator="\n\n",        # 优先按段落分割
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,     # 按字符计数
    is_separator_regex=False,
)
```

---

## 4. 文档导入方法

### 4.1 支持的文件格式

| 格式 | 扩展名 | 说明 |
|------|--------|------|
| PDF | `.pdf` | 自动提取文本内容 |
| 纯文本 | `.txt` | UTF-8 编码 |
| Markdown | `.md` | 保留格式标记 |

### 4.2 方法 A：通过前端上传（推荐）

```bash
# 调用后端上传接口
curl -X POST http://106.54.2.239:5173/api/upload \
  -F "file=@/path/to/your/document.pdf"
```

响应示例：
```json
{
  "status": "ok",
  "filename": "document.pdf",
  "message": "文档已索引到知识库"
}
```

### 4.3 方法 B：批量索引目录

```bash
cd backend && source venv/bin/activate

python3 << 'PYEOF'
import sys; sys.path.insert(0, 'src')
from agent.vector_store import VectorStore

vs = VectorStore()

# 索引整个 docs 目录
vs.index_directory('/home/ubuntu/.openclaw/workspace/RAG教学/docs')

# 验证
print(f"知识库文档数: {vs.db._collection.count()}")
PYEOF
```

### 4.4 方法 C：索引单个文件

```bash
cd backend && source venv/bin/activate

python3 << 'PYEOF'
import sys; sys.path.insert(0, 'src')
from agent.vector_store import VectorStore

vs = VectorStore()

# 索引单个 PDF
vs.index_file('/home/ubuntu/.openclaw/workspace/RAG教学/docs/新文档.pdf')

# 或索引单个文本文件
vs.index_file('/home/ubuntu/.openclaw/workspace/RAG教学/docs/笔记.txt')

print(f"知识库文档数: {vs.db._collection.count()}")
PYEOF
```

### 4.5 导入后验证

```bash
# 检查文档数量
curl -s http://106.54.2.239:5173/api/status

# 测试检索
curl -s -X POST http://106.54.2.239:5173/runs/stream \
  -H "Content-Type: application/json" \
  -d '{
    "thread_id": null,
    "assistant_id": "agent",
    "input": {"messages": [{"type": "human", "content": "测试问题"}]},
    "stream_mode": ["messages"]
  }'
```

---

## 5. 检索参数配置

### 5.1 检索数量（Top-K）

编辑 `backend/src/agent/graph.py`：

```python
# 节点 2：检索
def retrieve_node(state: RAGState, config: RunnableConfig) -> RAGState:
    # ...
    docs = vector_store.search(question, k=4)  # 返回最相关的 4 条文档
```

| 场景 | 推荐 k 值 | 说明 |
|------|----------|------|
| **简单问答** | 2-3 | 问题明确，文档冗余度低 |
| **综合推理** | 4-6 | 需要多源信息交叉验证 |
| **开放探索** | 8-10 | 用户问题模糊，需要广泛检索 |
| **长文档生成** | 6-8 | 需要大量上下文支撑 |

### 5.2 检索阈值（相似度过滤）

当前实现使用 Chroma 的默认余弦相似度，未设置阈值。如需添加阈值过滤：

```python
# 在 vector_store.py 中修改 search 方法
def search(self, query: str, k: int = 4, score_threshold: float = None) -> list:
    """检索与问题最相关的文档片段"""
    results = self.db.similarity_search_with_score(query, k=k)
    
    if score_threshold:
        # 过滤低相似度结果（Chroma 返回的是距离，越小越相似）
        results = [(doc, score) for doc, score in results if score < score_threshold]
    
    return [doc for doc, _ in results]
```

### 5.3 混合检索策略（高级）

结合关键词检索 + 向量检索：

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# 创建混合检索器
bm25_retriever = BM25Retriever.from_documents(all_docs, k=3)
vector_retriever = self.db.as_retriever(search_kwargs={"k": 3})

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.3, 0.7],  # 向量检索权重更高
)
```

---

## 6. 常见问题与排查

### 6.1 嵌入模型切换后检索失败

**现象**：切换模型后，查询报错或返回无关结果

**原因**：向量维度不匹配

**解决**：
```bash
# 删除旧向量数据库，重建索引
rm -rf backend/chroma_db
mkdir -p backend/chroma_db
# 然后重新索引所有文档
```

### 6.2 OpenRouter 嵌入 API 返回空

**现象**：`ValueError: No embedding data received`

**原因**：`langchain-openai` 版本与 OpenRouter 响应格式不兼容

**解决**：已使用自定义 `OpenRouterEmbeddings` 类绕过此问题，无需额外处理。

### 6.3 检索结果不相关

**排查步骤**：

1. **检查嵌入模型是否正常工作**：
```python
vs = VectorStore()
emb = vs.embedding.embed_documents(["测试文本"])
print(f"嵌入维度: {len(emb[0])}")  # 应输出 2048
```

2. **检查知识库是否为空**：
```python
print(vs.db._collection.count())  # 应 > 0
```

3. **调整分块大小**：增大 `chunk_size` 保留更多上下文

4. **调整检索数量**：增大 `k` 值获取更多候选

### 6.4 服务器无外网访问（HuggingFace 模型下载失败）

**方案 A**：使用 OpenRouter 在线嵌入（当前方案）

**方案 B**：在有网机器下载模型，上传到服务器：
```bash
# 在有网机器
pip install sentence-transformers
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2').save('./models/all-MiniLM-L6-v2')"
tar -czvf model.tar.gz models/
scp model.tar.gz ubuntu@106.54.2.239:/home/ubuntu/

# 在服务器
cd /home/ubuntu/.openclaw/workspace/RAG教学/backend
mkdir -p models
tar -xzvf ~/model.tar.gz -C models/
```

---

## 7. 配置速查表

### 7.1 环境变量配置

```bash
# backend/.env

# === LLM 配置 ===
OPENROUTER_API_KEY=sk-or-...
DEFAULT_MODEL=moonshotai/kimi-k2.6:free

# === 嵌入模型配置 ===
EMBEDDING_PROVIDER=openrouter          # openrouter | huggingface | openai | fake
EMBEDDING_MODEL=nvidia/llama-nemotron-embed-vl-1b-v2:free

# === 存储配置 ===
CHROMA_DB_PATH=./chroma_db
UPLOAD_DIR=./uploads
```

### 7.2 代码配置项

| 配置项 | 文件位置 | 默认值 | 说明 |
|--------|----------|--------|------|
| chunk_size | `vector_store.py` | 500 | 分块大小 |
| chunk_overlap | `vector_store.py` | 50 | 分块重叠 |
| 检索数量 k | `graph.py` | 4 | Top-K 检索 |
| 批处理大小 | `vector_store.py` | 100 | OpenRouter 嵌入批处理 |

### 7.3 常用命令

```bash
# 启动后端
cd backend && source venv/bin/activate && langgraph dev --host 0.0.0.0 --port 47569

# 启动前端
cd frontend && npx vite --host 0.0.0.0 --port 5173

# 检查知识库状态
curl -s http://106.54.2.239:5173/api/status

# 重建知识库（切换模型后）
rm -rf backend/chroma_db && mkdir -p backend/chroma_db

# 查看后端日志
tail -f /tmp/langgraph.log

# 查看前端日志
tail -f /tmp/vite.log
```

---

## 附录：当前项目文件结构

```
RAG教学/
├── backend/
│   ├── .env                    # 环境变量配置（嵌入模型等）
│   ├── .env.example            # 配置模板
│   ├── src/
│   │   └── agent/
│   │       ├── vector_store.py # 向量数据库 + 嵌入模型配置
│   │       ├── graph.py        # LangGraph 工作流（含检索参数 k）
│   │       ├── app.py          # FastAPI 接口（上传/状态）
│   │       ├── llm.py          # LLM 配置
│   │       └── state.py        # 状态定义
│   ├── chroma_db/              # 向量数据库持久化目录
│   └── venv/                   # Python 虚拟环境
├── frontend/
│   ├── src/App.tsx             # 前端主组件
│   ├── vite.config.ts          # Vite 代理配置
│   └── .env                    # 前端环境变量
├── docs/                       # 文档存放目录
│   └── 自指学口播文稿_第三版.pdf
└── docs/                       # 本手册存放目录
    └── RAG文档导入与向量化操作手册.md
```

---

> **维护提示**：修改本手册后，建议同步更新 `backend/.env.example` 中的注释说明，保持文档一致性。
