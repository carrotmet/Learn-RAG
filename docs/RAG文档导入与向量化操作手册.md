# RAG 教学项目 — 文档导入与向量化操作手册

> 版本：v1.1 | 更新日期：2026-06-12 | 适用项目：RAG教学

---

## 目录

1. [快速开始](#1-快速开始)
2. [嵌入模型配置](#2-嵌入模型配置)
3. [分块策略配置](#3-分块策略配置)
4. [文档导入方法](#4-文档导入方法)
5. [检索参数配置](#5-检索参数配置)
6. [数据采集与导出](#6-数据采集与导出)
7. [常见问题与排查](#7-常见问题与排查)
8. [配置速查表](#8-配置速查表)

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
| 当前知识库文档数 | 106 条（含《自指学口播文稿》+《Diagonal Arguments》PDF） |

### 1.2 验证服务状态

```bash
cd backend && source venv/bin/activate

# 检查知识库文档数量
curl -s http://127.0.0.1:47569/api/status | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'文档数: {d[\"documents_count\"]}')"

# 检查采集统计
curl -s http://127.0.0.1:47569/api/collect/statistics | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(d.get('statistics', {}), indent=2))"
```

### 1.3 快速测试检索

```bash
# 测试语义检索
curl -s -X POST http://127.0.0.1:47569/runs/stream \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id": "fe096781-5601-53d2-b2f6-0d3403f7e9ca",
    "input": {"messages": [{"role": "human", "content": "什么是对角线论证"}]},
    "stream_mode": ["messages"]
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
curl -X POST http://127.0.0.1:47569/api/upload \
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
curl -s http://127.0.0.1:47569/api/status

# 测试检索
curl -s -X POST http://127.0.0.1:47569/runs/stream \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id": "fe096781-5601-53d2-b2f6-0d3403f7e9ca",
    "input": {"messages": [{"role": "human", "content": "测试问题"}]},
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

---

## 6. 数据采集与导出

### 6.1 功能概述

本项目已集成轻量级数据采集系统，支持 **在线采集**（API 调用时自动写入）和 **数据导出**（JSONL 格式）。

| 功能 | 说明 | 存储位置 |
|------|------|----------|
| **在线采集** | RAG 对话时自动保存 question/answer/contexts | SQLite `data/rag_data.db` |
| **手动采集** | 通过 API 手动上传对话记录 | SQLite `data/rag_data.db` |
| **离线上传** | 批量导入 JSON/JSONL/CSV/Excel | SQLite `raw_data` 表 |
| **数据导出** | 导出为 JSONL（RAGAS 兼容） | `data/export_*.jsonl` |

### 6.2 数据库表结构

**路径**: `backend/data/rag_data.db`

| 表名 | 说明 | 字段 |
|------|------|------|
| `conversations` | 用户对话记录 | id, question, answer, contexts, ground_truth, model_version, timestamp, source, metadata, processing_stage |
| `retrieval_logs` | 检索结果日志 | id, conversation_id, query, retrieved_chunks, scores, latency_ms, timestamp |
| `llm_calls` | LLM 调用记录 | id, conversation_id, prompt, response, model_name, token_usage, latency_ms, timestamp |
| `user_feedback` | 用户反馈 | id, conversation_id, feedback_type, content, rating, timestamp |
| `raw_data` | 离线上传原始数据 | id, source_type, original_format, raw_content, upload_batch, metadata, status, timestamp |
| `processed_data` | 解析后的标准格式数据 | id, raw_id, question, question_type, domain, difficulty, contexts, answer, ground_truth, metadata, evaluation, processing_stage, created_at |

### 6.3 在线采集（自动）

RAG 对话时自动采集，无需手动操作。每次对话会记录：
- 对话内容（question, answer, contexts）
- 检索日志（query, retrieved_chunks, latency_ms）
- LLM 调用记录（prompt, response, model_name, token_usage）

采集使用**后台线程**，不阻塞主流程。

### 6.4 手动采集接口

#### 采集对话

```bash
curl -X POST http://127.0.0.1:47569/api/collect/conversation \
  -H "Content-Type: application/json" \
  -d '{
    "question": "用户问题",
    "answer": "系统回答",
    "contexts": ["上下文片段1", "上下文片段2"],
    "source": "online_api",
    "metadata": {"custom": "data"}
  }'
```

#### 采集用户反馈

```bash
curl -X POST http://127.0.0.1:47569/api/collect/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "uuid",
    "feedback_type": "thumbs_up",
    "content": "回答很准确",
    "rating": 5
  }'
```

#### 查看采集统计

```bash
curl -s http://127.0.0.1:47569/api/collect/statistics | python3 -m json.tool
```

#### 查看最近对话

```bash
curl -s "http://127.0.0.1:47569/api/collect/conversations?limit=10&source=online_api" | python3 -m json.tool
```

### 6.5 数据导出接口

#### 导出对话数据

```bash
curl -X POST http://127.0.0.1:47569/api/export/conversations \
  -H "Content-Type: application/json" \
  -d '{
    "output_path": "data/export_conversations.jsonl",
    "conditions": "source = \"online_api\"",
    "limit": 1000
  }'
```

#### 导出标准格式数据（测试集用）

```bash
curl -X POST http://127.0.0.1:47569/api/export/processed \
  -H "Content-Type: application/json" \
  -d '{
    "output_path": "data/export_processed.jsonl",
    "limit": 1000
  }'
```

#### 导出原始数据

```bash
curl -X POST http://127.0.0.1:47569/api/export/raw \
  -H "Content-Type: application/json" \
  -d '{
    "output_path": "data/export_raw.jsonl",
    "conditions": "status = \"parsed\"",
    "limit": 1000
  }'
```

#### 查看导出文件状态

```bash
curl -s http://127.0.0.1:47569/api/export/status | python3 -m json.tool
```

### 6.6 CLI 导出工具

```bash
cd backend && source venv/bin/activate

# 预览数据
python -m data_collection.exporter --output data/preview.jsonl --preview --limit 5

# 导出对话
python -m data_collection.exporter --output data/export.jsonl --table conversations

# 导出测试集（RAGAS 格式）
python -m data_collection.exporter --output data/testset.jsonl --testset

# 查看摘要
python -m data_collection.exporter --output data/summary.json --summary
```

### 6.7 Python API 导出

```python
from data_collection.exporter import DataExporter

exporter = DataExporter(db_path="data/rag_data.db")

# 导出对话
count = exporter.export_conversations("data/export.jsonl", limit=1000)

# 导出测试集（RAGAS 兼容）
count = exporter.export_testset(
    "data/testset.jsonl",
    testset_type="validation",
    domain="数学",
    limit=500
)

# 预览
preview = exporter.get_export_preview("conversations", limit=5)

# 摘要
summary = exporter.get_export_summary("conversations")
```

---

## 7. 常见问题与排查

### 7.1 嵌入模型切换后检索失败

**现象**：切换模型后，查询报错或返回无关结果

**原因**：向量维度不匹配

**解决**：
```bash
# 删除旧向量数据库，重建索引
rm -rf backend/chroma_db
mkdir -p backend/chroma_db
# 然后重新索引所有文档
```

### 7.2 OpenRouter 嵌入 API 返回空

**现象**：`ValueError: No embedding data received`

**原因**：`langchain-openai` 版本与 OpenRouter 响应格式不兼容

**解决**：已使用自定义 `OpenRouterEmbeddings` 类绕过此问题，无需额外处理。

### 7.3 检索结果不相关

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

### 7.4 采集接口返回阻塞错误

**现象**：`Blocking call to sqlite3.Connection.execute`

**原因**：SQLite 同步写入阻塞了 ASGI 事件循环

**解决**：已使用 `asyncio.to_thread()` 包装所有 SQLite 操作，无需额外处理。

### 7.5 服务器无外网访问（HuggingFace 模型下载失败）

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

## 8. 配置速查表

### 8.1 环境变量配置

```bash
# backend/.env

# === LLM 配置 ===
OPENROUTER_API_KEY=sk-or-...
DEFAULT_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free
FALLBACK_MODELS=meta-llama/llama-3.3-70b-instruct:free,openai/gpt-oss-20b:free

# === 嵌入模型配置 ===
EMBEDDING_PROVIDER=openrouter          # openrouter | huggingface | openai | fake
EMBEDDING_MODEL=nvidia/llama-nemotron-embed-vl-1b-v2:free

# === 存储配置 ===
CHROMA_DB_PATH=./chroma_db
UPLOAD_DIR=./uploads
```

### 8.2 代码配置项

| 配置项 | 文件位置 | 默认值 | 说明 |
|--------|----------|--------|------|
| chunk_size | `vector_store.py` | 500 | 分块大小 |
| chunk_overlap | `vector_store.py` | 50 | 分块重叠 |
| 检索数量 k | `graph.py` | 4 | Top-K 检索 |
| 批处理大小 | `vector_store.py` | 100 | OpenRouter 嵌入批处理 |
| 采集数据库 | `sqlite_store.py` | `data/rag_data.db` | SQLite 数据库路径 |

### 8.3 API 接口速查

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/upload` | POST | 上传文档并索引 |
| `/api/status` | GET | 知识库状态 |
| `/api/collect/conversation` | POST | 手动采集对话 |
| `/api/collect/feedback` | POST | 采集用户反馈 |
| `/api/collect/statistics` | GET | 采集统计 |
| `/api/collect/conversations` | GET | 最近对话 |
| `/api/export/conversations` | POST | 导出对话 |
| `/api/export/processed` | POST | 导出标准格式 |
| `/api/export/raw` | POST | 导出原始数据 |
| `/api/export/status` | GET | 导出文件状态 |
| `/api/testset/import` | POST | 3.2 离线数据导入 |
| `/api/testset/parse` | POST | 3.3 数据解析 |
| `/api/testset/build` | POST | 3.4 测试集搭建 |
| `/api/testset/versions` | GET | 测试集版本列表 |
| `/api/testset/pipeline` | GET | 完整流水线运行 |
| `/api/evaluate/testset` | POST | 4.1 RAGAS 评估测试集 |
| `/api/evaluate/single` | POST | 4.1 单条评估 |
| `/api/evaluate/summary` | GET | 4.1 评估汇总 |
| `/api/evaluate/failures` | GET | 4.1 低分样本 |
| `/api/evaluate/report` | POST | 4.2 生成可视化报告 |
| `/api/evaluate/all` | POST | 4.1-4.2 评估所有测试集+报告 |

### 6.8 测试集搭建（3.1-3.4）

#### 完整流水线

```bash
# 一键运行完整测试集搭建流水线
curl -X GET "http://127.0.0.1:47569/api/testset/pipeline?db_path=data/rag_data.db&output_prefix=data/testset"
```

#### 分步操作

```bash
# 3.2 数据导入（从 conversations 导入到 processed_data）
curl -X POST http://127.0.0.1:47569/api/testset/import \
  -H "Content-Type: application/json" \
  -d '{"source": "all"}'

# 3.3 数据解析（parsed -> validated）
curl -X POST http://127.0.0.1:47569/api/testset/parse \
  -H "Content-Type: application/json" \
  -d '{"stage": "parsed"}'

# 3.4 测试集搭建
curl -X POST http://127.0.0.1:47569/api/testset/build \
  -H "Content-Type: application/json" \
  -d '{
    "output_prefix": "data/testset",
    "golden_size": 20,
    "validation_size": 50,
    "stress_size": 10
  }'

# 查看版本列表
curl -s http://127.0.0.1:47569/api/testset/versions | python3 -m json.tool
```

#### CLI 测试集搭建

```bash
cd backend && source venv/bin/activate

# 完整流水线
python src/testset/test_pipeline.py

# 分步操作
python -m testset.testset_builder --action import --db data/rag_data.db
python -m testset.testset_builder --action parse --db data/rag_data.db
python -m testset.testset_builder --action build --db data/rag_data.db --output data/testset
```

### 6.9 RAGAS 评估与可视化（4.1-4.2）

#### 评估测试集

```bash
# 评估 golden 测试集
curl -X POST http://127.0.0.1:47569/api/evaluate/testset \
  -H "Content-Type: application/json" \
  -d '{
    "testset_path": "data/testset_golden.jsonl",
    "testset_version": "v1_golden",
    "testset_type": "golden"
  }'

# 评估所有测试集并生成报告
curl -X POST http://127.0.0.1:47569/api/evaluate/all
```

#### 单条评估

```bash
curl -X POST http://127.0.0.1:47569/api/evaluate/single \
  -H "Content-Type: application/json" \
  -d '{
    "question": "什么是对角线论证？",
    "answer": "对角线论证是康托尔提出的证明方法。",
    "contexts": ["对角线论证由康托尔提出，用于证明实数集不可数。"]
  }'
```

#### 生成报告

```bash
# 生成完整报告
curl -X POST http://127.0.0.1:47569/api/evaluate/report \
  -H "Content-Type: application/json" \
  -d '{"testset_version": "v1_golden", "output_path": "reports/evaluation.html"}'

# 生成简化报告
curl -X POST http://127.0.0.1:47569/api/evaluate/report \
  -H "Content-Type: application/json" \
  -d '{"testset_version": "v1_golden", "mini": true}'

# 查看评估汇总
curl -s "http://127.0.0.1:47569/api/evaluate/summary?testset_version=v1_golden"

# 查看低分样本
curl -s "http://127.0.0.1:47569/api/evaluate/failures?testset_version=v1_golden&limit=5"
```

#### CLI 评估工具

```bash
cd backend && source venv/bin/activate

# 评估测试集
python -m evaluation.ragas_eval --testset data/testset_golden.jsonl --version v1

# 评估单条
python -m evaluation.ragas_eval --single

# 生成报告
python -m feedback.visualizer --version v1 --output reports/evaluation.html
```

### 8.4 常用命令

```bash
# 启动后端
cd backend && source venv/bin/activate && langgraph dev --host 0.0.0.0 --port 47569

# 启动前端
cd frontend && npx vite --host 0.0.0.0 --port 5173

# 检查知识库状态
curl -s http://127.0.0.1:47569/api/status

# 检查采集统计
curl -s http://127.0.0.1:47569/api/collect/statistics

# 重建知识库（切换模型后）
rm -rf backend/chroma_db && mkdir -p backend/chroma_db

# 查看后端日志
tail -f /tmp/rag-backend-47569.log

# 查看前端日志
tail -f /tmp/vite.log
```

---

## 附录：当前项目文件结构

```
RAG教学/
├── backend/
│   ├── .env                    # 环境变量配置
│   ├── src/
│   │   ├── agent/
│   │   │   ├── vector_store.py # 向量数据库 + 嵌入模型
│   │   │   ├── graph.py        # LangGraph 工作流（含在线采集）
│   │   │   ├── app.py          # FastAPI 接口（上传/采集/导出/测试集搭建/评估）
│   │   │   ├── llm.py          # OpenRouter LLM 封装
│   │   │   └── state.py        # 状态定义
│   │   ├── data_collection/    # 数据采集模块（2.1-2.4）
│   │   │   ├── sqlite_store.py # SQLite 存储层（7张表+testset_versions）
│   │   │   ├── uploader.py     # 离线上传解析器
│   │   │   ├── exporter.py     # 数据导出器（JSONL/RAGAS）
│   │   │   └── config.py       # 采集配置
│   │   ├── testset/            # 测试集搭建模块（3.1-3.4）
│   │   │   ├── __init__.py
│   │   │   ├── testset_builder.py # 核心模块：导入/解析/搭建/版本管理
│   │   │   └── test_pipeline.py   # 测试脚本：虚拟数据生成+流程验证
│   │   ├── evaluation/         # RAGAS 评估模块（4.1）
│   │   │   └── ragas_eval.py   # RAGAS 评估器：指标计算/结果存储
│   │   └── feedback/           # 可视化报告模块（4.2）
│   │       └── visualizer.py   # HTML 报告生成：指标看板/趋势/错误分析
│   ├── chroma_db/              # 向量数据库持久化
│   └── data/                   # SQLite 数据库 + 导出文件 + 测试集 + 报告
│       └── reports/            # 评估报告 HTML
├── frontend/
│   ├── src/App.tsx
│   └── vite.config.ts
├── docs/
│   ├── RAG文档导入与向量化操作手册.md
│   ├── 监控指标体系搭建规划.md
│   ├── 数据库清单与作用说明.md
│   └── *.pdf                   # 文档文件
└── skills/rag-teaching-info/   # 项目技能文件
```

---

> **维护提示**：修改本手册后，建议同步更新 `skills/rag-teaching-info/SKILL.md` 技能文件，保持文档一致性。
> **最后更新**：2026-06-12