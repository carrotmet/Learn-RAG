# HybridRAG 文档上传与处理 SOP (v2.2 向量质量版)

> 版本: v2.2
> 适用阶段: HybridRAG 第二阶段（策略索引层 5.1-5.3）
> 最后更新: 2026-06-22

---

## 一、概述

本文档定义 HybridRAG 系统中，从原始文档上传到多策略索引完成的 CLI 操作流程。

**v2.0 核心变更**：
- 使用 `python -m hybrid.cli` 命令替代手写 Python 脚本
- 支持一键索引、状态查看、检索测试
- 策略与通道映射自动处理，无需手动控制

---

## 二、CLI 工具

### 2.1 CLI 位置与运行方式

```bash
# 方式一：在项目根目录执行（推荐）
cd RAG教学
PYTHONPATH=backend/src python -m hybrid.cli <command> [options]

# 方式二：在 backend/src 目录执行
cd RAG教学/backend/src
python -m hybrid.cli <command> [options]
```

### 2.2 命令速查

| 命令 | 说明 | 示例 |
|------|------|------|
| `index` | 索引 PDF 文档 | `python -m hybrid.cli index doc.pdf` |
| `status` | 查看索引状态 | `python -m hybrid.cli status` |
| `search` | 检索测试 | `python -m hybrid.cli search "自指"` |

---

## 三、前置条件

### 3.1 环境要求

**必须先激活虚拟环境**，否则会出现 `ModuleNotFoundError: No module named 'langchain_core'` 错误。

```bash
# 进入项目根目录
cd RAG教学

# 激活虚拟环境
source backend/venv/bin/activate

# 确认依赖安装
pip list | grep -E "langchain|chromadb|networkx|pdfplumber"

# 确认环境变量配置
cat backend/.env | grep -E "^(DEFAULT_MODEL|OPENROUTER_API_KEY)"
```

> ⚠️ **常见错误**: 未激活虚拟环境直接运行 `python3 -m hybrid.cli` 会报错找不到依赖。务必先 `source backend/venv/bin/activate`。  

### 3.2 模型配置

**.env 中配置（用于 summary/hypothetical 策略的 LLM 生成）**:

```bash
DEFAULT_MODEL=qwen/qwen3.5-flash-02-23
OPENROUTER_API_KEY=sk-or-...
```

### 3.3 嵌入模型配置（⚠️ 向量质量关键）

**必须配置嵌入模型**，否则向量检索将使用假向量（FakeEmbeddings），导致语义搜索完全失效。

```bash
# .env 中配置（用于生成真实语义向量）
EMBEDDING_PROVIDER=openrouter
EMBEDDING_MODEL=nvidia/llama-nemotron-embed-vl-1b-v2:free
```

**验证配置生效**（索引前务必检查）：

```bash
cd RAG教学
source backend/venv/bin/activate
PYTHONPATH=backend/src python -c "
from hybrid.channels.vector import VectorChannel
vc = VectorChannel()
print(type(vc.embedding).__name__)
"
```

**预期输出**:
```
[VectorChannel] 使用 OpenRouter 嵌入: nvidia/llama-nemotron-embed-vl-1b-v2:free
OpenRouterEmbeddings
```

**⚠️ 危险信号**（如果输出以下内容，说明向量为假）：
```
[VectorChannel] 使用 FakeEmbeddings（测试模式）
FakeEmbeddings
```

**FakeEmbeddings 的危害**:
- 向量由 MD5 哈希生成，无真实语义
- 语义搜索（vector）完全失效，只能依赖关键词匹配（fts）
- 不同文档的向量可能随机相似或相异，检索结果不可靠

---

## 四、操作流程（CLI 方式）

### 步骤 1: 索引文档（全部策略）

```bash
# 在项目根目录执行（推荐）
cd RAG教学
source backend/venv/bin/activate
PYTHONPATH=backend/src python -m hybrid.cli index docs/自指学口播文稿_第三版.pdf
```

**路径说明**:
- PDF 路径支持**相对项目根目录**（如 `docs/xxx.pdf`）
- 也支持**绝对路径**（如 `/home/ubuntu/.../xxx.pdf`）
- 路径会自动解析，无论从项目根目录还是 `backend/src` 执行都能正确找到文件

**输出示例**:
```
============================================================
HybridRAG 文档索引
============================================================
PDF: docs/自指学口播文稿_第三版.pdf
DB:  /home/ubuntu/.../backend/data/rag_data.db
策略: standard,summary,parent_child,hypothetical
模型: qwen/qwen3.5-flash-02-23
嵌入模型: nvidia/llama-nemotron-embed-vl-1b-v2:free (OpenRouter)  ← 新增

[1/4] 初始化组件...
  → DocumentStore, Vector, FTS, Graph 就绪
  → Embedding: OpenRouterEmbeddings  ← 确认不是 FakeEmbeddings

[2/4] 加载 PDF...
  → 10 页, 18500 字

[3/4] 分块保存...
  → 10 个父块 (chunk_id: 1000001~1000010)

[4/4] 构建策略索引...

  [standard] StandardStrategy...
    → 索引: 10 条, 耗时: 0.5s
    → 通道: vector/fts/graph

  [parent_child] ParentChildStrategy...
    → 索引: 55 条, 耗时: 1.2s
    → 通道: vector/fts

  [summary] SummaryStrategy...
    → 索引: 10 条, 耗时: 45.3s  ← LLM 生成摘要
    → 通道: vector/fts

  [hypothetical] HypotheticalStrategy...
    → 索引: 30 条, 耗时: 38.7s  ← LLM 生成问题
    → 通道: vector

============================================================
索引完成
============================================================
  chunks:       10
  derivatives:  95
  indexed:      315
  ChromaDB:     105
  FTS5:         105
  Graph:        10 nodes, 126 edges
```

> **注意**: summary 和 hypothetical 策略依赖 LLM，耗时较长。若模型限流，策略会自动 fallback（摘要取前100字，问题使用通用模板）。

---

### 步骤 2: 选择策略索引

```bash
# 仅使用标准策略（快速，无 LLM）
PYTHONPATH=backend/src python -m hybrid.cli index docs/神经-心血管机制-20260301.pdf --strategies standard

# 使用标准+摘要策略
PYTHONPATH=backend/src python -m hybrid.cli index docs/神经-心血管机制-20260301.pdf --strategies standard,summary

# 使用标准+父子策略
PYTHONPATH=backend/src python -m hybrid.cli index docs/神经-心血管机制-20260301.pdf --strategies standard,parent_child
```

---

### 步骤 3: 重置后重新索引

```bash
# 删除旧数据，重新索引
PYTHONPATH=backend/src python -m hybrid.cli index docs/神经-心血管机制-20260301.pdf --reset
```

---

### 步骤 4: 查看索引状态

```bash
PYTHONPATH=backend/src python -m hybrid.cli status
```

**输出示例**:
```
============================================================
HybridRAG 状态
============================================================
  DB: .../backend/data/rag_data.db
  chunks:      10
  derivatives: 95
  indexed:     315

  各策略派生:
    standard      :  0 条,  0 个 chunk
    parent_child  : 55 条, 10 个 chunk
    summary       : 10 条, 10 个 chunk
    hypothetical  : 30 条, 10 个 chunk

  各策略索引状态:
    hypothetical  / vector  : 30 条
    parent_child  / fts     : 55 条
    parent_child  / vector  : 55 条
    standard      / fts     : 10 条
    standard      / graph   : 10 条
    standard      / vector  : 10 条
    summary       / fts     : 10 条
    summary       / vector  : 10 条
```

---

### 步骤 5: 索引后向量质量检查（⚠️ 关键）

索引完成后，**务必验证向量是否为真实嵌入**。如果向量为假（FakeEmbeddings），语义搜索将完全失效。

**快速验证**:
```bash
PYTHONPATH=backend/src python -c "
import chromadb, numpy as np
from hybrid.config import DEFAULT_CHROMA_DIR
client = chromadb.PersistentClient(path=DEFAULT_CHROMA_DIR)
col = client.get_collection('hybrid_docs')
data = col.get(include=['embeddings'], limit=5)
arr = np.array(data['embeddings'])
print(f'均值: {arr.mean():.4f}, 最小: {arr.min():.4f}, 最大: {arr.max():.4f}')
if arr.mean() > 0.4 and arr.min() >= 0:
    print('⚠️ 警告: 向量可能是 FakeEmbeddings（值全在0-1之间，均值~0.5）')
else:
    print('✅ 向量分布正常（真实嵌入）')
"
```

**预期输出（真实嵌入）**:
```
均值: -0.0012, 最小: -0.8934, 最大: 0.9123
✅ 向量分布正常（真实嵌入）
```

**危险信号（FakeEmbeddings）**:
```
均值: 0.4995, 最小: 0.0000, 最大: 0.9999
⚠️ 警告: 向量可能是 FakeEmbeddings（值全在0-1之间，均值~0.5）
```

**处理**: 如果发现是 FakeEmbeddings，请检查 `.env` 中的 `EMBEDDING_PROVIDER` 和 `OPENROUTER_API_KEY` 配置，然后 `--reset` 重新索引。

---

### 步骤 6: 检索测试

```bash
# 全部策略检索
PYTHONPATH=backend/src python -m hybrid.cli search "自指"

# 指定策略检索
PYTHONPATH=backend/src python -m hybrid.cli search "自指" --strategy summary

# 返回更多结果
PYTHONPATH=backend/src python -m hybrid.cli search "哥德尔" -k 5
```

**输出示例**:
```
============================================================
检索: '自指'
============================================================

[standard]...
  命中 4 条 (top 3):
  [1] score=0.5234, channel=vector
      自指这一概念如何从哥德尔理论延伸到大语言模型？...
  [2] score=0.4812, channel=fts
      为什么大多数生物、社会及 AI 研究中的"自指"无法满足...
  [3] score=0.4521, channel=graph
      自指是指一个语句、概念或系统在其表达中定义中直接指向...

[summary]...
  命中 2 条 (top 3):
  [1] score=0.6123, channel=vector
      【摘要】本文探讨自指概念从哥德尔不完备定理到当代AI...
      
      【原文】自指是一个有趣的概念，它指的是一个系统或...
```

---

## 五、策略与通道映射

| 策略 | CLI 参数 | 写入通道 | 说明 |
|------|---------|---------|------|
| standard | `standard` | vector + fts + graph | 父块原文，全通道索引 |
| summary | `summary` | vector + fts | LLM 生成摘要，语义+关键词 |
| parent_child | `parent_child` | vector + fts | 300字子块，细粒度检索 |
| hypothetical | `hypothetical` | vector | LLM 生成假设问题，仅语义 |

> **注意**: 策略内部已通过 `supported_channels` 自动过滤通道，CLI 无需手动控制。

---

## 六、常见问题与排查

### 6.1 未激活虚拟环境

**现象**: `ModuleNotFoundError: No module named 'langchain_core'`

**原因**: 未执行 `source backend/venv/bin/activate`，使用的是系统 Python。

**处理**:
```bash
cd RAG教学
source backend/venv/bin/activate
PYTHONPATH=backend/src python -m hybrid.cli ...
```

### 6.2 文件不存在

**现象**: `❌ 文件不存在: docs/xxx.pdf`

**原因**: PDF 路径相对于当前工作目录找不到文件。

**处理**:
1. 确认文件确实存在: `ls docs/xxx.pdf`
2. 使用绝对路径: `python -m hybrid.cli index /home/ubuntu/.../xxx.pdf`
3. 使用相对项目根目录的路径: `docs/xxx.pdf`

### 6.3 LLM 调用超时

**现象**: `summary` 或 `hypothetical` 策略耗时过长（>60s）。

**原因**: OpenRouter 免费模型限流（429）。

**处理**: 策略已内置 fallback，超时后自动使用本地兜底：
- 摘要: 取前100字作为摘要
- 问题: 使用通用模板（"这段文字的主要内容是什么？"）

```bash
# 查看策略是否有 fallback 日志
PYTHONPATH=backend/src python -m hybrid.cli index doc.pdf --verbose
```

### 6.4 索引状态不对

**现象**: `status` 显示 derivatives 为 0，但 indexed 有值。

**原因**: 使用了旧数据库或不同的 `--db` 路径。

**处理**:
```bash
# 确认数据库路径
PYTHONPATH=backend/src python -m hybrid.cli status --db backend/data/rag_data.db

# 重置后重新索引
PYTHONPATH=backend/src python -m hybrid.cli index doc.pdf --reset --db backend/data/rag_data.db
```

### 6.5 策略写入了不该写入的通道

**现象**: `status` 中发现 `hypothetical` 有 `graph` 索引记录。

**原因**: 使用旧代码（修复前）。

**验证**:
```bash
# 查看当前策略的 supported_channels
PYTHONPATH=backend/src python -c "from hybrid.registry import Registry; \
for n in Registry.list_strategies(): \
    print(f'{n}: {Registry.get(n).supported_channels}')"
```

**预期输出**:
```
standard: ['vector', 'fts', 'graph']
summary: ['vector', 'fts']
parent_child: ['vector', 'fts']
hypothetical: ['vector']
```

### 6.6 向量为 FakeEmbeddings（⚠️ 严重）

**现象**: `VectorChannel` 初始化时输出 `[VectorChannel] 使用 FakeEmbeddings（测试模式）`，索引后语义搜索失效。

**原因**: `OpenRouterEmbeddings` 初始化失败，静默 fallback 到 `FakeEmbeddings`。

**可能原因**:
1. `.env` 中 `EMBEDDING_PROVIDER` 或 `EMBEDDING_MODEL` 未配置
2. `.env` 中 `OPENROUTER_API_KEY` 缺失或过期
3. `VectorChannel` 初始化时 `.env` 文件尚未加载

**处理**:
```bash
# 1. 检查 .env 配置
cat backend/.env | grep -E "^(EMBEDDING|OPENROUTER)"

# 预期输出:
# EMBEDDING_PROVIDER=openrouter
# EMBEDDING_MODEL=nvidia/llama-nemotron-embed-vl-1b-v2:free
# OPENROUTER_API_KEY=sk-or-...

# 2. 验证配置生效
PYTHONPATH=backend/src python -c "
from hybrid.channels.vector import VectorChannel
vc = VectorChannel()
print(type(vc.embedding).__name__)
"
# 预期输出: OpenRouterEmbeddings

# 3. 如果仍为 FakeEmbeddings，重置后重新索引
PYTHONPATH=backend/src python -m hybrid.cli index docs/文档.pdf --reset
```

**FakeEmbeddings 回退机制说明**:
- 代码设计：当 `OpenRouterEmbeddings` 初始化失败时，自动回退到 `FakeEmbeddings`，保证系统可用
- 风险：FakeEmbeddings 基于 MD5 哈希生成，无真实语义，仅适合测试环境
- 生产环境：必须配置有效的 `OPENROUTER_API_KEY` 和嵌入模型

---

## 七、数据表结构速查

### 7.1 chunks（父块表）

| 字段 | 类型 | 说明 |
|------|------|------|
| chunk_id | INTEGER PK AUTOINCREMENT | 从 1000001 开始 |
| content | TEXT | 父块原文（~2000字） |
| source | TEXT | 来源文档名 |
| page | INTEGER | 页码 |
| chunk_index | INTEGER | 块序号 |

### 7.2 chunk_derivatives（派生表）

| 字段 | 类型 | 说明 |
|------|------|------|
| derivative_id | BIGINT PK | 14位纯数字: {type(1)}{chunk_id(10)}{seq(3)} |
| chunk_id | FK → chunks | 父块 ID |
| strategy | TEXT | standard/summary/parent_child/hypothetical |
| derivative_type | TEXT | summary/child/hyde/custom |
| content | TEXT | 派生内容 |
| metadata | TEXT (JSON) | 附加信息 |

### 7.3 index_status（索引状态表）

| 字段 | 类型 | 说明 |
|------|------|------|
| derivative_id | BIGINT | 派生 ID |
| strategy | TEXT | 所属策略 |
| channel | TEXT | vector/fts/graph |
| indexed | INTEGER | 0/1 |
| channel_doc_id | TEXT | 通道内部文档 ID |

---

## 八、CLI 完整参数说明

### index 命令

```bash
PYTHONPATH=backend/src python -m hybrid.cli index <pdf> [options]

参数:
  pdf                 PDF 文件路径（支持相对项目根目录或绝对路径）
  --db PATH           数据库路径 (默认: backend/data/rag_data.db)
  --strategies LIST   策略列表，逗号分隔 (默认: standard,summary,parent_child,hypothetical)
  --reset             重置后重新索引
  -v, --verbose       显示详细输出
```

### status 命令

```bash
PYTHONPATH=backend/src python -m hybrid.cli status [options]

参数:
  --db PATH           数据库路径 (默认: backend/data/rag_data.db)
```

### search 命令

```bash
PYTHONPATH=backend/src python -m hybrid.cli search <query> [options]

参数:
  query               查询字符串
  --db PATH           数据库路径 (默认: backend/data/rag_data.db)
  -k N                返回结果数 (默认: 3)
  --strategy NAME     指定策略检索 (默认: 全部策略)
```

---

## 九、变更日志

| 日期 | 版本 | 变更内容 |
|------|------|---------|
| 2026-06-22 | v2.2 | 新增嵌入模型配置检查（3.3节） |
| | | 新增索引后向量质量检查步骤（步骤5） |
| | | 新增 FakeEmbeddings 排查（6.6节） |
| | | CLI 输出示例增加嵌入模型信息 |
| | | 强调向量真实性对语义搜索的关键影响 |
| 2026-06-21 | v2.1 | 修复 CLI 路径解析，支持从项目根目录执行 |
| | | 修正 `PROJECT_ROOT` 和 `DEFAULT_DB` 路径计算 |
| | | 新增 `_resolve_pdf_path` 智能路径解析 |
| | | 优化错误提示，显示当前目录和项目根目录 |
| | | SOP 文档新增「环境激活」和「文件不存在」排查 |
| 2026-06-20 | v2.0 | 新增 CLI 工具 (`hybrid.cli`) |
| | | 用命令行替代手写 Python 脚本 |
| | | 支持 `index`/`status`/`search` 三个子命令 |
| | | 支持 `--strategies` 策略选择 |
| | | 支持 `--reset` 重置索引 |
