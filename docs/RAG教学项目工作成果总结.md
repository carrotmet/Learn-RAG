# RAG 教学项目工作成果总结

> 版本：v1.0 | 归档日期：2026-06-12 | 适用场景：教学演示与类似项目落地参考

---

## 目录

1. [项目核心架构](#1-项目核心架构)
   - 1.1 [Naive RAG 三阶段](#11-naive-rag-三阶段)
   - 1.2 [数据库搭建](#12-数据库搭建)
   - 1.3 [前端搭建](#13-前端搭建)
   - 1.4 [数据交互](#14-数据交互)
2. [监控指标体系核心架构](#2-监控指标体系核心架构)
   - 2.1 [数据导入](#21-数据导入)
   - 2.2 [数据格式统一](#22-数据格式统一)
   - 2.3 [数据解析](#23-数据解析)
   - 2.4 [测试集搭建](#24-测试集搭建)
   - 2.5 [RAGAS 评估](#25-ragas-评估)
   - 2.6 [可视化输出](#26-可视化输出)
   - 2.7 [反馈闭环](#27-反馈闭环)
3. [核心代码与关键文件](#3-核心代码与关键文件)
   - 3.1 [Naive RAG 核心代码](#31-naive-rag-核心代码)
   - 3.2 [数据采集核心代码](#32-数据采集核心代码)
   - 3.3 [测试集搭建核心代码](#33-测试集搭建核心代码)
   - 3.4 [RAGAS 评估核心代码](#34-ragas-评估核心代码)
   - 3.5 [可视化报告核心代码](#35-可视化报告核心代码)
4. [主要功能与操作方法](#4-主要功能与操作方法)
   - 4.1 [API 接口速查](#41-api-接口速查)
   - 4.2 [CLI 命令速查](#42-cli-命令速查)
   - 4.3 [SOP 标准流程](#43-sop-标准流程)
5. [项目文件结构](#5-项目文件结构)
6. [附录：技能文件](#6-附录技能文件)

---

## 1. 项目核心架构

### 1.1 Naive RAG 三阶段

```
┌─────────────────────────────────────────────────────────────┐
│                     Naive RAG Pipeline                       │
│                                                              │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│   │  索引    │───▶│  检索    │───▶│  生成    │              │
│   │ Index    │    │ Retrieve│    │ Generate│              │
│   └──────────┘    └──────────┘    └──────────┘              │
│        │               │               │                     │
│        ▼               ▼               ▼                     │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│   │PDF/文本  │    │语义相似度 │    │LLM 生成  │              │
│   │文本提取  │    │Top-K 召回│    │多模型容错│              │
│   │分块嵌入  │    │向量数据库│    │带引用回答│              │
│   └──────────┘    └──────────┘    └──────────┘              │
│                                                              │
│   核心文件：vector_store.py  │  graph.py  │  llm.py          │
└─────────────────────────────────────────────────────────────┘
```

#### 1.1.1 索引（Index）

| 功能 | 实现 | 关键配置 |
|------|------|---------|
| 文本提取 | `pypdf` / `langchain` DocumentLoader | 支持 PDF/TXT/MD |
| 文本分块 | `RecursiveCharacterTextSplitter` | chunk_size=500, overlap=50 |
| 嵌入生成 | `OpenRouterEmbeddings`（自定义） | nvidia/llama-nemotron-embed-vl-1b-v2:free, 2048维 |
| 向量存储 | `ChromaDB`（本地持久化） | `./chroma_db` |

#### 1.1.2 检索（Retrieve）

| 功能 | 实现 | 关键配置 |
|------|------|---------|
| 查询嵌入 | 同索引模型 | 与索引模型一致 |
| 相似度计算 | 余弦相似度 | ChromaDB 内置 |
| Top-K 召回 | `k=4`（默认） | 可调：2-10 |
| 延迟记录 | 自动采集到 `retrieval_logs` | 毫秒级记录 |

#### 1.1.3 生成（Generate）

| 功能 | 实现 | 关键配置 |
|------|------|---------|
| LLM 调用 | `OpenRouterLLM`（自定义封装） | 多模型轮询容错 |
| 提示词模板 | 带上下文拼接 | 优先文档，次选知识 |
| 空检索降级 | 无结果时直接调用 LLM | 避免完全失败 |
| 生成延迟 | 自动采集到 `llm_calls` | 毫秒级记录 |

---

### 1.2 数据库搭建

```
┌─────────────────────────────────────────────────────────────┐
│                    SQLite 数据库架构                           │
│                                                              │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│   │conversations│  │retrieval_logs│  │  llm_calls  │        │
│   │  对话记录    │  │  检索日志    │  │ LLM调用记录 │        │
│   └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                              │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│   │user_feedback│  │  raw_data   │  │processed_data│        │
│   │  用户反馈    │  │  原始上传    │  │ 标准格式数据 │        │
│   └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                              │
│   ┌─────────────┐  ┌─────────────┐                        │
│   │testset_versions│ │evaluation_results│                    │
│   │ 测试集版本   │  │ 评估结果     │                        │
│   └─────────────┘  └─────────────┘                        │
│                                                              │
│   数据库：data/rag_data.db（生产）/ data/test_rag_data.db（测试）│
└─────────────────────────────────────────────────────────────┘
```

#### 1.2.1 核心表结构

| 表名 | 用途 | 关键字段 | 记录数（示例） |
|------|------|---------|-------------|
| `conversations` | 用户对话记录 | question, answer, contexts, ground_truth, model_version, timestamp | 50+ |
| `retrieval_logs` | 检索结果日志 | query, retrieved_chunks, scores, latency_ms | 50+ |
| `llm_calls` | LLM 调用记录 | prompt, response, model_name, token_usage, latency_ms | 50+ |
| `user_feedback` | 用户反馈 | feedback_type, content, rating | 动态 |
| `raw_data` | 离线上传原始数据 | source_type, original_format, raw_content | 10+ |
| `processed_data` | 解析后的标准格式 | question, question_type, domain, difficulty, processing_stage | 动态 |
| `testset_versions` | 测试集版本注册 | version_id, size, domain_distribution, file_paths | 1+ |
| `evaluation_results` | RAGAS 评估结果 | faithfulness, relevance, precision, ragas_score, passed | 动态 |

---

### 1.3 前端搭建

| 技术栈 | 用途 | 说明 |
|--------|------|------|
| **React + Vite** | 前端框架 | 轻量、快速热更新 |
| **LangServe SDK** | 后端通信 | 与 LangGraph 后端流式交互 |
| **静态挂载** | 部署 | FastAPI 挂载 `frontend/dist` 到 `/app` |

#### 1.3.1 前端核心功能

- **对话界面**：用户输入问题，流式展示回答
- **知识库状态**：显示当前文档数量
- **文件上传**：拖拽/选择文件上传并索引

---

### 1.4 数据交互

```
┌─────────────────────────────────────────────────────────────┐
│                     数据交互链路                               │
│                                                              │
│   用户 ──▶ 前端 ──▶ FastAPI ──▶ LangGraph ──▶ LLM          │
│              │        │           │                          │
│              │        │           ▼                          │
│              │        │     ┌──────────┐                      │
│              │        │     │ VectorStore │  ◀── ChromaDB    │
│              │        │     │  (检索)     │                   │
│              │        │     └──────────┘                      │
│              │        │                                      │
│              │        ▼                                      │
│              │   ┌──────────┐                                │
│              │   │ SQLite   │  ◀── 采集数据                 │
│              │   │ Collector│                                │
│              │   └──────────┘                                │
│              │                                              │
│              ▼                                              │
│         ┌──────────┐                                       │
│         │ 静态文件  │  ◀── 前端构建产物                      │
│         │ /app      │                                       │
│         └──────────┘                                       │
│                                                              │
│   端口：47569（后端 API + LangGraph）                       │
│   端口：5173（前端开发服务器）                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 监控指标体系核心架构

### 2.1 数据导入

```
┌─────────────────────────────────────────────────────────────┐
│                    数据导入链路（2.1-2.2）                    │
│                                                              │
│   上传文件 / API 请求                                         │
│      │                                                       │
│      ▼                                                       │
│   ┌─────────────┐                                            │
│   │ 格式解析     │  ◀── JSON/JSONL/CSV/Excel 统一解析         │
│   │ (Pandas)    │                                            │
│   └──────┬──────┘                                            │
│          │                                                    │
│          ▼                                                    │
│   ┌─────────────┐                                            │
│   │ 字段映射     │  ◀── 原始字段 → 标准字段                    │
│   │ (column map)│                                            │
│   └──────┬──────┘                                            │
│          │                                                    │
│          ▼                                                    │
│   ┌─────────────┐                                            │
│   │ 基础校验     │  ◀── 必填字段、格式、长度                    │
│   │ (Schema)    │                                            │
│   └──────┬──────┘                                            │
│          │                                                    │
│          ▼                                                    │
│   ┌─────────────┐                                            │
│   │ 写入 SQLite │  ◀── raw_data / conversations 表            │
│   │             │                                            │
│   └─────────────┘                                            │
│                                                              │
│   关键文件：uploader.py / sqlite_store.py                   │
└─────────────────────────────────────────────────────────────┘
```

#### 2.1.1 导入方式

| 方式 | 格式 | 场景 | 限制 |
|------|------|------|------|
| Web 上传 | JSON/JSONL/CSV/Excel | 小批量 | < 50MB |
| API 上传 | JSON/JSONL | 程序化导入 | 需认证 |
| CLI 工具 | JSON/JSONL/CSV | 开发调试 | 本地运行 |

---

### 2.2 数据格式统一

#### 2.2.1 标准数据格式（唯一标准）

```json
{
  "id": "uuid",
  "question": "用户原始问题",
  "question_type": "factual|comparative|procedural|open",
  "domain": "自指学|数学|AI|其他",
  "difficulty": "easy|medium|hard",
  "contexts": [
    {
      "content": "检索到的文本片段",
      "source": "文档路径或ID",
      "page": 1,
      "relevance_score": 0.95
    }
  ],
  "answer": "系统生成的回答",
  "ground_truth": "人工标注的标准答案（可选）",
  "metadata": {
    "timestamp": "2026-06-10T10:00:00Z",
    "model_version": "v1.2.3",
    "embedding_model": "nvidia/llama-nemotron-embed-vl-1b-v2:free",
    "chunk_size": 500,
    "retrieval_k": 4,
    "data_source": "offline_upload|online_api",
    "processing_stage": "raw|parsed|validated|cleaned|testset"
  },
  "evaluation": {
    "ragas_faithfulness": 0.85,
    "ragas_relevancy": 0.92,
    "human_rating": 4
  }
}
```

#### 2.2.2 字段验证规则

| 字段 | 类型 | 必填 | 验证规则 |
|------|------|------|---------|
| `id` | string | 是 | UUID 格式 |
| `question` | string | 是 | 长度 5-2000 字符 |
| `question_type` | enum | 是 | factual/comparative/procedural/open |
| `domain` | string | 是 | 预定义领域 |
| `difficulty` | enum | 是 | easy/medium/hard |
| `contexts` | array | 否 | 每项含 content 和 source |
| `processing_stage` | enum | 是 | raw/parsed/validated/cleaned/testset |

---

### 2.3 数据解析

```
┌─────────────────────────────────────────────────────────────┐
│                    数据解析流水线（2.3）                       │
│                                                              │
│   processed_data (stage=parsed)                               │
│      │                                                       │
│      ▼                                                       │
│   ┌─────────────┐                                            │
│   │ 内容解析     │  ◀── 清洗文本、提取实体                    │
│   │ (NLP clean) │                                            │
│   └──────┬──────┘                                            │
│          │                                                    │
│          ▼                                                    │
│   ┌─────────────┐                                            │
│   │ 语义分析     │  ◀── 意图识别、难度评估                    │
│   │ (semantic)  │                                            │
│   └──────┬──────┘                                            │
│          │                                                    │
│          ▼                                                    │
│   ┌─────────────┐                                            │
│   │ 质量评分     │  ◀── 完整性、可读性、相关性                │
│   │ (quality)   │     加权：问题(0.3) + 上下文(0.3) + 答案(0.2) + 标准答案(0.2) │
│   └──────┬──────┘                                            │
│          │                                                    │
│          ▼                                                    │
│   ┌─────────────┐                                            │
│   │ 标记 stage   │  ◀── validated / invalid                   │
│   │             │                                            │
│   └─────────────┘                                            │
│                                                              │
│   关键文件：testset_builder.py (DataParser)                  │
└─────────────────────────────────────────────────────────────┘
```

---

### 2.4 测试集搭建

```
┌─────────────────────────────────────────────────────────────┐
│                   测试集搭建流程（2.4）                        │
│                                                              │
│   validated 数据（高质量）                                    │
│      │                                                       │
│      ▼                                                       │
│   ┌─────────────┐                                            │
│   │ 质量筛选     │  ◀── quality_score > 0.7                  │
│   │ (filter)    │                                            │
│   └──────┬──────┘                                            │
│          │                                                    │
│          ▼                                                    │
│   ┌─────────────┐                                            │
│   │ 去重         │  ◀── Jaccard 相似度 < 0.9                  │
│   │ (dedup)     │                                            │
│   └──────┬──────┘                                            │
│          │                                                    │
│          ▼                                                    │
│   ┌─────────────┐                                            │
│   │ 分类标注     │  ◀── 领域/题型/难度                        │
│   │ (classify)  │                                            │
│   └──────┬──────┘                                            │
│          │                                                    │
│          ▼                                                    │
│   ┌─────────────┐                                            │
│   │ 分层采样     │  ◀── 按领域/难度均衡采样                    │
│   │ (stratify)  │                                            │
│   └──────┬──────┘                                            │
│          │                                                    │
│          ▼                                                    │
│   ┌─────────────────────────────────────────────┐            │
│   │ 生成三集：Golden │ Validation │ Stress       │            │
│   │                                              │            │
│   │ Golden：按质量排序取 Top-N                     │            │
│   │ Validation：均衡分层采样                       │            │
│   │ Stress：取边界条件（短问题/长问题/无上下文）      │            │
│   └─────────────────────────────────────────────┘            │
│                                                              │
│   关键文件：testset_builder.py (TestSetBuilder)              │
└─────────────────────────────────────────────────────────────┘
```

#### 2.4.1 测试集分层

| 数据集 | 规模 | 用途 | 建设方式 | 优先级 |
|--------|------|------|---------|--------|
| **Golden** | 50-100 条 | 核心功能回归测试 | 按质量排序取 Top | P0 |
| **Validation** | 200-500 条 | 日常批量评估 | 均衡分层采样 | P1 |
| **Stress** | 50-100 条 | 边界/异常场景 | 人工构造 adversarial | P1 |

---

### 2.5 RAGAS 评估

```
┌─────────────────────────────────────────────────────────────┐
│                   RAGAS 评估流程（4.1）                       │
│                                                              │
│   测试集 JSONL（Golden/Validation/Stress）                    │
│      │                                                       │
│      ▼                                                       │
│   ┌─────────────┐                                            │
│   │ 加载数据     │  ◀── 读取标准格式                          │
│   │ (load)      │                                            │
│   └──────┬──────┘                                            │
│          │                                                    │
│          ▼                                                    │
│   ┌─────────────┐                                            │
│   │ 逐条评估     │  ◀── 关键词重叠启发式算法                   │
│   │ (evaluate)  │     Faithfulness(0.4) + Relevance(0.3) + Precision(0.3) │
│   └──────┬──────┘                                            │
│          │                                                    │
│          ▼                                                    │
│   ┌─────────────┐                                            │
│   │ 存储结果     │  ◀── SQLite evaluation_results 表          │
│   │ (save)      │                                            │
│   └──────┬──────┘                                            │
│          │                                                    │
│          ▼                                                    │
│   ┌─────────────┐                                            │
│   │ 判断阈值     │  ◀── RAGAS Score > 0.75?                   │
│   │ (threshold) │     通过 / 失败                             │
│   └──────┬──────┘                                            │
│          │                                                    │
│          ▼                                                    │
│   ┌─────────────┐                                            │
│   │ 生成反馈     │  ◀── 报告/告警/优化建议                    │
│   │ (feedback)  │                                            │
│   └─────────────┘                                            │
│                                                              │
│   关键文件：ragas_eval.py (RAGASEvaluator)                   │
└─────────────────────────────────────────────────────────────┘
```

#### 2.5.1 评估指标

| 指标 | 权重 | 阈值 | 说明 | 计算方式 |
|------|------|------|------|---------|
| **Faithfulness** | 40% | > 0.8 | 回答是否忠实于上下文 | 回答关键词与上下文关键词重叠率 |
| **Answer Relevance** | 30% | > 0.8 | 回答与问题的相关度 | 回答关键词与问题关键词重叠率 |
| **Context Precision** | 30% | > 0.7 | 检索上下文的相关性 | 上下文关键词与问题关键词重叠率 |
| **RAGAS Score** | 综合 | > 0.75 | 加权平均综合得分 | 0.4×F + 0.3×R + 0.3×P |

---

### 2.6 可视化输出

```
┌─────────────────────────────────────────────────────────────┐
│                  可视化报告架构（4.2）                          │
│                                                              │
│   SQLite evaluation_results                                   │
│      │                                                       │
│      ▼                                                       │
│   ┌─────────────┐                                            │
│   │ 读取数据     │  ◀── 按 testset_version 聚合               │
│   │ (query)     │                                            │
│   └──────┬──────┘                                            │
│          │                                                    │
│          ▼                                                    │
│   ┌─────────────┐                                            │
│   │ 指标看板     │  ◀── 4 个核心指标卡片（当前值/目标值）      │
│   │ (dashboard) │                                            │
│   └──────┬──────┘                                            │
│          │                                                    │
│          ▼                                                    │
│   ┌─────────────┐                                            │
│   │ 分布图表     │  ◀── Chart.js 柱状图（RAGAS Score 分布）    │
│   │ (chart)     │     Chart.js 雷达图（指标对比 vs 目标）     │
│   └──────┬──────┘                                            │
│          │                                                    │
│          ▼                                                    │
│   ┌─────────────┐                                            │
│   │ 低分样本     │  ◀── 失败样本表格（问题/得分/优化建议）     │
│   │ (failures)  │                                            │
│   └──────┬──────┘                                            │
│          │                                                    │
│          ▼                                                    │
│   ┌─────────────┐                                            │
│   │ 导出 HTML    │  ◀── 完整报告 / 简化报告                    │
│   │ (report)    │     reports/evaluation_*.html              │
│   └─────────────┘                                            │
│                                                              │
│   关键文件：visualizer.py (EvaluationVisualizer)             │
└─────────────────────────────────────────────────────────────┘
```

#### 2.6.1 报告类型

| 类型 | 文件 | 内容 | 大小（示例） |
|------|------|------|-------------|
| **完整报告** | `reports/evaluation_*.html` | 看板 + 图表 + 低分样本 + 趋势 | 14-18 KB |
| **简化报告** | `reports/evaluation_mini.html` | 5 个核心指标卡片 | ~8 KB |

---

### 2.7 反馈闭环

```
┌─────────────────────────────────────────────────────────────┐
│                   测试反馈闭环（4.3-4.4）                      │
│                                                              │
│   ┌─────────────────────────────────────────────┐          │
│   │           RAGAS 评估结果                      │          │
│   │    (SQLite evaluation_results)                │          │
│   └────────────────────┬────────────────────────────┘          │
│                        │                                      │
│                        ▼                                      │
│   ┌─────────────────────────────────────────────┐          │
│   │  1. 阈值判断                                  │          │
│   │     RAGAS Score < 0.75?                       │          │
│   │     Faithfulness < 0.6? (检索问题)             │          │
│   │     Relevance < 0.6? (生成问题)                │          │
│   │     Precision < 0.6? (知识库问题)              │          │
│   └────────────────────┬────────────────────────────┘          │
│                        │                                      │
│             ┌─────────┴─────────┐                            │
│             │                   │                            │
│             ▼                   ▼                            │
│   ┌──────────────┐     ┌──────────────┐                      │
│   │   通过        │     │   失败        │                      │
│   │   (绿灯)      │     │   (红灯)      │                      │
│   └──────────────┘     └──────┬───────┘                      │
│                                │                            │
│                                ▼                            │
│   ┌─────────────────────────────────────────────┐          │
│   │  2. 根因分析                                  │          │
│   │     ├─ Faithfulness 低 → 检索/生成问题        │          │
│   │     ├─ Relevance 低   → 提示词/模型问题       │          │
│   │     └─ Precision 低   → 知识库覆盖问题        │          │
│   └────────────────────┬────────────────────────────┘          │
│                        │                                      │
│                        ▼                                      │
│   ┌─────────────────────────────────────────────┐          │
│   │  3. 生成优化建议                               │          │
│   │     ├─ 更新知识库文档                         │          │
│   │     ├─ 优化提示词模板                         │          │
│   │     └─ 调整模型参数                           │          │
│   └────────────────────┬────────────────────────────┘          │
│                        │                                      │
│                        ▼                                      │
│   ┌─────────────────────────────────────────────┐          │
│   │  4. 重新评估                                   │          │
│   │     (验证优化效果)                             │          │
│   └─────────────────────────────────────────────┘          │
│                                                              │
│   触发条件：连续下降 / 领域通过率 < 60% / 用户负面反馈 > 20%    │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 核心代码与关键文件

### 3.1 Naive RAG 核心代码

#### 3.1.1 索引（vector_store.py）

```python
# 核心类：VectorStore
# 关键方法：index_file() / search()

class VectorStore:
    def __init__(self, persist_dir=None):
        self.persist_dir = persist_dir or os.getenv("CHROMA_DB_PATH", "./chroma_db")
        
        # 嵌入模型选择（OpenRouter / HuggingFace / OpenAI / Fake）
        provider = os.getenv("EMBEDDING_PROVIDER", "fake")
        if provider == "openrouter":
            self.embedding = OpenRouterEmbeddings(
                model=os.getenv("EMBEDDING_MODEL"),
                api_key=os.getenv("OPENROUTER_API_KEY")
            )
        # ... 其他 provider
        
        # ChromaDB 向量存储
        self.db = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embedding
        )
    
    def index_file(self, file_path: str) -> int:
        """索引单个文件（PDF/TXT/MD）"""
        # 1. 文本提取
        docs = self._load_document(file_path)
        # 2. 分块
        chunks = self.splitter.split_documents(docs)
        # 3. 嵌入 + 存储
        self.db.add_documents(chunks)
        return len(chunks)
    
    def search(self, query: str, k: int = 4) -> list:
        """语义检索 Top-K"""
        return self.db.similarity_search(query, k=k)
```

#### 3.1.2 检索与生成（graph.py）

```python
# 核心：LangGraph 状态机（3 个节点）

from langgraph.graph import StateGraph, START, END

# 节点 1：检索
def retrieve_node(state, config):
    question = extract_last_human_message(state)
    retrieval_start = time.time()
    docs = vector_store.search(question, k=4)
    retrieval_latency = int((time.time() - retrieval_start) * 1000)
    return {
        "retrieved_docs": docs,
        "question": question,
        "retrieval_latency": retrieval_latency
    }

# 节点 2：生成
def generate_node(state, config):
    question = state["question"]
    docs = state["retrieved_docs"]
    llm = OpenRouterLLM()  # 多模型轮询容错
    
    if not docs:
        # 空检索降级：直接调用 LLM
        answer = llm.generate(f"请回答：{question}")
    else:
        # 基于上下文生成
        context = "\n\n".join([d.page_content for d in docs])
        prompt = f"基于以下文档回答：\n{context}\n\n问题：{question}"
        answer = llm.generate(prompt)
    
    # 在线采集（后台线程）
    asyncio.create_task(save_to_sqlite(question, answer, docs))
    
    return {"messages": [AIMessage(content=answer)]}

# 构建图
graph = StateGraph(RAGState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)
```

#### 3.1.3 LLM 封装（llm.py）

```python
# 核心：OpenRouterLLM（多模型轮询容错）

class OpenRouterLLM:
    def __init__(self):
        self.models = os.getenv("DEFAULT_MODEL").split(",")
        self.fallbacks = os.getenv("FALLBACK_MODELS", "").split(",")
    
    def generate(self, prompt: str, system: str = None) -> str:
        # 主模型 → 备选模型轮询
        for model in self.models + self.fallbacks:
            try:
                response = self._call_openrouter(prompt, model, system)
                return response
            except Exception as e:
                continue
        raise RuntimeError("所有模型均失败")
```

---

### 3.2 数据采集核心代码

#### 3.2.1 SQLite 存储层（sqlite_store.py）

```python
# 核心类：SQLiteCollector
# 关键方法：save_conversation() / save_feedback() / get_recent_conversations()

class SQLiteCollector:
    def __init__(self, db_path="data/rag_data.db"):
        self.db_path = db_path
        self._init_tables()  # 自动创建 7 张表
    
    def save_conversation(self, question, answer, contexts, 
                         ground_truth=None, model_version="v1.0.0",
                         source="online_api", metadata=None) -> str:
        """保存对话记录（同步，约 5ms）"""
        conv_id = str(uuid.uuid4())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO conversations 
                (id, question, answer, contexts, ground_truth, 
                 model_version, source, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (conv_id, question, answer, 
                  json.dumps(contexts), ground_truth,
                  model_version, source, json.dumps(metadata or {})))
            conn.commit()
        return conv_id
    
    def save_feedback(self, conversation_id, feedback_type, content=None) -> str:
        """保存用户反馈"""
        fb_id = str(uuid.uuid4())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO user_feedback 
                (id, conversation_id, feedback_type, content)
                VALUES (?, ?, ?, ?)
            """, (fb_id, conversation_id, feedback_type, content))
            conn.commit()
        return fb_id
```

#### 3.2.2 数据导出（exporter.py）

```python
# 核心类：DataExporter
# 关键方法：export_conversations() / export_testset() / export_raw()

class DataExporter:
    def __init__(self, db_path="data/rag_data.db"):
        self.db_path = db_path
    
    def export_conversations(self, output_path, conditions=None, limit=1000):
        """导出对话为 JSONL"""
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM conversations WHERE 1=1"
            if conditions:
                query += f" AND {conditions}"
            query += f" ORDER BY timestamp DESC LIMIT {limit}"
            rows = conn.execute(query).fetchall()
        
        with open(output_path, 'w') as f:
            for row in rows:
                f.write(json.dumps(dict(row), ensure_ascii=False) + '\n')
        
        return len(rows)
    
    def export_testset(self, output_path, testset_type="validation", 
                       domain=None, limit=500):
        """导出 RAGAS 格式测试集"""
        # ... 条件筛选 + 字段映射
```

---

### 3.3 测试集搭建核心代码

#### 3.3.1 数据导入（testset_builder.py - DataImporter）

```python
class DataImporter:
    def __init__(self, db_path="data/rag_data.db"):
        self.db_path = db_path
    
    def import_from_conversations(self, source="all", batch_size=100):
        """从 conversations 导入到 processed_data"""
        with sqlite3.connect(self.db_path) as conn:
            # 读取未处理的数据
            rows = conn.execute("""
                SELECT * FROM conversations 
                WHERE id NOT IN (SELECT raw_id FROM processed_data)
                LIMIT ?
            """, (batch_size,)).fetchall()
            
            for row in rows:
                # 字段映射：conversations → processed_data
                record = self._map_fields(dict(row))
                conn.execute("""
                    INSERT INTO processed_data 
                    (id, question, question_type, domain, difficulty,
                     contexts, answer, ground_truth, metadata, processing_stage)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'parsed')
                """, (record['id'], record['question'], ...))
            conn.commit()
        
        return len(rows)
    
    def _map_fields(self, raw):
        """字段映射 + 自动推断"""
        return {
            "id": str(uuid.uuid4()),
            "question": raw.get('question', ''),
            "question_type": self._infer_question_type(raw.get('question', '')),
            "domain": raw.get('domain', '其他'),
            "difficulty": raw.get('difficulty', 'medium'),
            "contexts": parse_json(raw.get('contexts', '[]')),
            "metadata": {"raw_id": raw.get('id'), ...}
        }
```

#### 3.3.2 数据解析（testset_builder.py - DataParser）

```python
class DataParser:
    def parse(self, record):
        """解析单条记录：清洗 → 验证 → 评分"""
        errors = []
        warnings = []
        
        # 1. 清洗
        question_clean = self._clean_text(record['question'])
        contexts_clean = [self._clean_text(c['content']) for c in record['contexts']]
        
        # 2. 验证
        if len(question_clean) < 5:
            errors.append("问题过短")
        
        # 3. 质量评分（0-1）
        quality_score = (
            min(len(question_clean) / 100, 0.3) +      # 问题质量
            min(len(contexts_clean) * 0.1, 0.3) +      # 上下文质量
            min(len(record.get('answer', '')) / 500, 0.2) +  # 答案质量
            (0.2 if record.get('ground_truth') else 0)  # 标准答案
        )
        
        is_valid = len(errors) == 0 and quality_score > 0.5
        
        return ParseResult(
            is_valid=is_valid,
            quality_score=quality_score,
            errors=errors,
            warnings=warnings
        )
```

#### 3.3.3 测试集构建（testset_builder.py - TestSetBuilder）

```python
class TestSetBuilder:
    def build_testset(self, output_prefix, golden_size=50, 
                      validation_size=200, stress_size=50):
        """构建分层测试集"""
        # 1. 加载高质量数据
        records = self._load_validated_records(min_quality=0.7)
        
        # 2. 去重（Jaccard 相似度）
        records = self._deduplicate(records, threshold=0.9)
        
        # 3. 分类（按领域）
        classified = self._classify_by_domain(records)
        
        # 4. 分层采样
        golden = self._sample_by_quality(classified, golden_size)
        validation = self._sample_stratified(classified, validation_size, exclude=golden)
        stress = self._sample_stress_cases(classified, stress_size)
        
        # 5. 导出 JSONL
        self._export_jsonl(golden, f"{output_prefix}_golden.jsonl")
        self._export_jsonl(validation, f"{output_prefix}_validation.jsonl")
        self._export_jsonl(stress, f"{output_prefix}_stress.jsonl")
        
        # 6. 注册版本
        self._register_version(golden, validation, stress)
```

---

### 3.4 RAGAS 评估核心代码

#### 3.4.1 评估器（ragas_eval.py）

```python
class RAGASEvaluator:
    def __init__(self, db_path="data/rag_data.db"):
        self.db_path = db_path
        self._ensure_results_table()  # 创建 evaluation_results 表
    
    def evaluate_testset(self, testset_path, testset_version, testset_type):
        """评估整个测试集"""
        # 1. 加载
        records = self._load_testset(testset_path)
        
        # 2. 逐条评估
        results = []
        for r in records:
            result = self.evaluate_single(
                r['question'], r['answer'], 
                r['contexts'], r.get('ground_truth', '')
            )
            results.append(result)
        
        # 3. 存储
        self._save_results(results, testset_version, testset_type, records)
        
        # 4. 汇总
        return {
            "total": len(results),
            "avg_faithfulness": mean(r['faithfulness'] for r in results),
            "avg_relevance": mean(r['answer_relevance'] for r in results),
            "avg_precision": mean(r['context_precision'] for r in results),
            "avg_ragas_score": mean(r['ragas_score'] for r in results),
            "passed": sum(1 for r in results if r['passed']),
            "pass_rate": mean(r['passed'] for r in results)
        }
    
    def evaluate_single(self, question, answer, contexts, ground_truth=""):
        """评估单条记录（关键词重叠启发式）"""
        # 提取关键词
        q_kw = self._extract_keywords(question)
        a_kw = self._extract_keywords(answer)
        c_kw = set()
        for ctx in contexts:
            c_kw.update(self._extract_keywords(ctx))
        
        # 计算指标
        faithfulness = self._overlap_ratio(a_kw, c_kw)  # 回答 vs 上下文
        relevance = self._overlap_ratio(a_kw, q_kw)      # 回答 vs 问题
        precision = self._overlap_ratio(c_kw, q_kw)      # 上下文 vs 问题
        
        # 综合得分
        ragas_score = 0.4 * faithfulness + 0.3 * relevance + 0.3 * precision
        passed = ragas_score > 0.75
        
        return {
            "faithfulness": faithfulness,
            "answer_relevance": relevance,
            "context_precision": precision,
            "ragas_score": ragas_score,
            "passed": passed
        }
```

---

### 3.5 可视化报告核心代码

#### 3.5.1 报告生成器（visualizer.py）

```python
class EvaluationVisualizer:
    def __init__(self, db_path="data/rag_data.db"):
        self.db_path = db_path
    
    def generate_report(self, testset_version, output_path, mini=False):
        """生成 HTML 评估报告"""
        # 获取数据
        summary = self._get_summary(testset_version)
        failures = self._get_failures(testset_version, limit=20)
        
        # 生成 HTML（含 Chart.js）
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                .metric {{ display: inline-block; margin: 20px; padding: 20px; 
                          border: 1px solid #ddd; border-radius: 8px; }}
                .pass {{ color: #28a745; }} .fail {{ color: #dc3545; }}
                .warn {{ color: #ffc107; }}
            </style>
        </head>
        <body>
            <h1>RAGAS 评估报告</h1>
            <p>版本: {testset_version}</p>
            
            <!-- 指标看板 -->
            <div class="metric">
                <div class="metric-value {'pass' if summary['pass_rate'] > 0.75 else 'fail'}">
                    {summary['avg_ragas_score']:.3f}
                </div>
                <div>RAGAS Score</div>
            </div>
            <!-- 更多指标... -->
            
            <!-- 图表 -->
            <canvas id="distributionChart"></canvas>
            <script>
                // Chart.js 柱状图：RAGAS Score 分布
                // Chart.js 雷达图：指标对比
            </script>
            
            <!-- 低分样本表 -->
            <table>
                <tr><th>Question</th><th>RAGAS</th><th>Faithfulness</th></tr>
                {''.join(f'<tr>...</tr>' for f in failures)}
            </table>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html)
        
        return output_path
```

---

## 4. 主要功能与操作方法

### 4.1 API 接口速查

| 接口 | 方法 | 说明 | 核心文件 |
|------|------|------|---------|
| `/api/upload` | POST | 上传文档并索引 | app.py |
| `/api/status` | GET | 知识库状态 | app.py |
| `/api/collect/conversation` | POST | 手动采集对话 | app.py |
| `/api/collect/feedback` | POST | 采集用户反馈 | app.py |
| `/api/collect/statistics` | GET | 采集统计 | app.py |
| `/api/collect/conversations` | GET | 最近对话 | app.py |
| `/api/export/conversations` | POST | 导出对话 | app.py |
| `/api/export/processed` | POST | 导出标准格式 | app.py |
| `/api/export/raw` | POST | 导出原始数据 | app.py |
| `/api/export/status` | GET | 导出文件状态 | app.py |
| `/api/testset/import` | POST | 3.2 数据导入 | app.py |
| `/api/testset/parse` | POST | 3.3 数据解析 | app.py |
| `/api/testset/build` | POST | 3.4 测试集搭建 | app.py |
| `/api/testset/versions` | GET | 测试集版本列表 | app.py |
| `/api/testset/pipeline` | GET | 完整流水线运行 | app.py |
| `/api/evaluate/testset` | POST | 4.1 RAGAS 评估测试集 | app.py |
| `/api/evaluate/single` | POST | 4.1 单条评估 | app.py |
| `/api/evaluate/summary` | GET | 4.1 评估汇总 | app.py |
| `/api/evaluate/failures` | GET | 4.1 低分样本 | app.py |
| `/api/evaluate/report` | POST | 4.2 生成可视化报告 | app.py |
| `/api/evaluate/all` | POST | 4.1-4.2 评估所有+报告 | app.py |
| `/runs/stream` | POST | LangGraph 流式对话 | graph.py |

### 4.2 CLI 命令速查

#### 4.2.1 环境启动

```bash
# 启动后端（LangGraph + FastAPI）
cd backend && source venv/bin/activate
langgraph dev --host 0.0.0.0 --port 47569

# 启动前端（开发模式）
cd frontend && npx vite --host 0.0.0.0 --port 5173
```

#### 4.2.2 文档索引

```bash
# 索引单个文件
python3 -c "
import sys; sys.path.insert(0, 'src')
from agent.vector_store import VectorStore
vs = VectorStore()
vs.index_file('docs/新文档.pdf')
"

# 索引整个目录
python3 -c "
from agent.vector_store import VectorStore
vs = VectorStore()
vs.index_directory('docs/')
"
```

#### 4.2.3 数据采集

```bash
# 手动采集对话
curl -X POST http://127.0.0.1:47569/api/collect/conversation \
  -d '{"question": "...", "answer": "...", "contexts": ["..."]}'

# 采集反馈
curl -X POST http://127.0.0.1:47569/api/collect/feedback \
  -d '{"conversation_id": "...", "feedback_type": "thumbs_up"}'

# 查看统计
curl -s http://127.0.0.1:47569/api/collect/statistics
```

#### 4.2.4 数据导出

```bash
# 导出对话
python -m data_collection.exporter --output data/export.jsonl --table conversations

# 导出测试集（RAGAS 格式）
python -m data_collection.exporter --output data/testset.jsonl --testset

# 预览数据
python -m data_collection.exporter --preview --limit 5
```

#### 4.2.5 测试集搭建

```bash
# 完整流水线（含虚拟数据生成）
python src/testset/test_pipeline.py

# 分步操作
python -m testset.testset_builder --action import --db data/rag_data.db
python -m testset.testset_builder --action parse --db data/rag_data.db
python -m testset.testset_builder --action build --db data/rag_data.db --output data/testset
```

#### 4.2.6 RAGAS 评估

```bash
# 评估单个测试集
python -m evaluation.ragas_eval --testset data/testset_golden.jsonl --version v1

# 生成报告
python -m feedback.visualizer --version v1 --output reports/evaluation.html
```

### 4.3 SOP 标准流程

#### SOP 1：新文档入库流程

```
1. 准备文档（PDF/TXT/MD）
   ↓
2. 调用上传接口
   curl -X POST /api/upload -F "file=@doc.pdf"
   ↓
3. 验证索引
   curl -s /api/status
   ↓
4. 测试检索
   curl -X POST /runs/stream -d '{"input": {"messages": [...]}}'
```

#### SOP 2：监控指标完整闭环流程

```
1. 数据导入（3.2）
   curl -X POST /api/testset/import -d '{"source": "all"}'
   ↓
2. 数据解析（3.3）
   curl -X POST /api/testset/parse -d '{"stage": "parsed"}'
   ↓
3. 测试集搭建（3.4）
   curl -X POST /api/testset/build -d '{"output_prefix": "data/testset"}'
   ↓
4. RAGAS 评估（4.1）
   curl -X POST /api/evaluate/testset -d '{"testset_path": "...", "testset_version": "v1"}'
   ↓
5. 生成报告（4.2）
   curl -X POST /api/evaluate/report -d '{"testset_version": "v1"}'
   ↓
6. 分析反馈（4.3-4.4）
   curl -s /api/evaluate/failures?testset_version=v1&limit=5
   ↓
7. 优化迭代
   更新知识库 / 优化提示词 / 调整模型
   ↓
8. 重新评估（验证优化效果）
   curl -X POST /api/evaluate/testset -d '{"testset_version": "v2"}'
```

#### SOP 3：一键评估所有测试集

```bash
# 评估 Golden + Validation + Stress，生成报告
curl -X POST http://127.0.0.1:47569/api/evaluate/all

# 查看汇总
curl -s "http://127.0.0.1:47569/api/evaluate/summary?testset_version=v1"
```

---

## 5. 项目文件结构

```
RAG教学/
├── backend/
│   ├── .env                        # 环境变量（LLM/嵌入模型/路径配置）
│   ├── src/
│   │   ├── agent/                  # Naive RAG 核心
│   │   │   ├── vector_store.py     # 向量数据库 + 嵌入模型（索引/检索）
│   │   │   ├── graph.py            # LangGraph 工作流（3节点：索引→检索→生成）
│   │   │   ├── app.py              # FastAPI 接口（21个 API 端点）
│   │   │   ├── llm.py              # OpenRouter LLM 封装（多模型轮询）
│   │   │   └── state.py            # RAGState 状态定义
│   │   ├── data_collection/        # 数据采集模块（2.1-2.4）
│   │   │   ├── sqlite_store.py     # SQLite 存储层（8张表：采集/日志/反馈/评估）
│   │   │   ├── uploader.py         # 离线上传解析器（JSON/JSONL/CSV/Excel）
│   │   │   ├── exporter.py         # 数据导出器（JSONL/RAGAS 格式）
│   │   │   └── config.py           # 采集配置
│   │   ├── testset/                # 测试集搭建模块（3.1-3.4）
│   │   │   ├── testset_builder.py  # 核心：导入/解析/搭建/版本管理（4类）
│   │   │   └── test_pipeline.py    # 测试脚本：虚拟数据生成 + 流程验证
│   │   ├── evaluation/             # RAGAS 评估模块（4.1）
│   │   │   └── ragas_eval.py       # RAGAS 评估器：指标计算 + 结果存储
│   │   └── feedback/               # 可视化报告模块（4.2）
│   │       └── visualizer.py       # HTML 报告生成：指标看板/图表/低分分析
│   ├── chroma_db/                  # 向量数据库持久化（ChromaDB）
│   └── data/                       # SQLite 数据库 + 导出文件 + 测试集 + 报告
│       ├── rag_data.db             # 生产数据库（8张表）
│       ├── test_rag_data.db          # 测试数据库（安全删除）
│       ├── testset_golden.jsonl    # 核心测试集
│       ├── testset_validation.jsonl # 验证集
│       ├── testset_stress.jsonl    # 压力测试集
│       └── reports/                # 评估报告 HTML
│           ├── evaluation_golden.html
│           ├── evaluation_validation.html
│           └── evaluation_stress.html
├── frontend/                       # React + Vite 前端
│   ├── src/App.tsx                 # 对话界面 + 上传组件
│   └── vite.config.ts              # 开发服务器配置
├── docs/                           # 项目文档
│   ├── RAG文档导入与向量化操作手册.md   # 操作手册（v1.1）
│   ├── 监控指标体系搭建规划.md          # 项目规划（v1.0）
│   ├── 数据库清单与作用说明.md          # 数据库说明
│   ├── RAG教学项目工作成果总结.md       # 本文档
│   └── *.pdf                       # 索引文档（自指学/对角线论证）
└── skills/rag-teaching-info/       # OpenClaw 技能文件
    └── SKILL.md                    # 项目技能（供 AI 读取）
```

---

## 6. 附录：技能文件

项目已制作 `skills/rag-teaching-info/SKILL.md` 技能文件，供 OpenClaw 自动读取，包含：

- 项目基本信息（目录、技术栈、环境）
- 接口速查表（21 个 API 端点）
- 文件定位表（关键文件路径与用途）
- 常用命令（启动/索引/评估/导出）
- 项目路径（数据/数据库/日志位置）
- 联系人（维护者信息）

**技能文件最后更新**：2026-06-12（新增 4.1-4.2 RAGAS 评估与可视化模块）

---

> **维护提示**：本总结为活文档，建议随项目迭代每季度更新。核心闭环跑通后，可逐步扩展：在线采集自动化、Prompt 版本管理、多模型 A/B 测试、用户反馈闭环触发机制。
>
> **教学建议**：本文档配合以下实践使用最佳：
> 1. 按 1.1-1.4 理解架构全貌
> 2. 按 2.1-2.7 理解监控体系设计思路
> 3. 按 3.1-3.5 阅读核心代码（最小化教学版本）
> 4. 按 4.1-4.3 动手操作 API 和 CLI
> 5. 按 SOP 2 跑通完整闭环
>
> **最后归档**：2026-06-12
