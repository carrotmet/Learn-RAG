# RAG 教学项目

基于 LangGraph + ChromaDB + OpenRouter 的检索增强生成（RAG）教学演示项目。

**当前状态**: HybridRAG 基本框架已完成（二阶段 1-7 章）
- ✅ 多策略索引（Standard / Summary / Parent-Child / Hypothetical）
- ✅ 多通道检索（Vector / FTS / Graph）
- ✅ 意图识别（两层架构 + 多选混合）
- ✅ 多路召回 + RRF 融合
- ✅ 问题完善（Enrich）

## 快速开始

### 环境要求

- Python 3.11+
- Node.js 18+
- OpenRouter API Key

### 启动后端

```bash
cd backend
pip install -e .
langgraph dev --host 0.0.0.0 --port 47569
```

### 启动前端

```bash
cd frontend
npm install
npm run dev
```

### 访问

前端：`http://localhost:5173/app/`

## 项目结构

```
.
├── backend/              # LangGraph + FastAPI 后端
│   ├── src/
│   │   ├── agent/        # RAG 工作流定义（一阶段）
│   │   ├── hybrid/       # HybridRAG 核心模块（二阶段）
│   │   │   ├── channels/     # 三通道：Vector / FTS / Graph
│   │   │   ├── strategies/   # 四策略：Standard / Summary / Parent-Child / Hypothetical
│   │   │   ├── retrieval/    # 意图识别 + 多路召回 + 融合
│   │   │   ├── document_store.py  # 统一 Chunk 存储层
│   │   │   ├── registry.py        # 策略注册中心
│   │   │   └── cli.py             # CLI 工具（index / status / search）
│   │   └── data_collection/  # 数据采集层
│   ├── data/           # SQLite + ChromaDB + Graph 数据
│   └── .env            # 环境变量配置
├── frontend/           # React + Vite 前端
│   └── src/App.tsx     # 主应用组件
├── docs/               # 文档与知识库
│   └── SOP_HybridRAG文档上传与处理.md
└── README.md           # 项目说明
```

## 配置说明

详见 `docs/RAG文档导入与向量化操作手册.md`

## 技术栈

- **后端**: LangGraph, FastAPI, ChromaDB, LangChain, SQLite(FTS5), NetworkX
- **前端**: React, Vite, TailwindCSS, @langchain/langgraph-sdk
- **模型**: OpenRouter (LLM + Embedding)

## License

MIT
