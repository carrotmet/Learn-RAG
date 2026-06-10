# RAG 教学项目

基于 LangGraph + ChromaDB + OpenRouter 的检索增强生成（RAG）教学演示项目。

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
├── backend/          # LangGraph + FastAPI 后端
│   ├── src/agent/    # RAG 工作流定义
│   ├── chroma_db/    # 向量数据库持久化
│   └── .env          # 环境变量配置
├── frontend/         # React + Vite 前端
│   └── src/App.tsx   # 主应用组件
├── docs/             # 文档与知识库
└── README.md         # 项目说明
```

## 配置说明

详见 `docs/RAG文档导入与向量化操作手册.md`

## 技术栈

- **后端**: LangGraph, FastAPI, ChromaDB, LangChain
- **前端**: React, Vite, TailwindCSS, @langchain/langgraph-sdk
- **模型**: OpenRouter (LLM + Embedding)

## License

MIT
