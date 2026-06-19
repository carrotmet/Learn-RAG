# RAG 教学项目 — 数据采集模块

## 模块说明

数据采集模块负责 RAG 教学项目的数据收集与存储，采用轻量 SQLite 方案。

## 核心组件

| 文件 | 说明 |
|------|------|
| `config.py` | 采集配置 |
| `sqlite_store.py` | SQLite 存储层（核心） |
| `uploader.py` | 离线上传解析器 |
| `demo_data.py` | 虚拟数据生成与测试 |

## 数据表结构

- **conversations**: 用户对话记录（在线采集主表）
- **retrieval_logs**: 检索结果日志
- **llm_calls**: LLM 调用记录
- **user_feedback**: 用户反馈
- **raw_data**: 离线上传原始数据
- **processed_data**: 解析后的标准格式数据

## 使用方式

```python
from data_collection.sqlite_store import SQLiteCollector
from data_collection.config import CollectionConfig

# 初始化
config = CollectionConfig(db_path="data/rag_data.db")
collector = SQLiteCollector(config.db_path)

# 保存对话
collector.save_conversation(
    question="什么是RAG？",
    answer="RAG是检索增强生成...",
    contexts=["RAG是一种技术..."],
    ground_truth="检索增强生成（RAG）..."
)
```

## 测试

```bash
python backend/src/data_collection/demo_data.py
```
