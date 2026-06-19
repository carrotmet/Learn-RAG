# RAG 教学项目 — 启动脚本使用说明

## 当前状态

✅ **后端服务已运行**（PID: 3348153，端口: **47569**）
- 健康检查: http://127.0.0.1:47569/ok ✅
- 助手列表: http://127.0.0.1:47569/assistants/search ✅
- 日志: /tmp/rag-backend-47569.log

## 服务端口

| 服务 | 端口 | 状态 |
|------|------|------|
| 后端 API | **47569** | ✅ 运行中（主端口） |
| 前端 DevServer | 5173 | ⬜ 未启动 |

> ⚠️ 注意：端口 2024 被其他项目占用，非本服务端口。本服务使用 **47569**。

## 启动脚本

### 后端启动
```bash
cd /home/ubuntu/.openclaw/workspace/RAG教学
./start_backend.sh 47569  # 必须使用端口 47569
```

### 后端停止
```bash
cd /home/ubuntu/.openclaw/workspace/RAG教学
./stop_backend.sh 47569  # 停止端口 47569
```

### 前端启动
```bash
cd /home/ubuntu/.openclaw/workspace/RAG教学
./start_frontend.sh [port]  # 默认端口 5173
```

### 前端停止
```bash
cd /home/ubuntu/.openclaw/workspace/RAG教学
./stop_frontend.sh [port]  # 默认端口 5173
```

## 持久化配置

已配置 **每 5 分钟** 自动心跳检查：
- 检查端口 **47569** 是否被 langgraph 进程监听
- 如果服务未运行，自动重新启动
- 记录检查结果到 `memory/heartbeat-rag.json`

## 手动检查

```bash
# 健康检查（端口 47569）
curl http://127.0.0.1:47569/ok

# 查看后端日志
tail -f /tmp/rag-backend-47569.log

# 查看进程
ps aux | grep "langgraph.*47569"
```
