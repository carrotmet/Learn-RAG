#!/bin/bash
# 一键启动前后端（开发环境）

# 启动后端（后台）
cd backend
pip install -e .
langgraph dev &
BACKEND_PID=$!

# 等待后端启动
sleep 3

# 启动前端
cd ../frontend
npm install
npm run dev

# 清理后端进程
trap "kill $BACKEND_PID" EXIT
