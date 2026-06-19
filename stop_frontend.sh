#!/bin/bash
# RAG教学项目前端停止脚本
# 用法: ./stop_frontend.sh [port]
# 默认端口: 5173

PORT=${1:-5173}
PID_FILE="/tmp/rag-frontend-${PORT}.pid"

echo "========================================"
echo "RAG 教学项目前端停止脚本"
echo "端口: ${PORT}"
echo "========================================"

# 从 PID 文件停止
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE" 2>/dev/null)
    if [ -n "$PID" ] && ps -p "$PID" > /dev/null 2>&1; then
        echo "🛑 停止前端进程 (PID: $PID)..."
        kill "$PID" 2>/dev/null
        sleep 2
        
        # 强制终止
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "⚠️  进程未响应，强制终止..."
            kill -9 "$PID" 2>/dev/null
        fi
        
        rm -f "$PID_FILE"
        echo "✅ 前端服务已停止"
        exit 0
    else
        echo "🧹 清理残留 PID 文件"
        rm -f "$PID_FILE"
    fi
fi

# 通过端口查找进程
echo "🔍 通过端口查找进程..."
PID=$(ss -tlnp | grep ":${PORT} " | grep -oP 'pid=\K[0-9]+' | head -1)

if [ -n "$PID" ]; then
    echo "🛑 停止前端进程 (PID: $PID)..."
    kill "$PID" 2>/dev/null
    sleep 2
    
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "⚠️  进程未响应，强制终止..."
        kill -9 "$PID" 2>/dev/null
    fi
    
    echo "✅ 前端服务已停止"
else
    echo "ℹ️  未找到运行中的前端进程"
fi
