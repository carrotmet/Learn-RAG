#!/bin/bash
# RAG教学项目前端启动脚本
# 用法: ./start_frontend.sh [port]
# 默认端口: 5173

set -e

PORT=${1:-5173}
LOG_FILE="/tmp/rag-frontend-${PORT}.log"
PID_FILE="/tmp/rag-frontend-${PORT}.pid"
PROJECT_DIR="/home/ubuntu/.openclaw/workspace/RAG教学/frontend"

echo "========================================"
echo "RAG 教学项目前端启动脚本"
echo "========================================"
echo "端口: ${PORT}"
echo "日志: ${LOG_FILE}"
echo "项目: ${PROJECT_DIR}"
echo "========================================"

# 检查是否已有进程在运行
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE" 2>/dev/null)
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "⚠️  检测到已有前端进程在运行 (PID: $OLD_PID)"
        echo "   如需重启，请先执行: ./stop_frontend.sh"
        exit 0
    else
        echo "🧹 清理残留 PID 文件"
        rm -f "$PID_FILE"
    fi
fi

# 检查端口是否被占用
if ss -tln | grep -q ":${PORT} "; then
    echo "❌ 端口 ${PORT} 已被占用"
    ss -tlnp | grep ":${PORT} "
    exit 1
fi

# 进入项目目录
cd "$PROJECT_DIR"

# 检查 node_modules
if [ ! -d "node_modules" ]; then
    echo "📦 安装前端依赖..."
    npm install > /tmp/rag-frontend-install.log 2>&1
fi

# 启动前端服务
echo "🚀 启动前端服务 (端口: ${PORT})..."
nohup npm run dev -- --host 0.0.0.0 --port "${PORT}" > "$LOG_FILE" 2>&1 &
PID=$!

# 保存 PID
echo $PID > "$PID_FILE"

# 等待启动
echo "⏳ 等待服务启动..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:${PORT} > /dev/null 2>&1; then
        echo "✅ 前端服务启动成功！"
        echo "   PID: $PID"
        echo "   端口: ${PORT}"
        echo "   访问: http://127.0.0.1:${PORT}"
        echo "   日志: ${LOG_FILE}"
        echo ""
        echo "停止服务: ./stop_frontend.sh ${PORT}"
        exit 0
    fi
    sleep 1
    echo "   等待中... (${i}/30)"
done

# 启动失败
echo "❌ 前端服务启动失败"
echo "日志内容:"
tail -50 "$LOG_FILE"
rm -f "$PID_FILE"
exit 1

