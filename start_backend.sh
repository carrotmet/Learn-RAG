#!/bin/bash
# RAG教学项目后端启动脚本
# 用法: ./start_backend.sh [port]
# 默认端口: 47569（RAG教学项目主端口）

set -e

PORT=${1:-47569}
LOG_FILE="/tmp/rag-backend-${PORT}.log"
PID_FILE="/tmp/rag-backend-${PORT}.pid"
PROJECT_DIR="/home/ubuntu/.openclaw/workspace/RAG教学/backend"

echo "========================================"
echo "RAG 教学项目后端启动脚本"
echo "========================================"
echo "端口: ${PORT}"
echo "日志: ${LOG_FILE}"
echo "项目: ${PROJECT_DIR}"
echo "========================================"

# 检查是否已有进程在运行
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE" 2>/dev/null)
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "⚠️  检测到已有后端进程在运行 (PID: $OLD_PID)"
        echo "   端口: $(ss -tlnp | grep ":${PORT} " | awk '{print $4}')"
        echo "   如需重启，请先执行: ./stop_backend.sh ${PORT}"
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

# 激活虚拟环境
if [ -f "venv/bin/activate" ]; then
    echo "✅ 激活虚拟环境"
    source venv/bin/activate
else
    echo "❌ 虚拟环境不存在: venv/bin/activate"
    exit 1
fi

# 检查依赖
echo "🔍 检查关键依赖..."
python3 -c "import langgraph; import fastapi; import chromadb" 2>/dev/null || {
    echo "⚠️  依赖缺失，尝试安装..."
    pip install -e . > /tmp/rag-backend-install.log 2>&1
}

# 启动后端服务
echo "🚀 启动后端服务 (端口: ${PORT})..."
nohup langgraph dev --host 0.0.0.0 --port "${PORT}" > "$LOG_FILE" 2>&1 &
PID=$!

# 保存 PID
echo $PID > "$PID_FILE"

# 等待启动
echo "⏳ 等待服务启动..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:${PORT}/ok > /dev/null 2>&1; then
        echo "✅ 后端服务启动成功！"
        echo "   PID: $PID"
        echo "   端口: ${PORT}"
        echo "   健康检查: http://127.0.0.1:${PORT}/ok"
        echo "   日志: ${LOG_FILE}"
        echo ""
        echo "停止服务: ./stop_backend.sh ${PORT}"
        exit 0
    fi
    sleep 1
    echo "   等待中... (${i}/30)"
done

# 启动失败
echo "❌ 后端服务启动失败"
echo "日志内容:"
tail -50 "$LOG_FILE"
rm -f "$PID_FILE"
exit 1
