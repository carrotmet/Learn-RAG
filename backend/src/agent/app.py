from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import pathlib
import os

# 使用绝对导入，避免路径问题
from src.agent.vector_store import VectorStore

app = FastAPI(docs_url=None, redoc_url=None)
vector_store = VectorStore()

# 挂载前端静态文件
build_path = pathlib.Path(__file__).parent.parent.parent.parent / "frontend" / "dist"
if build_path.is_dir() and (build_path / "index.html").is_file():
    app.mount("/app", StaticFiles(directory=build_path, html=True), name="frontend")

# 上传文档接口
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    upload_dir = os.getenv("UPLOAD_DIR", "./uploads")
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # 自动索引
    vector_store.index_file(file_path)
    
    return JSONResponse({"status": "ok", "filename": file.filename, "message": "文档已索引到知识库"})

# 获取知识库状态
@app.get("/api/status")
async def get_status():
    try:
        count = vector_store.db._collection.count()
    except Exception:
        count = 0
    return JSONResponse({"documents_count": count})
