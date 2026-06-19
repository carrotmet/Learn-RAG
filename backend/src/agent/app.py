from fastapi import FastAPI, UploadFile, File, Request, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import pathlib
import os
import sys
import asyncio

# 使用绝对导入，避免路径问题
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.agent.vector_store import VectorStore
from src.data_collection.sqlite_store import SQLiteCollector

app = FastAPI(docs_url=None, redoc_url=None)
vector_store = VectorStore()

# 初始化采集器（在线采集核心）
collector = SQLiteCollector(db_path="data/rag_data.db")

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

# ========== 2.3 在线采集接口 ==========

@app.post("/api/collect/conversation")
async def collect_conversation(data: dict = Body(...)):
    """手动采集对话记录（用于测试或外部系统集成）"""
    try:
        # 使用线程执行 SQLite 写入避免阻塞事件循环
        def _save():
            return collector.save_conversation(
                question=data.get("question", ""),
                answer=data.get("answer"),
                contexts=data.get("contexts"),
                ground_truth=data.get("ground_truth"),
                model_version=data.get("model_version", "v1.0.0"),
                source=data.get("source", "online_api"),
                metadata=data.get("metadata")
            )
        
        conv_id = await asyncio.to_thread(_save)
        return JSONResponse({"status": "ok", "conversation_id": conv_id})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=400)

@app.post("/api/collect/feedback")
async def collect_feedback(data: dict = Body(...)):
    """采集用户反馈"""
    try:
        def _save():
            return collector.save_feedback(
                conversation_id=data.get("conversation_id"),
                feedback_type=data.get("feedback_type"),
                content=data.get("content"),
                rating=data.get("rating")
            )
        
        fb_id = await asyncio.to_thread(_save)
        return JSONResponse({"status": "ok", "feedback_id": fb_id})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=400)

@app.get("/api/collect/statistics")
async def get_collection_statistics():
    """获取采集统计信息"""
    try:
        def _get_stats():
            return collector.get_statistics()
        
        stats = await asyncio.to_thread(_get_stats)
        return JSONResponse({"status": "ok", "statistics": stats})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.get("/api/collect/conversations")
async def get_conversations(limit: int = 100, source: str = None):
    """获取最近对话记录"""
    try:
        def _get():
            return collector.get_recent_conversations(limit=limit, source=source)
        
        conversations = await asyncio.to_thread(_get)
        return JSONResponse({"status": "ok", "conversations": conversations})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

# ========== 2.4 数据导出接口 ==========

@app.post("/api/export/conversations")
async def export_conversations(data: dict = Body(...)):
    """导出对话数据到 JSONL"""
    try:
        output_path = data.get("output_path", "data/export_conversations.jsonl")
        conditions = data.get("conditions")
        limit = data.get("limit", 10000)
        
        def _export():
            return collector.export_to_jsonl(
                output_path=output_path,
                table="conversations",
                conditions=conditions,
                limit=limit
            )
        
        count = await asyncio.to_thread(_export)
        return JSONResponse({
            "status": "ok", 
            "exported_count": count,
            "output_path": output_path
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/api/export/processed")
async def export_processed(data: dict = Body(...)):
    """导出解析后的标准格式数据到 JSONL（测试集搭建用）"""
    try:
        output_path = data.get("output_path", "data/export_processed.jsonl")
        conditions = data.get("conditions")
        limit = data.get("limit", 10000)
        
        def _export():
            return collector.export_to_jsonl(
                output_path=output_path,
                table="processed_data",
                conditions=conditions,
                limit=limit
            )
        
        count = await asyncio.to_thread(_export)
        return JSONResponse({
            "status": "ok", 
            "exported_count": count,
            "output_path": output_path
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/api/export/raw")
async def export_raw(data: dict = Body(...)):
    """导出原始数据到 JSONL"""
    try:
        output_path = data.get("output_path", "data/export_raw.jsonl")
        conditions = data.get("conditions")
        limit = data.get("limit", 10000)
        
        def _export():
            return collector.export_to_jsonl(
                output_path=output_path,
                table="raw_data",
                conditions=conditions,
                limit=limit
            )
        
        count = await asyncio.to_thread(_export)
        return JSONResponse({
            "status": "ok", 
            "exported_count": count,
            "output_path": output_path
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.get("/api/export/status")
async def get_export_status():
    """获取导出目录状态"""
    try:
        data_dir = "data"
        exports = []
        if os.path.exists(data_dir):
            for f in os.listdir(data_dir):
                if f.startswith("export_") and f.endswith(".jsonl"):
                    file_path = os.path.join(data_dir, f)
                    exports.append({
                        "filename": f,
                        "path": file_path,
                        "size": os.path.getsize(file_path),
                        "modified": os.path.getmtime(file_path)
                    })
        return JSONResponse({"status": "ok", "exports": exports})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

# ========== 3.1-3.4 测试集搭建接口 ==========

from src.testset.testset_builder import DataImporter, DataParser, TestSetBuilder, TestSetRegistry

@app.post("/api/testset/import")
async def testset_import(data: dict = Body(...)):
    """3.2 离线数据导入：从 conversations/raw_data 导入到 processed_data"""
    try:
        source = data.get("source", "all")  # all, conversations, raw_data
        db_path = data.get("db_path", "data/rag_data.db")
        
        importer = DataImporter(db_path)
        
        if source == "raw_data":
            result = await asyncio.to_thread(importer.import_from_raw_data)
        else:
            result = await asyncio.to_thread(importer.import_from_conversations, None)
        
        return JSONResponse({"status": "ok", "result": result})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/api/testset/parse")
async def testset_parse(data: dict = Body(...)):
    """3.3 数据解析：清洗、验证、质量评分"""
    try:
        db_path = data.get("db_path", "data/rag_data.db")
        stage = data.get("stage", "parsed")
        
        parser = DataParser()
        result = await asyncio.to_thread(parser.parse_all, db_path, stage)
        
        return JSONResponse({"status": "ok", "result": result})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/api/testset/build")
async def testset_build(data: dict = Body(...)):
    """3.4 测试集搭建：分层采样、去重、导出"""
    try:
        db_path = data.get("db_path", "data/rag_data.db")
        output_prefix = data.get("output_prefix", "data/testset")
        golden_size = data.get("golden_size", 20)
        validation_size = data.get("validation_size", 50)
        stress_size = data.get("stress_size", 10)
        
        builder = TestSetBuilder(db_path)
        
        def _build():
            return builder.build_testset(
                output_prefix=output_prefix,
                golden_size=golden_size,
                validation_size=validation_size,
                stress_size=stress_size
            )
        
        result = await asyncio.to_thread(_build)
        
        return JSONResponse({"status": "ok", "result": result})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.get("/api/testset/versions")
async def testset_versions(limit: int = 10):
    """获取测试集版本列表"""
    try:
        registry = TestSetRegistry("data/rag_data.db")
        versions = await asyncio.to_thread(registry.list_versions, limit)
        return JSONResponse({"status": "ok", "versions": versions})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.get("/api/testset/pipeline")
async def testset_pipeline_run(
    db_path: str = "data/rag_data.db",
    output_prefix: str = "data/testset"
):
    """运行完整测试集搭建流水线（3.1-3.4）"""
    try:
        results = {}
        
        # 1. 导入
        importer = DataImporter(db_path)
        results["import"] = await asyncio.to_thread(importer.import_from_conversations, None)
        
        # 2. 解析
        parser = DataParser()
        results["parse"] = await asyncio.to_thread(parser.parse_all, db_path, "parsed")
        
        # 3. 搭建
        builder = TestSetBuilder(db_path)
        def _build():
            return builder.build_testset(output_prefix=output_prefix)
        results["build"] = await asyncio.to_thread(_build)
        
        return JSONResponse({"status": "ok", "pipeline": results})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

# ========== 4.1-4.2 RAGAS 评估与可视化接口 ==========

from src.evaluation.ragas_eval import RAGASEvaluator
from src.feedback.visualizer import EvaluationVisualizer

@app.post("/api/evaluate/testset")
async def evaluate_testset(data: dict = Body(...)):
    """4.1 RAGAS 评估：评估测试集"""
    try:
        testset_path = data.get("testset_path")
        testset_version = data.get("testset_version", "v1")
        testset_type = data.get("testset_type", "validation")
        db_path = data.get("db_path", "data/rag_data.db")
        
        if not testset_path:
            return JSONResponse({"status": "error", "message": "testset_path 必填"}, status_code=400)
        
        evaluator = RAGASEvaluator(db_path)
        
        def _eval():
            return evaluator.evaluate_testset(
                testset_path=testset_path,
                testset_version=testset_version,
                testset_type=testset_type
            )
        
        result = await asyncio.to_thread(_eval)
        return JSONResponse({"status": "ok", "evaluation": result})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/api/evaluate/single")
async def evaluate_single(data: dict = Body(...)):
    """4.1 单条评估：评估单个问答对"""
    try:
        question = data.get("question", "")
        answer = data.get("answer", "")
        contexts = data.get("contexts", [])
        ground_truth = data.get("ground_truth", "")
        db_path = data.get("db_path", "data/rag_data.db")
        
        evaluator = RAGASEvaluator(db_path)
        
        def _eval():
            return evaluator.evaluate_single(
                question=question,
                answer=answer,
                contexts=contexts,
                ground_truth=ground_truth
            )
        
        result = await asyncio.to_thread(_eval)
        return JSONResponse({"status": "ok", "evaluation": result})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.get("/api/evaluate/summary")
async def evaluate_summary(testset_version: str, db_path: str = "data/rag_data.db"):
    """4.1 获取评估汇总"""
    try:
        evaluator = RAGASEvaluator(db_path)
        
        def _get():
            return evaluator.get_summary(testset_version)
        
        result = await asyncio.to_thread(_get)
        return JSONResponse({"status": "ok", "summary": result})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.get("/api/evaluate/failures")
async def evaluate_failures(testset_version: str, limit: int = 20, db_path: str = "data/rag_data.db"):
    """4.1 获取低分样本"""
    try:
        evaluator = RAGASEvaluator(db_path)
        
        def _get():
            return evaluator.get_failures(testset_version, limit)
        
        result = await asyncio.to_thread(_get)
        return JSONResponse({"status": "ok", "failures": result})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/api/evaluate/report")
async def generate_report(data: dict = Body(...)):
    """4.2 生成可视化报告"""
    try:
        testset_version = data.get("testset_version")
        db_path = data.get("db_path", "data/rag_data.db")
        output_path = data.get("output_path", "reports/evaluation_report.html")
        mini = data.get("mini", False)
        
        if not testset_version:
            return JSONResponse({"status": "error", "message": "testset_version 必填"}, status_code=400)
        
        visualizer = EvaluationVisualizer(db_path)
        
        def _gen():
            if mini:
                return visualizer.generate_mini_report(testset_version, output_path)
            else:
                return visualizer.generate_report(testset_version, output_path)
        
        path = await asyncio.to_thread(_gen)
        
        return JSONResponse({
            "status": "ok",
            "report_path": path,
            "report_size": os.path.getsize(path)
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/api/evaluate/all")
async def evaluate_all_testsets(data: dict = Body(...)):
    """评估所有测试集并生成报告"""
    try:
        db_path = data.get("db_path", "data/rag_data.db")
        
        testsets = [
            ("data/testset_golden.jsonl", "golden"),
            ("data/testset_validation.jsonl", "validation"),
            ("data/testset_stress.jsonl", "stress"),
        ]
        
        evaluator = RAGASEvaluator(db_path)
        visualizer = EvaluationVisualizer(db_path)
        results = {}
        
        for path, ttype in testsets:
            if os.path.exists(path):
                # 评估
                def _eval():
                    return evaluator.evaluate_testset(path, f"auto_{ttype}", ttype)
                eval_result = await asyncio.to_thread(_eval)
                results[ttype] = eval_result
                
                # 生成报告
                report_path = f"reports/evaluation_{ttype}.html"
                def _gen():
                    return visualizer.generate_report(f"auto_{ttype}", report_path)
                await asyncio.to_thread(_gen)
        
        return JSONResponse({"status": "ok", "results": results})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
