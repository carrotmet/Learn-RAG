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


# ========== 第十章~第十一章：HybridRAG API 接口 ==========

import json
from pydantic import BaseModel
from typing import Optional, List

# HybridRAG 模块导入（用于 API）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from hybrid.document_store import DocumentStore
from hybrid.retrieval.enrich import Enrich
from hybrid.retrieval.intent import IntentRouter
from hybrid.retrieval.multi_recall import MultiRecall
from hybrid.retrieval.fusion import Fusion
from hybrid.channels.vector import VectorChannel
from hybrid.channels.fts import FTSChannel
from hybrid.channels.graph import GraphChannel

# 全局 HybridRAG 实例（使用正式数据库）
_hybrid_db_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'rag_data.db')
_hybrid_store = DocumentStore(db_path=_hybrid_db_path)
_hybrid_enrich = Enrich()
_hybrid_intent = IntentRouter()
_hybrid_recall = MultiRecall(
    _hybrid_store,
    vector=VectorChannel(),
    fts=FTSChannel(),
    graph=GraphChannel()
)


class EnrichRequest(BaseModel):
    query: str


class EnrichResponse(BaseModel):
    status: str
    complete: bool
    reason: Optional[str] = None
    rewritten: Optional[str] = None
    follow_up: Optional[str] = None


@app.post("/api/hybrid/enrich", response_model=EnrichResponse)
async def api_hybrid_enrich(data: EnrichRequest):
    """第十一章：问题完善 API"""
    try:
        result = _hybrid_enrich.check_completeness(data.query)
        return EnrichResponse(
            status="ok",
            complete=result.get("complete", True),
            reason=result.get("reason"),
            rewritten=result.get("rewritten_query"),
            follow_up=result.get("follow_up_question")
        )
    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )


class IntentRequest(BaseModel):
    query: str


class IntentResponse(BaseModel):
    status: str
    intents: List[dict]
    primary_intent: Optional[str] = None
    needs_retrieve: bool = True
    retrieve_config: Optional[dict] = None


@app.post("/api/hybrid/intent", response_model=IntentResponse)
async def api_hybrid_intent(data: IntentRequest):
    """第十一章：意图识别 API"""
    try:
        result = _hybrid_intent.recognize(data.query)
        intents = result.get("intents", [])
        needs_retrieve = result.get("needs_retrieve", True)
        
        retrieve_config = None
        if needs_retrieve and intents:
            retrieve_config = IntentRouter.get_retrieve_config(intents)
        
        return IntentResponse(
            status="ok",
            intents=intents,
            primary_intent=result.get("primary_intent"),
            needs_retrieve=needs_retrieve,
            retrieve_config=retrieve_config
        )
    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )


class RecallRequest(BaseModel):
    query: str
    strategies: Optional[List[str]] = None
    channels: Optional[List[str]] = None
    mode: Optional[str] = "hybrid"
    k: Optional[int] = 5


class RecallResponse(BaseModel):
    status: str
    query: str
    recall_results: dict
    total_results: int


@app.post("/api/hybrid/recall", response_model=RecallResponse)
async def api_hybrid_recall(data: RecallRequest):
    """第十一章：多路召回调试 API"""
    try:
        strategies = data.strategies or ["standard"]
        channels = data.channels or ["vector", "fts", "graph"]
        
        results = _hybrid_recall.recall(
            data.query,
            strategies=strategies,
            channels=channels,
            
            k=data.k
        )
        
        total = sum(len(v) for v in results.values())
        
        return RecallResponse(
            status="ok",
            query=data.query,
            recall_results=results,
            total_results=total
        )
    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )


class HybridGenerateRequest(BaseModel):
    query: str
    strategies: Optional[List[str]] = None
    mode: Optional[str] = "hybrid"
    k: Optional[int] = 5


@app.post("/api/hybrid/generate")
async def api_hybrid_generate(data: HybridGenerateRequest):
    """第十一章：HybridRAG 端到端生成 API"""
    try:
        # 1. Enrich
        enrich_result = _hybrid_enrich.check_completeness(data.query)
        if not enrich_result.get("complete", True):
            return JSONResponse({
                "status": "incomplete",
                "follow_up": enrich_result.get("follow_up_question", "您的问题似乎不完整，请补充更多信息。"),
                "enrich": enrich_result
            })
        
        query = enrich_result.get("rewritten_query", data.query)
        
        # 2. Intent
        intent_result = _hybrid_intent.recognize(query)
        needs_retrieve = intent_result.get("needs_retrieve", True)
        
        # 3. 不需要检索 → 直接生成
        if not needs_retrieve:
            from agent.llm import OpenRouterLLM
            llm = OpenRouterLLM()
            answer = llm.generate(
                f"请回答以下问题：{query}",
                system="你是一个通用的知识助手。"
            )
            return JSONResponse({
                "status": "ok",
                "answer": answer,
                "pipeline": ["enrich", "intent", "generate_direct"],
                "intent": intent_result
            })
        
        # 4. Recall
        strategies = data.strategies or ["standard"]
        recall_results = _hybrid_recall.recall(
            query,
            strategies=strategies,
            
            k=data.k
        )
        
        # 5. Fusion
        fusion = Fusion(weights={"vector": 0.5, "fts": 0.3, "graph": 0.2})
        fused = fusion.rrf(recall_results, top_k=data.k)
        
        # 6. Generate
        from agent.llm import OpenRouterLLM
        llm = OpenRouterLLM()
        
        if not fused:
            answer = llm.generate(
                f"请回答以下问题：{query}",
                system="你是一个通用的知识助手。当前知识库中没有相关文档，请基于你的训练知识回答问题。"
            )
        else:
            context = "\n\n".join([item.get("content", "") for item in fused])
            prompt = f"""基于以下检索到的文档，回答用户问题。如果文档中没有相关信息，请基于你的知识回答。

--- 检索到的文档 ---
{context}

--- 用户问题 ---
{query}

请给出清晰、准确的回答："""
            answer = llm.generate(prompt, system="你是一个专业的知识助手，优先基于提供的文档回答问题。")
        
        return JSONResponse({
            "status": "ok",
            "answer": answer,
            "pipeline": ["enrich", "intent", "recall", "fusion", "generate"],
            "enrich": enrich_result,
            "intent": intent_result,
            "fused_count": len(fused),
            "top_sources": [item.get("source") for item in fused[:3]]
        })
        
    except Exception as e:
        import traceback
        return JSONResponse(
            {"status": "error", "message": str(e), "traceback": traceback.format_exc()},
            status_code=500
        )


# ========== LangGraph /runs/stream 流式接口（第十一章）==========

from fastapi.responses import StreamingResponse


class StreamRequest(BaseModel):
    query: str
    strategies: Optional[List[str]] = None
    mode: Optional[str] = "hybrid"


@app.post("/runs/stream")
async def runs_stream(data: StreamRequest):
    """第十一章：流式对话接口
    
    使用 LangGraph 执行 HybridRAG 链路，流式返回结果。
    兼容前端现有调用方式。
    """
    from agent.graph import graph
    from langchain_core.messages import HumanMessage
    
    async def event_generator():
        try:
            # 构建初始状态
            initial_state = {
                "messages": [HumanMessage(content=data.query)],
                "question": data.query,
                "enrich_complete": None,
                "enrich_reason": None,
                "enrich_rewritten": None,
                "enrich_follow_up": None,
                "intents": None,
                "primary_intent": None,
                "needs_retrieve": None,
                "retrieve_strategies": data.strategies or ["standard"],
                "retrieve_weights": None,
                "retrieve_mode": data.mode or "hybrid",
                "recall_results": None,
                "fused_results": None,
                "retrieved_docs": [],
                "answer": None,
                "retrieval_latency": 0
            }
            
            # 流式执行
            async for event in graph.astream_events(
                initial_state,
                version="v2",
                config={"recursion_limit": 50}
            ):
                kind = event.get("event")
                name = event.get("name", "")
                data_evt = event.get("data", {})
                
                # 发送节点开始/结束事件
                if kind in ("on_chain_start", "on_chain_end"):
                    if name in ("enrich", "intent", "recall", "fusion", "hybrid_generate"):
                        yield f"data: {json.dumps({'type': 'node', 'name': name, 'status': 'start' if kind == 'on_chain_start' else 'end'})}\n\n"
                
                # 发送生成结果
                if kind == "on_chain_end" and name == "hybrid_generate":
                    output = data_evt.get("output", {})
                    answer = output.get("answer", "")
                    if answer:
                        yield f"data: {json.dumps({'type': 'answer', 'content': answer})}\n\n"
            
            # 结束标记
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        except Exception as e:
            import traceback
            yield f"data: {json.dumps({'type': 'error', 'message': str(e), 'traceback': traceback.format_exc()})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

