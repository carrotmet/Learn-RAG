from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
import traceback
import os
import time
import sys

# 确保 hybrid 模块可导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agent.state import RAGState
from agent.llm import OpenRouterLLM
from agent.vector_store import VectorStore

# HybridRAG 模块（第十章扩展）
from hybrid.document_store import DocumentStore
from hybrid.retrieval.enrich import Enrich
from hybrid.retrieval.intent import IntentRouter
from hybrid.retrieval.multi_recall import MultiRecall
from hybrid.retrieval.fusion import Fusion

# 全局实例（教学项目简化处理）
vector_store = VectorStore()

# ========== HybridRAG 实例 ==========
_db_path = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'data', 'rag_data.db'
)
_hybrid_store = DocumentStore(db_path=_db_path)
_hybrid_enrich = Enrich()
_hybrid_intent = IntentRouter()
_hybrid_recall = MultiRecall(
    _hybrid_store,
    vector=None,  # 懒加载
    fts=None,
    graph=None
)
_hybrid_fusion = Fusion(weights={"vector": 0.5, "fts": 0.3, "graph": 0.2})


def _extract_question(state: RAGState) -> str:
    """从 messages 中提取最后一条用户问题"""
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def _ensure_recaller():
    """懒加载通道（避免启动时 import 失败）"""
    if _hybrid_recall.vector is None:
        from hybrid.channels.vector import VectorChannel
        from hybrid.channels.fts import FTSChannel
        from hybrid.channels.graph import GraphChannel
        _hybrid_recall.vector = VectorChannel()
        _hybrid_recall.fts = FTSChannel()
        _hybrid_recall.graph = GraphChannel()


# ========== 一阶段节点（保留向后兼容）==========

def index_node(state: RAGState, config: RunnableConfig) -> RAGState:
    """索引节点：确保知识库有内容"""
    try:
        count = vector_store.db._collection.count()
    except Exception:
        count = 0
    
    if count == 0:
        from langchain_core.documents import Document
        docs = [
            Document(page_content="RAG（检索增强生成）是一种将外部知识检索与大型语言模型生成能力结合的技术。"),
            Document(page_content="LangGraph 是 LangChain 的扩展，用于构建有状态的多 Agent 工作流。"),
            Document(page_content="向量数据库通过将文本转换为高维向量来实现语义搜索。"),
            Document(page_content="OpenRouter 是一个统一的 AI 模型路由平台，支持访问多个提供商的模型。"),
        ]
        vector_store.db.add_documents(docs)
    
    return state

def retrieve_node(state: RAGState, config: RunnableConfig) -> RAGState:
    """检索节点：根据问题搜索相关文档（一阶段）"""
    question = _extract_question(state)
    retrieval_start = time.time()
    docs = vector_store.search(question, k=4)
    retrieval_latency = int((time.time() - retrieval_start) * 1000)
    
    return {
        "retrieved_docs": docs,
        "question": question,
        "retrieval_latency": retrieval_latency,
    }

def generate_node(state: RAGState, config: RunnableConfig) -> RAGState:
    """生成节点：基于检索结果生成答案"""
    question = state.get("question", "")
    docs = state.get("retrieved_docs", [])
    retrieval_latency = state.get("retrieval_latency", 0)
    
    llm = OpenRouterLLM()
    llm_start = time.time()
    
    try:
        if not docs or len(docs) == 0:
            prompt = f"请回答以下问题：{question}"
            system_msg = "你是一个通用的知识助手。当前知识库中没有相关文档，请基于你的训练知识回答问题。"
            answer = llm.generate(prompt, system=system_msg)
            contexts_for_collect = []
        else:
            context = "\n\n".join([doc.page_content for doc in docs])
            contexts_for_collect = [doc.page_content for doc in docs]
            prompt = f"""基于以下检索到的文档，回答用户问题。如果文档中没有相关信息，请基于你的知识回答。

--- 检索到的文档 ---
{context}

--- 用户问题 ---
{question}

请给出清晰、准确的回答："""
            system_msg = "你是一个专业的知识助手，优先基于提供的文档回答问题。"
            answer = llm.generate(prompt, system=system_msg)
        
        llm_latency = int((time.time() - llm_start) * 1000)
        
        # 在线采集
        _collect_conversation(question, answer, docs, retrieval_latency, llm_latency, prompt)
        
    except Exception as e:
        error_detail = traceback.format_exc()
        print(f"[LLM Error] {type(e).__name__}: {e}")
        print(f"[LLM Error Detail] {error_detail[:500]}")
        answer = f"抱歉，调用大模型时出错：{type(e).__name__}: {str(e)[:200]}。请检查模型配置或稍后重试。"
    
    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
    }


def _collect_conversation(question, answer, docs, retrieval_latency, llm_latency, prompt):
    """后台采集对话数据"""
    try:
        from data_collection.sqlite_store import SQLiteCollector
        collector = SQLiteCollector(db_path="data/rag_data.db")
        
        def _collect_data():
            conv_id = collector.save_conversation(
                question=question,
                answer=answer,
                contexts=[d.page_content for d in docs] if docs else [],
                model_version=os.getenv("DEFAULT_MODEL", "unknown"),
                source="online_api",
                metadata={
                    "retrieval_latency_ms": retrieval_latency,
                    "llm_latency_ms": llm_latency,
                    "retrieved_count": len(docs) if docs else 0,
                    "has_context": bool(docs),
                    "pipeline_version": "hybrid_v2"
                }
            )
            
            if docs:
                retrieved_chunks = []
                for doc in docs:
                    chunk = {
                        "content": doc.page_content[:500],
                        "source": doc.metadata.get("source", "unknown"),
                    }
                    retrieved_chunks.append(chunk)
                
                collector.save_retrieval_log(
                    conversation_id=conv_id,
                    query=question,
                    retrieved_chunks=retrieved_chunks,
                    latency_ms=retrieval_latency
                )
            
            collector.save_llm_call(
                conversation_id=conv_id,
                prompt=prompt[:2000],
                response=answer[:2000],
                model_name=os.getenv("DEFAULT_MODEL", "unknown"),
                token_usage={"prompt_tokens": len(prompt), "completion_tokens": len(answer)},
                latency_ms=llm_latency
            )
            
            return conv_id
        
        import threading
        collect_thread = threading.Thread(target=_collect_data, daemon=True)
        collect_thread.start()
        
    except Exception as collect_err:
        print(f"[在线采集] 保存失败（非阻塞）: {collect_err}")


# ========== 二阶段 HybridRAG 节点（第十章扩展）==========

def enrich_node(state: RAGState, config: RunnableConfig) -> RAGState:
    """第十章：问题完善节点
    
    检查问题完整度，不完整时改写或标记追问。
    """
    question = state.get("question", "") or _extract_question(state)
    
    try:
        result = _hybrid_enrich.check_completeness(question)
    except Exception as e:
        print(f"[enrich_node] 错误: {e}")
        # 回退：假设完整
        result = {"complete": True, "reason": "回退：检查失败"}
    
    return {
        "question": question,
        "enrich_complete": result.get("complete", True),
        "enrich_reason": result.get("reason", ""),
        "enrich_rewritten": result.get("rewritten_query", question),
        "enrich_follow_up": result.get("follow_up_question", ""),
    }


def enrich_router(state: RAGState) -> str:
    """第十章：Enrich 条件路由
    
    问题完整 → 走意图识别
    问题不完整 → 追问用户（结束）
    """
    if state.get("enrich_complete", True):
        return "intent"
    return "follow_up"


def follow_up_node(state: RAGState, config: RunnableConfig) -> RAGState:
    """第十章：追问节点
    
    返回追问提示，结束本次交互。
    """
    follow_up = state.get("enrich_follow_up", "")
    if not follow_up:
        follow_up = "您的问题似乎不完整，请补充更多信息。"
    
    return {
        "answer": follow_up,
        "messages": [AIMessage(content=follow_up)],
    }


def intent_node(state: RAGState, config: RunnableConfig) -> RAGState:
    """第十章：意图识别节点
    
    识别用户意图，生成检索配置。
    """
    question = state.get("enrich_rewritten", "") or state.get("question", "")
    
    try:
        result = _hybrid_intent.recognize(question)
    except Exception as e:
        print(f"[intent_node] 错误: {e}")
        # 回退：假设需要检索，使用通用配置
        result = {
            "intents": [{"path": "retrieve.keyword", "confidence": "medium"}],
            "primary_intent": "retrieve.keyword",
            "needs_retrieve": True
        }
    
    intents = result.get("intents", [])
    needs_retrieve = result.get("needs_retrieve", True)
    
    # 生成检索配置
    retrieve_config = None
    if needs_retrieve and intents:
        try:
            retrieve_config = IntentRouter.get_retrieve_config(intents)
        except Exception as e:
            print(f"[intent_node] 获取检索配置失败: {e}")
            retrieve_config = {
                "strategies": ["standard"],
                "weights": {"vector": 0.5, "fts": 0.3, "graph": 0.2},
                "mode": "hybrid"
            }
    
    return {
        "intents": intents,
        "primary_intent": result.get("primary_intent"),
        "needs_retrieve": needs_retrieve,
        "retrieve_strategies": retrieve_config.get("strategies", ["standard"]) if retrieve_config else ["standard"],
        "retrieve_weights": retrieve_config.get("weights", {}) if retrieve_config else {},
        "retrieve_mode": retrieve_config.get("mode", "hybrid") if retrieve_config else "hybrid",
    }


def intent_router(state: RAGState) -> str:
    """第十章：Intent 条件路由
    
    需要检索 → 走召回
    不需要检索 → 直接生成（闲聊/其他）
    """
    if state.get("needs_retrieve", True):
        return "recall"
    return "generate_direct"


def recall_node(state: RAGState, config: RunnableConfig) -> RAGState:
    """第十章：多路召回节点
    
    基于意图配置执行多路召回。
    """
    question = state.get("enrich_rewritten", "") or state.get("question", "")
    strategies = state.get("retrieve_strategies", ["standard"])
    mode = state.get("retrieve_mode", "hybrid")
    
    _ensure_recaller()
    
    try:
        recall_result = _hybrid_recall.recall(
            question,
            strategies=strategies,
            k=5
        )
    except Exception as e:
        print(f"[recall_node] 错误: {e}")
        traceback.print_exc()
        # 回退：空结果
        recall_result = {}
    
    return {
        "recall_results": recall_result,
    }


def fusion_node(state: RAGState, config: RunnableConfig) -> RAGState:
    """第十章：融合节点
    
    对召回结果进行 RRF 融合和去重。
    """
    recall_results = state.get("recall_results", {})
    weights = state.get("retrieve_weights", {"vector": 0.5, "fts": 0.3, "graph": 0.2})
    
    if not recall_results:
        return {"fused_results": [], "retrieved_docs": []}
    
    try:
        fusion = Fusion(weights=weights)
        fused = fusion.rrf(recall_results, top_k=5)
        
        # 转换为 LangChain Document 格式（兼容 generate）
        from langchain_core.documents import Document
        docs = []
        for item in fused:
            content = item.get("content", "")
            if content:
                docs.append(Document(
                    page_content=content,
                    metadata={
                        "parent_chunk_id": item.get("parent_chunk_id"),
                        "fusion_score": item.get("fusion_score"),
                        "hit_channels": item.get("hit_channels", []),
                        "source": item.get("source", "hybrid")
                    }
                ))
        
        return {
            "fused_results": fused,
            "retrieved_docs": docs,
        }
    except Exception as e:
        print(f"[fusion_node] 错误: {e}")
        return {"fused_results": [], "retrieved_docs": []}


def generate_direct_node(state: RAGState, config: RunnableConfig) -> RAGState:
    """第十章：直接生成节点（不需要检索时）"""
    question = state.get("enrich_rewritten", "") or state.get("question", "")
    
    llm = OpenRouterLLM()
    try:
        answer = llm.generate(
            f"请回答以下问题：{question}",
            system="你是一个通用的知识助手。"
        )
    except Exception as e:
        answer = f"抱歉，调用大模型时出错：{type(e).__name__}: {str(e)[:200]}。"
    
    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
    }


def hybrid_generate_node(state: RAGState, config: RunnableConfig) -> RAGState:
    """第十章：HybridRAG 生成节点
    
    基于融合结果生成答案。
    """
    question = state.get("enrich_rewritten", "") or state.get("question", "")
    docs = state.get("retrieved_docs", [])
    retrieval_latency = state.get("retrieval_latency", 0)
    
    llm = OpenRouterLLM()
    llm_start = time.time()
    
    try:
        if not docs or len(docs) == 0:
            prompt = f"请回答以下问题：{question}"
            system_msg = "你是一个通用的知识助手。当前知识库中没有相关文档，请基于你的训练知识回答问题。"
            answer = llm.generate(prompt, system=system_msg)
            contexts_for_collect = []
        else:
            context = "\n\n".join([doc.page_content for doc in docs])
            contexts_for_collect = [doc.page_content for doc in docs]
            prompt = f"""基于以下检索到的文档，回答用户问题。如果文档中没有相关信息，请基于你的知识回答。

--- 检索到的文档 ---
{context}

--- 用户问题 ---
{question}

请给出清晰、准确的回答："""
            system_msg = "你是一个专业的知识助手，优先基于提供的文档回答问题。"
            answer = llm.generate(prompt, system=system_msg)
        
        llm_latency = int((time.time() - llm_start) * 1000)
        
        # 在线采集
        _collect_conversation(question, answer, docs, retrieval_latency, llm_latency, prompt)
        
    except Exception as e:
        error_detail = traceback.format_exc()
        print(f"[LLM Error] {type(e).__name__}: {e}")
        print(f"[LLM Error Detail] {error_detail[:500]}")
        answer = f"抱歉，调用大模型时出错：{type(e).__name__}: {str(e)[:200]}。请检查模型配置或稍后重试。"
    
    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
    }


# ========== 构建图 ==========

builder = StateGraph(RAGState)

# 添加节点
# 一阶段节点（保留向后兼容，但默认不连接）
builder.add_node("index", index_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("generate", generate_node)

# 二阶段 HybridRAG 节点
builder.add_node("enrich", enrich_node)
builder.add_node("intent", intent_node)
builder.add_node("recall", recall_node)
builder.add_node("fusion", fusion_node)
builder.add_node("hybrid_generate", hybrid_generate_node)
builder.add_node("follow_up", follow_up_node)
builder.add_node("generate_direct", generate_direct_node)

# 默认链路：HybridRAG 二阶段
builder.add_edge(START, "enrich")

# Enrich 条件路由
builder.add_conditional_edges(
    "enrich",
    enrich_router,
    {
        "intent": "intent",
        "follow_up": "follow_up"
    }
)

# Intent 条件路由
builder.add_conditional_edges(
    "intent",
    intent_router,
    {
        "recall": "recall",
        "generate_direct": "generate_direct"
    }
)

# 后续链路
builder.add_edge("recall", "fusion")
builder.add_edge("fusion", "hybrid_generate")
builder.add_edge("hybrid_generate", END)
builder.add_edge("generate_direct", END)
builder.add_edge("follow_up", END)

# 编译图
graph = builder.compile(name="hybrid-rag-agent")
