from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
import traceback
import os
import time
import asyncio

from agent.state import RAGState
from agent.llm import OpenRouterLLM
from agent.vector_store import VectorStore

# 全局向量数据库实例（教学项目简化处理）
vector_store = VectorStore()

# 节点 1：索引（仅在知识库为空时自动初始化示例文档）
def index_node(state: RAGState, config: RunnableConfig) -> RAGState:
    """索引节点：确保知识库有内容"""
    # 教学演示：如果知识库为空，自动创建示例文档
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

# 节点 2：检索
def retrieve_node(state: RAGState, config: RunnableConfig) -> RAGState:
    """检索节点：根据问题搜索相关文档"""
    # 从 messages 中提取最后一条用户问题
    messages = state.get("messages", [])
    question = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            question = msg.content
            break
    
    # 记录检索开始时间
    retrieval_start = time.time()
    docs = vector_store.search(question, k=4)
    retrieval_latency = int((time.time() - retrieval_start) * 1000)
    
    return {
        "retrieved_docs": docs,
        "question": question,
        "retrieval_latency": retrieval_latency,
    }

# 节点 3：生成（带多模型轮询容错：检索失败或无结果时直接调用 LLM）
def generate_node(state: RAGState, config: RunnableConfig) -> RAGState:
    """生成节点：基于检索结果生成答案，支持多模型自动轮询"""
    question = state.get("question", "")
    docs = state.get("retrieved_docs", [])
    retrieval_latency = state.get("retrieval_latency", 0)
    
    # 调用 LLM（带多模型轮询容错）
    llm = OpenRouterLLM()
    
    # 记录 LLM 调用开始时间
    llm_start = time.time()
    
    try:
        if not docs or len(docs) == 0:
            # 知识库无结果，直接调用 LLM 回答
            prompt = f"请回答以下问题：{question}"
            system_msg = "你是一个通用的知识助手。当前知识库中没有相关文档，请基于你的训练知识回答问题。"
            answer = llm.generate(prompt, system=system_msg)
            contexts_for_collect = []
        else:
            # 有检索结果，基于上下文生成
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
        
        # ========== 2.3 在线采集：保存对话记录 ==========
        try:
            # 延迟导入避免循环依赖
            from data_collection.sqlite_store import SQLiteCollector
            collector = SQLiteCollector(db_path="data/rag_data.db")
            
            # 在线采集使用线程避免阻塞
            def _collect_data():
                # 保存对话
                conv_id = collector.save_conversation(
                    question=question,
                    answer=answer,
                    contexts=contexts_for_collect,
                    model_version=os.getenv("DEFAULT_MODEL", "unknown"),
                    source="online_api",
                    metadata={
                        "retrieval_latency_ms": retrieval_latency,
                        "llm_latency_ms": llm_latency,
                        "retrieved_count": len(docs),
                        "has_context": len(docs) > 0
                    }
                )
                
                # 保存检索日志
                if docs:
                    retrieved_chunks = []
                    for doc in docs:
                        chunk = {
                            "content": doc.page_content[:500],  # 截断避免过大
                            "source": doc.metadata.get("source", "unknown"),
                        }
                        retrieved_chunks.append(chunk)
                    
                    collector.save_retrieval_log(
                        conversation_id=conv_id,
                        query=question,
                        retrieved_chunks=retrieved_chunks,
                        latency_ms=retrieval_latency
                    )
                
                # 保存 LLM 调用记录
                collector.save_llm_call(
                    conversation_id=conv_id,
                    prompt=prompt[:2000],  # 截断避免过大
                    response=answer[:2000],
                    model_name=os.getenv("DEFAULT_MODEL", "unknown"),
                    token_usage={"prompt_tokens": len(prompt), "completion_tokens": len(answer)},
                    latency_ms=llm_latency
                )
                
                return conv_id
            
            # 使用线程执行采集（非阻塞）
            import threading
            collect_thread = threading.Thread(target=_collect_data, daemon=True)
            collect_thread.start()
            
            print(f"[在线采集] 已触发后台采集")
            
        except Exception as collect_err:
            # 采集失败不影响主流程
            print(f"[在线采集] 保存失败（非阻塞）: {collect_err}")
        
    except Exception as e:
        # 捕获所有 LLM 调用错误，返回友好的错误信息
        error_detail = traceback.format_exc()
        print(f"[LLM Error] {type(e).__name__}: {e}")
        print(f"[LLM Error Detail] {error_detail[:500]}")
        
        # 返回错误信息作为回答，避免前端收到 NotFoundError
        answer = f"抱歉，调用大模型时出错：{type(e).__name__}: {str(e)[:200]}。请检查模型配置或稍后重试。"
    
    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
    }

# 构建图
builder = StateGraph(RAGState)

# 添加节点
builder.add_node("index", index_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("generate", generate_node)

# 添加边
builder.add_edge(START, "index")
builder.add_edge("index", "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)

# 编译图
graph = builder.compile(name="rag-agent")
