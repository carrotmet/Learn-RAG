from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig

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
    
    docs = vector_store.search(question, k=4)
    
    return {
        "retrieved_docs": docs,
        "question": question,
    }

# 节点 3：生成（带容错：检索失败或无结果时直接调用 LLM）
def generate_node(state: RAGState, config: RunnableConfig) -> RAGState:
    """生成节点：基于检索结果生成答案，检索失败时直接调用 LLM"""
    question = state.get("question", "")
    docs = state.get("retrieved_docs", [])
    
    # 调用 LLM
    llm = OpenRouterLLM()
    
    if not docs or len(docs) == 0:
        # 知识库无结果，直接调用 LLM 回答
        answer = llm.generate(
            f"请回答以下问题：{question}",
            system="你是一个通用的知识助手。当前知识库中没有相关文档，请基于你的训练知识回答问题。"
        )
    else:
        # 有检索结果，基于上下文生成
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"""基于以下检索到的文档，回答用户问题。如果文档中没有相关信息，请基于你的知识回答。

--- 检索到的文档 ---
{context}

--- 用户问题 ---
{question}

请给出清晰、准确的回答："""
        answer = llm.generate(prompt, system="你是一个专业的知识助手，优先基于提供的文档回答问题。")
    
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
