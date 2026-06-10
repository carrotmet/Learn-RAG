from typing import TypedDict, Annotated
from langgraph.graph import add_messages


class RAGState(TypedDict):
    """RAG 图状态定义"""
    messages: Annotated[list, add_messages]  # 对话消息历史
    question: str                            # 当前用户问题
    retrieved_docs: list                     # 检索到的文档片段
    answer: str                              # 生成的答案
