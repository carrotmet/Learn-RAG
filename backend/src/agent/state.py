from typing import TypedDict, Annotated, Optional
from langgraph.graph import add_messages


class RAGState(TypedDict):
    """RAG 图状态定义
    
    扩展字段（二阶段 HybridRAG）：
    - enrich_complete: 问题完整度判定
    - enrich_reason: 完整度判定原因
    - enrich_rewritten: 改写后的问题
    - enrich_follow_up: 追问提示（不完整时）
    - intents: 识别到的意图列表
    - primary_intent: 主意图
    - needs_retrieve: 是否需要检索
    - retrieve_strategies: 检索策略列表
    - retrieve_weights: 通道权重
    - retrieve_mode: 检索模式
    - recall_results: 多路召回原始结果
    - fused_results: 融合后结果
    
    扩展字段（用于在线采集）：
    - retrieval_latency: 检索耗时（毫秒）
    """
    messages: Annotated[list, add_messages]  # 对话消息历史
    question: str                            # 当前用户问题
    retrieved_docs: list                     # 检索到的文档片段
    answer: str                              # 生成的答案
    retrieval_latency: int                   # 检索耗时（毫秒）
    
    # HybridRAG 扩展字段
    enrich_complete: Optional[bool]
    enrich_reason: Optional[str]
    enrich_rewritten: Optional[str]
    enrich_follow_up: Optional[str]
    intents: Optional[list]
    primary_intent: Optional[str]
    needs_retrieve: Optional[bool]
    retrieve_strategies: Optional[list]
    retrieve_weights: Optional[dict]
    retrieve_mode: Optional[str]
    recall_results: Optional[dict]
    fused_results: Optional[list]
