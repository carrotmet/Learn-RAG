"""
RAG 教学项目 — SQLite 存储层（核心）

参考: /home/ubuntu/.openclaw/workspace/RAG教学/docs/监控指标体系搭建规划.md 2.2-2.3 节

功能:
- 初始化 SQLite 数据库及所有采集表
- 在线采集: API 调用时直写 SQLite
- 离线导入: 原始数据写入 raw_data 表
- 数据查询: 支持测试集构建的数据导出
- 用户反馈: 收集点赞/点踩等交互

设计原则:
- 轻量: 无需 Kafka/Prometheus 等企业级组件
- 可靠: 同步写入保证数据不丢
- 简单: 单文件 SQLite，易于备份和迁移
"""

import sqlite3
import json
import uuid
import os
import threading
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

# 使用绝对导入（支持直接运行和包导入两种模式）
try:
    from config import CollectionConfig
except ImportError:
    from .config import CollectionConfig


class SQLiteCollector:
    """轻量 SQLite 采集器 — 教学项目首选
    
    核心表结构:
    - conversations: 用户对话记录（主表）
    - retrieval_logs: 检索结果日志
    - llm_calls: LLM 调用详情
    - user_feedback: 用户反馈记录
    - raw_data: 离线上传原始数据
    - processed_data: 解析后的标准格式数据（测试集搭建用）
    """
    
    def __init__(self, db_path: str = "data/rag_data.db", config: Optional[CollectionConfig] = None):
        self.db_path = db_path
        self.config = config or CollectionConfig(db_path=db_path)
        
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        
        # 异步写入线程锁
        self._lock = threading.Lock()
        
        # 初始化所有表
        self._init_all_tables()
    
    @contextmanager
    def _get_conn(self):
        """获取数据库连接上下文管理器"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    
    def _init_all_tables(self):
        """初始化所有采集表 — 一次性创建"""
        with self._get_conn() as conn:
            conn.executescript("""
                -- 1. 用户对话记录（在线采集主表）
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT,
                    contexts TEXT,  -- JSON 数组
                    ground_truth TEXT,
                    model_version TEXT DEFAULT 'v1.0.0',
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    source TEXT DEFAULT 'online_api',  -- online_api|offline_upload|batch_export
                    metadata TEXT,  -- JSON: {upload_time, uploader, original_format, ...}
                    processing_stage TEXT DEFAULT 'raw'  -- raw|parsed|validated|cleaned|testset
                );
                
                CREATE INDEX IF NOT EXISTS idx_conv_time 
                    ON conversations(timestamp);
                CREATE INDEX IF NOT EXISTS idx_conv_source 
                    ON conversations(source);
                CREATE INDEX IF NOT EXISTS idx_conv_stage 
                    ON conversations(processing_stage);
                
                -- 2. 检索结果日志
                CREATE TABLE IF NOT EXISTS retrieval_logs (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT,
                    query TEXT,
                    retrieved_chunks TEXT,  -- JSON 数组
                    scores TEXT,  -- JSON 数组
                    latency_ms INTEGER,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_retrieval_conv 
                    ON retrieval_logs(conversation_id);
                
                -- 3. LLM 调用记录
                CREATE TABLE IF NOT EXISTS llm_calls (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT,
                    prompt TEXT,
                    response TEXT,
                    model_name TEXT,
                    token_usage TEXT,  -- JSON: {prompt_tokens, completion_tokens, total}
                    latency_ms INTEGER,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_llm_conv 
                    ON llm_calls(conversation_id);
                
                -- 4. 用户反馈
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    feedback_type TEXT NOT NULL,  -- thumbs_up|thumbs_down|comment|correction
                    content TEXT,  -- 评论内容或修正内容
                    rating INTEGER,  -- 1-5 星评分（可选）
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_feedback_conv 
                    ON user_feedback(conversation_id);
                CREATE INDEX IF NOT EXISTS idx_feedback_type 
                    ON user_feedback(feedback_type);
                
                -- 5. 离线上传原始数据（未解析的原始格式）
                CREATE TABLE IF NOT EXISTS raw_data (
                    id TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL,  -- offline_upload|batch_export|manual_entry
                    original_format TEXT,  -- json|jsonl|csv|xlsx|manual
                    raw_content TEXT NOT NULL,  -- 原始 JSON 字符串
                    upload_batch TEXT,  -- 批次号
                    metadata TEXT,  -- JSON: {uploader, upload_time, original_filename}
                    status TEXT DEFAULT 'pending',  -- pending|parsed|error
                    error_message TEXT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_raw_status 
                    ON raw_data(status);
                CREATE INDEX IF NOT EXISTS idx_raw_batch 
                    ON raw_data(upload_batch);
                
                -- 6. 解析后的标准格式数据（测试集搭建用）
                CREATE TABLE IF NOT EXISTS processed_data (
                    id TEXT PRIMARY KEY,
                    raw_id TEXT,  -- 关联 raw_data
                    question TEXT NOT NULL,
                    question_type TEXT DEFAULT 'factual',  -- factual|comparative|procedural|open
                    domain TEXT DEFAULT '其他',  -- 自指学|数学|AI|其他
                    difficulty TEXT DEFAULT 'medium',  -- easy|medium|hard
                    contexts TEXT,  -- JSON 数组
                    answer TEXT,
                    ground_truth TEXT,
                    metadata TEXT,  -- JSON
                    evaluation TEXT,  -- JSON: {ragas_faithfulness, ragas_relevancy, ...}
                    processing_stage TEXT DEFAULT 'parsed',  -- parsed|validated|cleaned|testset
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (raw_id) REFERENCES raw_data(id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_proc_stage 
                    ON processed_data(processing_stage);
                CREATE INDEX IF NOT EXISTS idx_proc_domain 
                    ON processed_data(domain);
                CREATE INDEX IF NOT EXISTS idx_proc_difficulty 
                    ON processed_data(difficulty);
            """)
    
    # ==================== 在线采集接口 ====================
    
    def save_conversation(
        self,
        question: str,
        answer: Optional[str] = None,
        contexts: Optional[List[str]] = None,
        ground_truth: Optional[str] = None,
        model_version: str = "v1.0.0",
        source: str = "online_api",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """保存对话记录 — 同步调用，约 5ms
        
        Args:
            question: 用户问题（必填）
            answer: 系统回答（可选）
            contexts: 检索到的上下文片段列表（可选）
            ground_truth: 人工标注标准答案（可选）
            model_version: 模型版本
            source: 数据来源（online_api/offline_upload/batch_export）
            metadata: 额外元数据字典
        
        Returns:
            生成的 conversation_id
        """
        # 数据质量校验
        if self.config.require_question and (not question or len(question.strip()) < self.config.min_question_length):
            raise ValueError(f"问题长度需 >= {self.config.min_question_length} 字符")
        
        conv_id = str(uuid.uuid4())
        
        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO conversations 
                (id, question, answer, contexts, ground_truth, 
                 model_version, source, metadata, processing_stage)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                conv_id,
                question.strip(),
                answer,
                json.dumps(contexts or [], ensure_ascii=False),
                ground_truth,
                model_version,
                source,
                json.dumps(metadata or {}, ensure_ascii=False),
                'raw'
            ))
        
        return conv_id
    
    def save_retrieval_log(
        self,
        conversation_id: str,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        scores: Optional[List[float]] = None,
        latency_ms: Optional[int] = None
    ) -> str:
        """保存检索结果日志
        
        Args:
            conversation_id: 关联的对话 ID
            query: 检索查询文本
            retrieved_chunks: 检索到的文档片段列表 [{content, source, page, score}, ...]
            scores: 检索分数列表
            latency_ms: 检索耗时（毫秒）
        """
        log_id = str(uuid.uuid4())
        
        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO retrieval_logs 
                (id, conversation_id, query, retrieved_chunks, scores, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                log_id,
                conversation_id,
                query,
                json.dumps(retrieved_chunks, ensure_ascii=False),
                json.dumps(scores or [], ensure_ascii=False),
                latency_ms
            ))
        
        return log_id
    
    def save_llm_call(
        self,
        conversation_id: str,
        prompt: str,
        response: str,
        model_name: str,
        token_usage: Optional[Dict[str, int]] = None,
        latency_ms: Optional[int] = None
    ) -> str:
        """保存 LLM 调用记录
        
        Args:
            conversation_id: 关联的对话 ID
            prompt: 发送给 LLM 的完整提示词
            response: LLM 返回的响应文本
            model_name: 模型名称
            token_usage: Token 使用量 {prompt_tokens, completion_tokens, total}
            latency_ms: 调用耗时（毫秒）
        """
        call_id = str(uuid.uuid4())
        
        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO llm_calls 
                (id, conversation_id, prompt, response, model_name, token_usage, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                call_id,
                conversation_id,
                prompt,
                response,
                model_name,
                json.dumps(token_usage or {}, ensure_ascii=False),
                latency_ms
            ))
        
        return call_id
    
    def save_feedback(
        self,
        conversation_id: str,
        feedback_type: str,  # thumbs_up|thumbs_down|comment|correction
        content: Optional[str] = None,
        rating: Optional[int] = None
    ) -> str:
        """保存用户反馈
        
        Args:
            conversation_id: 关联的对话 ID
            feedback_type: 反馈类型
            content: 评论内容或修正内容
            rating: 1-5 星评分
        """
        fb_id = str(uuid.uuid4())
        
        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO user_feedback 
                (id, conversation_id, feedback_type, content, rating)
                VALUES (?, ?, ?, ?, ?)
            """, (fb_id, conversation_id, feedback_type, content, rating))
        
        return fb_id
    
    # ==================== 离线导入接口 ====================
    
    def save_raw_data(
        self,
        raw_content: Dict[str, Any],
        source_type: str = "offline_upload",
        original_format: str = "json",
        upload_batch: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """保存离线上传原始数据
        
        将上传的原始数据（JSON/CSV/Excel 解析后的字典）保存到 raw_data 表，
        等待后续解析流程处理。
        
        Args:
            raw_content: 原始数据字典（会序列化为 JSON）
            source_type: 数据来源类型
            original_format: 原始文件格式
            upload_batch: 批次号（如 batch_20250611_001）
            metadata: 上传元数据
        
        Returns:
            raw_data 记录 ID
        """
        raw_id = str(uuid.uuid4())
        
        # 自动生成批次号
        if upload_batch is None:
            upload_batch = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO raw_data 
                (id, source_type, original_format, raw_content, upload_batch, metadata, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                raw_id,
                source_type,
                original_format,
                json.dumps(raw_content, ensure_ascii=False),
                upload_batch,
                json.dumps(metadata or {}, ensure_ascii=False),
                'pending'
            ))
        
        return raw_id
    
    def save_processed_data(
        self,
        question: str,
        question_type: str = "factual",
        domain: str = "其他",
        difficulty: str = "medium",
        contexts: Optional[List[Dict[str, Any]]] = None,
        answer: Optional[str] = None,
        ground_truth: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        raw_id: Optional[str] = None,
        processing_stage: str = "parsed"
    ) -> str:
        """保存解析后的标准格式数据
        
        这是测试集搭建的核心输入，字段符合统一数据格式标准。
        
        Args:
            question: 用户问题（必填）
            question_type: 问题类型 factual|comparative|procedural|open
            domain: 领域 自指学|数学|AI|其他
            difficulty: 难度 easy|medium|hard
            contexts: 上下文片段列表 [{content, source, page, relevance_score}, ...]
            answer: 系统生成回答
            ground_truth: 人工标注标准答案
            metadata: 元数据
            raw_id: 关联的原始数据 ID
            processing_stage: 处理阶段 parsed|validated|cleaned|testset
        """
        proc_id = str(uuid.uuid4())
        
        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO processed_data 
                (id, raw_id, question, question_type, domain, difficulty,
                 contexts, answer, ground_truth, metadata, processing_stage)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                proc_id,
                raw_id,
                question,
                question_type,
                domain,
                difficulty,
                json.dumps(contexts or [], ensure_ascii=False),
                answer,
                ground_truth,
                json.dumps(metadata or {}, ensure_ascii=False),
                processing_stage
            ))
        
        return proc_id
    
    # ==================== 查询接口 ====================
    
    def get_conversation(self, conv_id: str) -> Optional[Dict[str, Any]]:
        """获取单条对话记录"""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE id = ?",
                (conv_id,)
            ).fetchone()
            return dict(row) if row else None
    
    def get_recent_conversations(
        self,
        limit: int = 100,
        source: Optional[str] = None,
        processing_stage: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """获取最近对话 — 用于测试集构建
        
        Args:
            limit: 返回数量限制
            source: 按数据来源过滤（online_api/offline_upload/batch_export）
            processing_stage: 按处理阶段过滤
        """
        with self._get_conn() as conn:
            conditions = []
            params = []
            
            if source:
                conditions.append("source = ?")
                params.append(source)
            if processing_stage:
                conditions.append("processing_stage = ?")
                params.append(processing_stage)
            
            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
            params.append(limit)
            
            query = f"""
                SELECT * FROM conversations 
                {where_clause}
                ORDER BY timestamp DESC 
                LIMIT ?
            """
            
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]
    
    def get_conversation_with_details(self, conv_id: str) -> Dict[str, Any]:
        """获取对话完整详情（含检索日志、LLM调用、用户反馈）"""
        with self._get_conn() as conn:
            # 主对话
            conv = conn.execute(
                "SELECT * FROM conversations WHERE id = ?",
                (conv_id,)
            ).fetchone()
            
            if not conv:
                return {}
            
            result = dict(conv)
            
            # 检索日志
            result['retrieval_logs'] = [
                dict(row) for row in conn.execute(
                    "SELECT * FROM retrieval_logs WHERE conversation_id = ? ORDER BY timestamp",
                    (conv_id,)
                ).fetchall()
            ]
            
            # LLM 调用
            result['llm_calls'] = [
                dict(row) for row in conn.execute(
                    "SELECT * FROM llm_calls WHERE conversation_id = ? ORDER BY timestamp",
                    (conv_id,)
                ).fetchall()
            ]
            
            # 用户反馈
            result['feedbacks'] = [
                dict(row) for row in conn.execute(
                    "SELECT * FROM user_feedback WHERE conversation_id = ? ORDER BY timestamp",
                    (conv_id,)
                ).fetchall()
            ]
            
            return result
    
    def get_pending_raw_data(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取待解析的原始数据"""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM raw_data WHERE status = 'pending' ORDER BY timestamp LIMIT ?",
                (limit,)
            ).fetchall()
            return [dict(row) for row in rows]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取采集统计信息"""
        with self._get_conn() as conn:
            stats = {}
            
            # 各表记录数
            for table in ['conversations', 'retrieval_logs', 'llm_calls', 
                         'user_feedback', 'raw_data', 'processed_data']:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                stats[table] = count
            
            # 数据来源分布
            source_dist = conn.execute(
                "SELECT source, COUNT(*) as count FROM conversations GROUP BY source"
            ).fetchall()
            stats['source_distribution'] = {row['source']: row['count'] for row in source_dist}
            
            # 反馈类型分布
            feedback_dist = conn.execute(
                "SELECT feedback_type, COUNT(*) as count FROM user_feedback GROUP BY feedback_type"
            ).fetchall()
            stats['feedback_distribution'] = {row['feedback_type']: row['count'] for row in feedback_dist}
            
            # 原始数据状态
            raw_status = conn.execute(
                "SELECT status, COUNT(*) as count FROM raw_data GROUP BY status"
            ).fetchall()
            stats['raw_status'] = {row['status']: row['count'] for row in raw_status}
            
            # 处理阶段分布
            stage_dist = conn.execute(
                "SELECT processing_stage, COUNT(*) as count FROM processed_data GROUP BY processing_stage"
            ).fetchall()
            stats['processing_stage_distribution'] = {row['processing_stage']: row['count'] for row in stage_dist}
            
            return stats
    
    def update_raw_data_status(
        self,
        raw_id: str,
        status: str,
        error_message: Optional[str] = None
    ):
        """更新原始数据处理状态"""
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE raw_data SET status = ?, error_message = ? WHERE id = ?",
                (status, error_message, raw_id)
            )
    
    def update_conversation_stage(
        self,
        conv_id: str,
        stage: str
    ):
        """更新对话处理阶段"""
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE conversations SET processing_stage = ? WHERE id = ?",
                (stage, conv_id)
            )
    
    def export_to_jsonl(
        self,
        output_path: str,
        table: str = "conversations",
        conditions: Optional[str] = None,
        limit: int = 10000
    ) -> int:
        """导出数据到 JSONL 文件
        
        Args:
            output_path: 输出文件路径
            table: 要导出的表名
            conditions: WHERE 条件字符串（如 "source = 'offline_upload'"）
            limit: 最大导出数量
        
        Returns:
            导出的记录数
        """
        import json
        
        with self._get_conn() as conn:
            where = f"WHERE {conditions}" if conditions else ""
            
            # 根据表名选择排序字段
            if table in ['conversations', 'retrieval_logs', 'llm_calls', 'user_feedback', 'raw_data']:
                order_by = "ORDER BY timestamp DESC"
            elif table == 'processed_data':
                order_by = "ORDER BY created_at DESC"
            else:
                order_by = ""
            
            query = f"SELECT * FROM {table} {where} {order_by} LIMIT ?"
            rows = conn.execute(query, (limit,)).fetchall()
        
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for row in rows:
                record = dict(row)
                # 解析 JSON 字段
                for key in ['contexts', 'metadata', 'retrieved_chunks', 'scores', 
                           'token_usage', 'raw_content', 'evaluation']:
                    if key in record and record[key]:
                        try:
                            record[key] = json.loads(record[key])
                        except (json.JSONDecodeError, TypeError):
                            pass
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        return len(rows)
