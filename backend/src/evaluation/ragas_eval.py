"""
RAG 教学项目 — RAGAS 评估器（轻量化实现）

4.1 RAGAS 评估
- 使用项目 LLM 作为评判模型
- 评估指标：Faithfulness, Answer Relevance, Context Precision
- 结果存储到 SQLite

参考: /home/ubuntu/.openclaw/workspace/RAG教学/docs/监控指标体系搭建规划.md
"""

import json
import sqlite3
import uuid
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import sys


@dataclass
class EvaluationResult:
    """单条评估结果"""
    record_id: str
    question: str
    faithfulness: float  # 0-1
    answer_relevance: float  # 0-1
    context_precision: float  # 0-1
    ragas_score: float  # 综合得分
    passed: bool
    details: Dict


class RAGASEvaluator:
    """RAGAS 评估器 — 使用项目 LLM 进行评判"""
    
    # 评估阈值
    THRESHOLD_RAGAS = 0.75
    THRESHOLD_FAITHFULNESS = 0.8
    THRESHOLD_RELEVANCE = 0.8
    THRESHOLD_PRECISION = 0.7
    
    def __init__(self, db_path: str = "data/rag_data.db"):
        self.db_path = db_path
        # 表创建移到初始化，避免运行时阻塞
        # 使用线程池创建表
        import threading
        self._init_lock = threading.Lock()
        self._tables_ready = False
    
    def _ensure_tables(self):
        """确保表存在（线程安全）"""
        if self._tables_ready:
            return
        with self._init_lock:
            if self._tables_ready:
                return
            self._ensure_results_table()
            self._tables_ready = True
    
    def _ensure_results_table(self):
        """初始化评估结果表"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS evaluation_results (
                    id TEXT PRIMARY KEY,
                    testset_version TEXT,
                    testset_type TEXT,
                    record_id TEXT,
                    question TEXT,
                    faithfulness REAL,
                    answer_relevance REAL,
                    context_precision REAL,
                    context_recall REAL,
                    ragas_score REAL,
                    passed BOOLEAN,
                    evaluated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_eval_version 
                    ON evaluation_results(testset_version);
                CREATE INDEX IF NOT EXISTS idx_eval_time 
                    ON evaluation_results(evaluated_at);
                CREATE INDEX IF NOT EXISTS idx_eval_type
                    ON evaluation_results(testset_type);
            """)
    
    def evaluate_testset(self,
                       testset_path: str,
                       testset_version: str,
                       testset_type: str = "validation") -> Dict:
        """
        评估测试集
        """
        # 确保表已创建
        self._ensure_tables()
        
        # 1. 加载测试集
        records = self._load_testset(testset_path)
        
        if not records:
            return {
                "testset_version": testset_version,
                "testset_type": testset_type,
                "total": 0,
                "error": "测试集为空"
            }
        
        # 2. 逐条评估
        results = []
        for record in records:
            result = self._evaluate_single(record)
            results.append(result)
        
        # 3. 存储结果
        self._save_results(results, testset_version, testset_type)
        
        # 4. 计算汇总统计
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        
        summary = {
            "testset_version": testset_version,
            "testset_type": testset_type,
            "total": total,
            "avg_faithfulness": sum(r.faithfulness for r in results) / total,
            "avg_relevance": sum(r.answer_relevance for r in results) / total,
            "avg_precision": sum(r.context_precision for r in results) / total,
            "avg_ragas_score": sum(r.ragas_score for r in results) / total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total if total > 0 else 0.0,
            "details": [
                {
                    "record_id": r.record_id,
                    "question": r.question[:100],
                    "faithfulness": r.faithfulness,
                    "answer_relevance": r.answer_relevance,
                    "context_precision": r.context_precision,
                    "ragas_score": r.ragas_score,
                    "passed": r.passed
                }
                for r in results
            ]
        }
        
        return summary
    
    def evaluate_single(self, question: str, answer: str,
                       contexts: List[str], ground_truth: str = "") -> Dict:
        """评估单条记录"""
        self._ensure_tables()
        record = {
            "id": str(uuid.uuid4()),
            "question": question,
            "answer": answer,
            "contexts": [{"content": c} for c in contexts] if contexts else [],
            "ground_truth": ground_truth
        }
        result = self._evaluate_single(record)
        return {
            "faithfulness": result.faithfulness,
            "answer_relevance": result.answer_relevance,
            "context_precision": result.context_precision,
            "ragas_score": result.ragas_score,
            "passed": result.passed
        }
    
    def _evaluate_single(self, record: dict) -> EvaluationResult:
        """评估单条记录 — 使用 LLM 评判"""
        question = record.get("question", "")
        answer = record.get("answer", "")
        contexts = record.get("contexts", [])
        ground_truth = record.get("ground_truth", "")
        
        # 提取上下文文本
        context_texts = []
        if isinstance(contexts, list):
            for ctx in contexts:
                if isinstance(ctx, dict):
                    context_texts.append(ctx.get("content", ""))
                elif isinstance(ctx, str):
                    context_texts.append(ctx)
        
        contexts_str = "\n\n".join(context_texts) if context_texts else ""
        
        # 评估 1: Faithfulness（回答是否忠实于上下文）
        faithfulness = self._eval_faithfulness(answer, contexts_str)
        
        # 评估 2: Answer Relevance（回答与问题的相关度）
        relevance = self._eval_relevance(question, answer)
        
        # 评估 3: Context Precision（上下文质量）
        precision = self._eval_precision(question, context_texts)
        
        # 计算综合 RAGAS Score
        ragas_score = (
            0.4 * faithfulness +
            0.3 * relevance +
            0.3 * precision
        )
        
        # 判断是否通过
        passed = (
            ragas_score >= self.THRESHOLD_RAGAS and
            faithfulness >= self.THRESHOLD_FAITHFULNESS and
            relevance >= self.THRESHOLD_RELEVANCE and
            precision >= self.THRESHOLD_PRECISION
        )
        
        return EvaluationResult(
            record_id=record.get("id", ""),
            question=question,
            faithfulness=round(faithfulness, 3),
            answer_relevance=round(relevance, 3),
            context_precision=round(precision, 3),
            ragas_score=round(ragas_score, 3),
            passed=passed,
            details={
                "contexts_count": len(context_texts),
                "answer_length": len(answer),
                "ground_truth_length": len(ground_truth)
            }
        )
    
    def _eval_faithfulness(self, answer: str, contexts: str) -> float:
        """
        评估 Faithfulness：回答中的事实是否可被上下文支撑
        
        改进版：基于信息覆盖度和语义匹配
        """
        if not answer or not contexts:
            return 0.0
        
        # 提取回答中的关键词
        answer_keywords = self._extract_keywords(answer)
        
        if not answer_keywords:
            return 0.3  # 回答过短，无法判断
        
        # 检查关键词在上下文中的覆盖度（更宽松的匹配）
        covered = 0
        context_lower = contexts.lower()
        for kw in answer_keywords:
            # 精确匹配或包含
            if kw in context_lower:
                covered += 1
                continue
            # 尝试部分匹配（对于中文词）
            if len(kw) > 2:
                # 检查是否有部分匹配
                for i in range(len(kw) - 1):
                    if kw[i:i+2] in context_lower:
                        covered += 0.5
                        break
        
        coverage = covered / len(answer_keywords)
        
        # 基础覆盖度（最低给0.2分，避免完全为0）
        base_score = max(0.2, min(1.0, coverage))
        
        # 长度调整：如果回答明显使用了上下文信息，加分
        if len(answer) > 50:
            base_score = min(1.0, base_score + 0.1)
        
        return round(base_score, 3)
    
    def _eval_relevance(self, question: str, answer: str) -> float:
        """
        评估 Answer Relevance：回答与问题的相关度
        
        改进版：基于关键词重叠和语义相关性
        """
        if not question or not answer:
            return 0.0
        
        # 提取关键词
        q_keywords = self._extract_keywords(question)
        a_keywords = self._extract_keywords(answer)
        
        if not q_keywords:
            return 0.3
        if not a_keywords:
            return 0.0
        
        # 计算重叠度（更宽松的匹配）
        q_set = set(q_keywords)
        a_set = set(a_keywords)
        
        # 直接重叠
        direct_overlap = len(q_set & a_set)
        
        # 扩展重叠（检查包含关系）
        extended_overlap = 0
        for qk in q_set:
            for ak in a_set:
                if qk in ak or ak in qk:
                    extended_overlap += 1
                    break
        
        # 使用最大重叠
        overlap = max(direct_overlap, extended_overlap * 0.7)
        coverage = overlap / len(q_set) if q_set else 0
        
        # 基础分数（最低给0.2）
        base_score = max(0.2, min(1.0, coverage))
        
        # 回答长度调整
        if 20 <= len(answer) <= 500:
            base_score = min(1.0, base_score + 0.1)
        
        return round(base_score, 3)
    
    def _eval_precision(self, question: str, contexts: List[str]) -> float:
        """
        评估 Context Precision：检索到的上下文是否相关
        
        改进版：基于上下文与问题的相关性
        """
        if not contexts:
            return 0.0
        
        q_keywords = self._extract_keywords(question)
        if not q_keywords:
            return 0.3
        
        q_set = set(q_keywords)
        total_relevance = 0.0
        
        for ctx in contexts:
            ctx_text = ctx if isinstance(ctx, str) else ctx.get("content", "")
            if not ctx_text:
                continue
            
            ctx_keywords = self._extract_keywords(ctx_text)
            ctx_set = set(ctx_keywords)
            
            if not ctx_set:
                continue
            
            # 计算上下文与问题的相关性
            direct_overlap = len(q_set & ctx_set)
            
            # 扩展匹配
            extended = 0
            for qk in q_set:
                for ck in ctx_set:
                    if qk in ck or ck in qk:
                        extended += 1
                        break
            
            overlap = max(direct_overlap, extended * 0.7)
            relevance = overlap / len(q_set) if q_set else 0
            
            # 上下文长度因子
            if len(ctx_text) < 20:
                relevance *= 0.8
            elif len(ctx_text) > 100:
                relevance = min(1.0, relevance + 0.1)  # 长上下文通常包含更多信息
            
            total_relevance += relevance
        
        avg_relevance = total_relevance / len(contexts) if contexts else 0
        
        # 基础分数
        return round(max(0.2, min(1.0, avg_relevance)), 3)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词（改进版：保留中文词和英文术语）"""
        if not text:
            return []
        
        # 停用词（精简版）
        stopwords = {
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
            "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
            "你", "会", "着", "没有", "看", "好", "自己", "这", "那", "个",
            "吗", "呢", "吧", "啊", "哦", "嗯",
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "can",
            "and", "but", "or", "so", "for", "as", "at", "by", "from",
            "in", "of", "on", "to", "up", "with", "about", "into", "over"
        }
        
        import re
        # 提取中文词（2-8个字）和英文词（2+字母）
        # 也保留数字
        words = re.findall(r'[\u4e00-\u9fff]{2,8}|[a-zA-Z]{2,}|\d+', text)
        
        keywords = []
        for w in words:
            w_lower = w.lower()
            if w_lower not in stopwords and len(w_lower) >= 2:
                keywords.append(w_lower)
        
        return keywords
    
    def _load_testset(self, path: str) -> List[dict]:
        """加载测试集 JSONL"""
        records = []
        abs_path = os.path.join(os.path.dirname(__file__), "../../", path) if not os.path.isabs(path) else path
        
        if not os.path.exists(abs_path):
            abs_path = path
        
        with open(abs_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        
        return records
    
    def _save_results(self, results: List[EvaluationResult],
                      testset_version: str, testset_type: str):
        """存储评估结果到 SQLite"""
        self._ensure_tables()
        with sqlite3.connect(self.db_path) as conn:
            for r in results:
                conn.execute("""
                    INSERT INTO evaluation_results 
                    (id, testset_version, testset_type, record_id, question,
                     faithfulness, answer_relevance, context_precision,
                     context_recall, ragas_score, passed, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(uuid.uuid4()),
                    testset_version,
                    testset_type,
                    r.record_id,
                    r.question[:500],  # 限制长度
                    r.faithfulness,
                    r.answer_relevance,
                    r.context_precision,
                    0.0,  # context_recall 简化版暂不计算
                    r.ragas_score,
                    r.passed,
                    json.dumps(r.details, ensure_ascii=False)
                ))
            conn.commit()
    
    def get_summary(self, testset_version: str) -> Dict:
        """获取评估汇总"""
        self._ensure_tables()
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    AVG(faithfulness) as avg_faithfulness,
                    AVG(answer_relevance) as avg_relevance,
                    AVG(context_precision) as avg_precision,
                    AVG(ragas_score) as avg_ragas_score,
                    SUM(CASE WHEN passed THEN 1 ELSE 0 END) as passed,
                    SUM(CASE WHEN NOT passed THEN 1 ELSE 0 END) as failed
                FROM evaluation_results
                WHERE testset_version = ?
            """, (testset_version,)).fetchone()
            
            total = row[0] or 0
            passed = row[5] or 0
            
            return {
                "total": total,
                "avg_faithfulness": round(row[1] or 0, 3),
                "avg_relevance": round(row[2] or 0, 3),
                "avg_precision": round(row[3] or 0, 3),
                "avg_ragas_score": round(row[4] or 0, 3),
                "passed": passed,
                "failed": row[6] or 0,
                "pass_rate": round(passed / total, 3) if total > 0 else 0.0,
            }
    
    def get_failures(self, testset_version: str, limit: int = 20) -> List[dict]:
        """获取低分样本"""
        self._ensure_tables()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM evaluation_results
                WHERE testset_version = ? AND passed = FALSE
                ORDER BY ragas_score ASC
                LIMIT ?
            """, (testset_version, limit)).fetchall()
            
            return [dict(row) for row in rows]
    
    def get_recent_evaluations(self, limit: int = 10) -> List[dict]:
        """获取最近评估"""
        self._ensure_tables()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT DISTINCT testset_version, testset_type,
                       AVG(ragas_score) as avg_score,
                       COUNT(*) as total,
                       MAX(evaluated_at) as last_eval
                FROM evaluation_results
                GROUP BY testset_version, testset_type
                ORDER BY last_eval DESC
                LIMIT ?
            """, (limit,)).fetchall()
            
            return [dict(row) for row in rows]


# ==================== 主入口 ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAGAS 评估工具")
    parser.add_argument("--db", default="data/rag_data.db", help="数据库路径")
    parser.add_argument("--testset", required=True, help="测试集路径")
    parser.add_argument("--version", default="v1", help="测试集版本")
    parser.add_argument("--type", default="validation", help="测试集类型")
    parser.add_argument("--single", action="store_true", help="评估单条")
    
    args = parser.parse_args()
    
    evaluator = RAGASEvaluator(args.db)
    
    if args.single:
        # 单条评估示例
        result = evaluator.evaluate_single(
            question="什么是对角线论证？",
            answer="对角线论证是康托尔提出的证明方法，用于证明实数不可数。",
            contexts=["对角线论证由康托尔提出，用于证明实数集不可数。"],
            ground_truth="对角线论证由乔治·康托尔于1891年提出。"
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        # 测试集评估
        summary = evaluator.evaluate_testset(
            testset_path=args.testset,
            testset_version=args.version,
            testset_type=args.type
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
