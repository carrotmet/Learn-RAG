"""
RAG 教学项目 — 测试集搭建模块

3.1 数据格式统一 + 3.2 离线数据导入
3.3 数据解析
3.4 数据集搭建

参考: /home/ubuntu/.openclaw/workspace/RAG教学/docs/监控指标体系搭建规划.md
"""

import sqlite3
import json
import uuid
import re
import random
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import os


# ==================== 3.1 数据格式统一 ====================

# 标准数据格式定义（JSON Schema）
STANDARD_SCHEMA = {
    "required": ["id", "question", "question_type", "domain", "difficulty"],
    "fields": {
        "id": {"type": "string", "pattern": "^[0-9a-f-]{36}$"},
        "question": {"type": "string", "min_length": 5, "max_length": 2000},
        "question_type": {"type": "enum", "values": ["factual", "comparative", "procedural", "open"]},
        "domain": {"type": "string"},
        "difficulty": {"type": "enum", "values": ["easy", "medium", "hard"]},
        "contexts": {"type": "array"},
        "answer": {"type": "string"},
        "ground_truth": {"type": "string"},
        "metadata": {"type": "object"},
        "evaluation": {"type": "object"},
    }
}


class DataFormatValidator:
    """数据格式统一验证器"""
    
    @classmethod
    def validate(cls, record: dict) -> Tuple[bool, List[str]]:
        """
        验证单条记录是否符合标准格式
        返回: (是否通过, 错误列表)
        """
        errors = []
        
        # 必填字段检查
        for field in STANDARD_SCHEMA["required"]:
            if field not in record or record[field] is None or record[field] == '':
                errors.append(f"缺少必填字段: {field}")
        
        if errors:
            return False, errors
        
        # 字段类型和长度检查
        question = record.get("question", "")
        if len(question) < 5:
            errors.append(f"question 过短: {len(question)} 字符 (最小 5)")
        if len(question) > 2000:
            errors.append(f"question 过长: {len(question)} 字符 (最大 2000)")
        
        # 枚举值检查
        q_type = record.get("question_type", "")
        if q_type not in STANDARD_SCHEMA["fields"]["question_type"]["values"]:
            errors.append(f"question_type 无效: {q_type}")
        
        diff = record.get("difficulty", "")
        if diff not in STANDARD_SCHEMA["fields"]["difficulty"]["values"]:
            errors.append(f"difficulty 无效: {diff}")
        
        # contexts 格式检查
        contexts = record.get("contexts", [])
        if contexts and isinstance(contexts, list):
            for i, ctx in enumerate(contexts):
                if isinstance(ctx, dict):
                    if not ctx.get("content"):
                        errors.append(f"contexts[{i}] 缺少 content")
                elif not isinstance(ctx, str) or not ctx.strip():
                    errors.append(f"contexts[{i}] 内容为空")
        
        return len(errors) == 0, errors
    
    @classmethod
    def normalize(cls, record: dict) -> dict:
        """
        将原始记录标准化为统一格式
        """
        # 先解析 metadata 以获取 domain/difficulty
        metadata = record.get("metadata", {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except:
                metadata = {}
        
        normalized = {
            "id": record.get("id") or str(uuid.uuid4()),
            "question": str(record.get("question", "")).strip(),
            "question_type": cls._infer_question_type(record.get("question", "")),
            "domain": metadata.get("domain", record.get("domain", "其他")),
            "difficulty": metadata.get("difficulty", record.get("difficulty", "medium")),
            "contexts": cls._normalize_contexts(record.get("contexts", [])),
            "answer": str(record.get("answer", "")).strip(),
            "ground_truth": str(record.get("ground_truth", "")).strip() or None,
            "metadata": cls._normalize_metadata(record),
            "evaluation": record.get("evaluation", {}),
        }
        return normalized
    
    @staticmethod
    def _infer_question_type(question: str) -> str:
        """推断问题类型"""
        if not question:
            return "open"
        q = question.lower()
        if any(w in q for w in ["什么", "是谁", "哪里", "when", "what", "who", "which", "几", "多少"]):
            return "factual"
        elif any(w in q for w in ["比较", "区别", "vs", "difference", "compare", "对比", "versus"]):
            return "comparative"
        elif any(w in q for w in ["如何", "怎么", "步骤", "how to", "procedure", "流程", "方法"]):
            return "procedural"
        else:
            return "open"
    
    @staticmethod
    def _normalize_contexts(contexts) -> List[dict]:
        """标准化上下文格式"""
        if not contexts:
            return []
        
        if isinstance(contexts, str):
            try:
                contexts = json.loads(contexts)
            except:
                contexts = [contexts]
        
        if not isinstance(contexts, list):
            contexts = [contexts]
        
        normalized = []
        for ctx in contexts:
            if isinstance(ctx, dict):
                normalized.append({
                    "content": str(ctx.get("content", "")).strip(),
                    "source": ctx.get("source", "unknown"),
                    "page": ctx.get("page", 1),
                    "relevance_score": ctx.get("relevance_score", 0.9)
                })
            elif isinstance(ctx, str) and ctx.strip():
                normalized.append({
                    "content": ctx.strip(),
                    "source": "unknown",
                    "page": 1,
                    "relevance_score": 0.9
                })
        
        return normalized
    
    @staticmethod
    def _normalize_metadata(record: dict) -> dict:
        """标准化元数据"""
        metadata = record.get("metadata", {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except:
                metadata = {}
        
        return {
            "raw_id": record.get("id"),
            "timestamp": metadata.get("timestamp", datetime.now().isoformat()),
            "model_version": metadata.get("model_version", "v1.0.0"),
            "data_source": metadata.get("data_source", "unknown"),
            "batch_id": metadata.get("batch_id", "import_" + datetime.now().strftime("%Y%m%d")),
            "processing_stage": "raw",
            "quality_score": 0.0,
            "domain": metadata.get("domain", "其他"),
            "difficulty": metadata.get("difficulty", "medium"),
            "question_type": metadata.get("question_type", "factual"),
        }


# ==================== 3.2 离线数据导入 ====================

class DataImporter:
    """离线数据导入器 — 从 conversations/raw_data 导入到 processed_data"""
    
    def __init__(self, db_path: str = "data/rag_data.db"):
        self.db_path = db_path
        self.validator = DataFormatValidator()
        self._ensure_processed_table()
    
    def _ensure_processed_table(self):
        """确保 processed_data 表存在（含新字段）"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS processed_data (
                    id TEXT PRIMARY KEY,
                    raw_id TEXT,
                    question TEXT NOT NULL,
                    question_type TEXT DEFAULT 'factual',
                    domain TEXT DEFAULT '其他',
                    difficulty TEXT DEFAULT 'medium',
                    contexts TEXT,
                    answer TEXT,
                    ground_truth TEXT,
                    metadata TEXT,
                    evaluation TEXT,
                    processing_stage TEXT DEFAULT 'parsed',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_proc_stage ON processed_data(processing_stage);
                CREATE INDEX IF NOT EXISTS idx_proc_domain ON processed_data(domain);
                CREATE INDEX IF NOT EXISTS idx_proc_difficulty ON processed_data(difficulty);
                CREATE INDEX IF NOT EXISTS idx_proc_type ON processed_data(question_type);
            """)
    
    def import_from_conversations(self, 
                                   source: Optional[str] = None,
                                   batch_size: int = 100) -> dict:
        """
        从 conversations 表导入到 processed_data
        
        Args:
            source: 过滤来源（online_api/offline_upload/batch_export）
            batch_size: 每批处理数量
        
        Returns:
            导入统计 {"imported": int, "skipped": int, "errors": int}
        """
        imported = 0
        skipped = 0
        errors = 0
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # 构建查询
            if source:
                raw_rows = conn.execute(
                    """SELECT * FROM conversations 
                       WHERE source = ? 
                       AND id NOT IN (
                           SELECT raw_id FROM processed_data 
                           WHERE raw_id IS NOT NULL
                       )
                       LIMIT ?""",
                    (source, batch_size)
                ).fetchall()
            else:
                raw_rows = conn.execute(
                    """SELECT * FROM conversations 
                       WHERE id NOT IN (
                           SELECT raw_id FROM processed_data 
                           WHERE raw_id IS NOT NULL
                       )
                       LIMIT ?""",
                    (batch_size,)
                ).fetchall()
            
            for row in raw_rows:
                try:
                    raw_dict = dict(row)
                    # 标准化
                    normalized = self.validator.normalize(raw_dict)
                    
                    # 验证
                    is_valid, err_list = self.validator.validate(normalized)
                    if not is_valid:
                        skipped += 1
                        continue
                    
                    # 写入 processed_data
                    conn.execute("""
                        INSERT INTO processed_data 
                        (id, raw_id, question, question_type, domain, difficulty,
                         contexts, answer, ground_truth, metadata, evaluation, 
                         processing_stage, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        normalized["id"],
                        raw_dict.get("id"),
                        normalized["question"],
                        normalized["question_type"],
                        normalized["domain"],
                        normalized["difficulty"],
                        json.dumps(normalized["contexts"], ensure_ascii=False),
                        normalized["answer"],
                        normalized["ground_truth"],
                        json.dumps(normalized["metadata"], ensure_ascii=False),
                        json.dumps(normalized["evaluation"], ensure_ascii=False),
                        "parsed",
                        datetime.now().isoformat()
                    ))
                    imported += 1
                    
                except Exception as e:
                    errors += 1
                    print(f"导入错误: {e}")
            
            conn.commit()
        
        return {
            "imported": imported,
            "skipped": skipped,
            "errors": errors,
            "source": source or "all"
        }
    
    def import_from_raw_data(self, 
                              status: str = "pending",
                              batch_size: int = 100) -> dict:
        """
        从 raw_data 表导入到 processed_data
        """
        imported = 0
        skipped = 0
        errors = 0
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            raw_rows = conn.execute(
                """SELECT * FROM raw_data 
                   WHERE status = ?
                   AND id NOT IN (
                       SELECT raw_id FROM processed_data 
                       WHERE raw_id IS NOT NULL
                   )
                   LIMIT ?""",
                (status, batch_size)
            ).fetchall()
            
            for row in raw_rows:
                try:
                    raw_dict = dict(row)
                    # 解析 raw_content
                    raw_content = raw_dict.get("raw_content", "{}")
                    try:
                        content = json.loads(raw_content)
                    except:
                        content = {"question": raw_content, "answer": ""}
                    
                    # 合并元数据
                    metadata = raw_dict.get("metadata", "{}")
                    try:
                        meta = json.loads(metadata)
                    except:
                        meta = {}
                    
                    record = {
                        "id": raw_dict.get("id"),
                        "question": content.get("question", ""),
                        "answer": content.get("answer", ""),
                        "contexts": content.get("contexts", []),
                        "ground_truth": content.get("ground_truth", ""),
                        "domain": content.get("domain", meta.get("domain", "其他")),
                        "difficulty": content.get("difficulty", meta.get("difficulty", "medium")),
                        "metadata": {
                            "raw_id": raw_dict.get("id"),
                            "timestamp": raw_dict.get("timestamp"),
                            "data_source": raw_dict.get("source_type", "unknown"),
                            "batch_id": raw_dict.get("upload_batch", "import_" + datetime.now().strftime("%Y%m%d")),
                            "original_format": raw_dict.get("original_format", "json"),
                        }
                    }
                    
                    normalized = self.validator.normalize(record)
                    is_valid, err_list = self.validator.validate(normalized)
                    if not is_valid:
                        skipped += 1
                        # 更新 raw_data 状态为 error
                        conn.execute(
                            "UPDATE raw_data SET status = 'error', error_message = ? WHERE id = ?",
                            (json.dumps(err_list, ensure_ascii=False), raw_dict.get("id"))
                        )
                        continue
                    
                    conn.execute("""
                        INSERT INTO processed_data 
                        (id, raw_id, question, question_type, domain, difficulty,
                         contexts, answer, ground_truth, metadata, evaluation, 
                         processing_stage, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        normalized["id"], raw_dict.get("id"),
                        normalized["question"], normalized["question_type"],
                        normalized["domain"], normalized["difficulty"],
                        json.dumps(normalized["contexts"], ensure_ascii=False),
                        normalized["answer"], normalized["ground_truth"],
                        json.dumps(normalized["metadata"], ensure_ascii=False),
                        json.dumps(normalized["evaluation"], ensure_ascii=False),
                        "parsed", datetime.now().isoformat()
                    ))
                    
                    # 更新 raw_data 状态
                    conn.execute(
                        "UPDATE raw_data SET status = 'parsed' WHERE id = ?",
                        (raw_dict.get("id"),)
                    )
                    
                    imported += 1
                    
                except Exception as e:
                    errors += 1
                    conn.execute(
                        "UPDATE raw_data SET status = 'error', error_message = ? WHERE id = ?",
                        (str(e), raw_dict.get("id"))
                    )
            
            conn.commit()
        
        return {
            "imported": imported,
            "skipped": skipped,
            "errors": errors,
            "source": f"raw_data (status={status})"
        }


# ==================== 3.3 数据解析 ====================

class DataParser:
    """数据解析器 — 清洗、验证、质量评分"""
    
    def __init__(self):
        self.min_question_length = 5
        self.max_question_length = 2000
    
    def parse(self, record: dict) -> dict:
        """
        解析单条记录，返回解析结果
        
        Returns:
            {
                "is_valid": bool,
                "record": dict,  # 解析后的记录
                "quality_score": float,
                "errors": List[str],
                "warnings": List[str]
            }
        """
        errors = []
        warnings = []
        
        # 1. 清洗问题
        question = self._clean_text(record.get("question", ""))
        if len(question) < self.min_question_length:
            errors.append(f"问题过短: {len(question)} 字符")
        if len(question) > self.max_question_length:
            warnings.append(f"问题过长: {len(question)} 字符，已截断")
            question = question[:self.max_question_length]
        
        # 2. 清洗上下文
        contexts = self._clean_contexts(record.get("contexts", []))
        if not contexts:
            warnings.append("上下文为空")
        
        # 3. 清洗答案
        answer = self._clean_text(record.get("answer", ""))
        
        # 4. 清洗标准答案
        ground_truth = self._clean_text(record.get("ground_truth", "")) or None
        
        # 5. 质量评分
        quality_score = self._calculate_quality(question, contexts, answer, ground_truth)
        
        is_valid = len(errors) == 0 and quality_score > 0.3
        
        # 更新 metadata 中的 quality_score
        metadata = record.get("metadata", {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except:
                metadata = {}
        metadata["quality_score"] = quality_score
        metadata["parsed_at"] = datetime.now().isoformat()
        
        parsed_record = {
            "id": record.get("id"),
            "question": question,
            "question_type": record.get("question_type", "factual"),
            "domain": record.get("domain", "其他"),
            "difficulty": record.get("difficulty", "medium"),
            "contexts": contexts,
            "answer": answer,
            "ground_truth": ground_truth,
            "metadata": metadata,
            "evaluation": record.get("evaluation", {}),
            "processing_stage": "validated" if is_valid else "invalid",
        }
        
        return {
            "is_valid": is_valid,
            "record": parsed_record,
            "quality_score": quality_score,
            "errors": errors,
            "warnings": warnings
        }
    
    def parse_all(self, db_path: str = "data/rag_data.db", 
                  stage: str = "parsed") -> dict:
        """
        批量解析 processed_data 中指定 stage 的数据
        """
        parsed_count = 0
        invalid_count = 0
        
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM processed_data WHERE processing_stage = ?",
                (stage,)
            ).fetchall()
            
            for row in rows:
                record = dict(row)
                # 解析 JSON 字段
                for field in ["contexts", "metadata", "evaluation"]:
                    val = record.get(field, "{}")
                    if isinstance(val, str):
                        try:
                            record[field] = json.loads(val)
                        except:
                            record[field] = {}
                
                result = self.parse(record)
                
                # 更新数据库
                conn.execute("""
                    UPDATE processed_data SET
                        question = ?,
                        question_type = ?,
                        domain = ?,
                        difficulty = ?,
                        contexts = ?,
                        answer = ?,
                        ground_truth = ?,
                        metadata = ?,
                        evaluation = ?,
                        processing_stage = ?
                    WHERE id = ?
                """, (
                    result["record"]["question"],
                    result["record"]["question_type"],
                    result["record"]["domain"],
                    result["record"]["difficulty"],
                    json.dumps(result["record"]["contexts"], ensure_ascii=False),
                    result["record"]["answer"],
                    result["record"]["ground_truth"],
                    json.dumps(result["record"]["metadata"], ensure_ascii=False),
                    json.dumps(result["record"]["evaluation"], ensure_ascii=False),
                    result["record"]["processing_stage"],
                    record["id"]
                ))
                
                if result["is_valid"]:
                    parsed_count += 1
                else:
                    invalid_count += 1
            
            conn.commit()
        
        return {
            "parsed": parsed_count,
            "invalid": invalid_count,
            "total": parsed_count + invalid_count
        }
    
    def _clean_text(self, text: str) -> str:
        """文本清洗"""
        if not text:
            return ""
        # 去除多余空白
        text = re.sub(r'\s+', ' ', text)
        # 去除特殊控制字符
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
        # 去除零宽字符
        text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
        return text.strip()
    
    def _clean_contexts(self, contexts) -> List[dict]:
        """清洗上下文"""
        if not contexts:
            return []
        
        if isinstance(contexts, str):
            try:
                contexts = json.loads(contexts)
            except:
                contexts = []
        
        if not isinstance(contexts, list):
            contexts = [contexts]
        
        cleaned = []
        for ctx in contexts:
            if isinstance(ctx, dict):
                content = self._clean_text(ctx.get("content", ""))
                if len(content) > 10:
                    cleaned.append({
                        "content": content,
                        "source": str(ctx.get("source", "unknown")),
                        "page": ctx.get("page", 1),
                        "relevance_score": ctx.get("relevance_score", 0.9)
                    })
            elif isinstance(ctx, str) and len(ctx.strip()) > 10:
                cleaned.append({
                    "content": self._clean_text(ctx),
                    "source": "unknown",
                    "page": 1,
                    "relevance_score": 0.9
                })
        
        return cleaned
    
    def _calculate_quality(self, question: str, contexts: List[dict], 
                          answer: str, ground_truth: Optional[str]) -> float:
        """计算质量评分 (0-1)"""
        scores = []
        
        # 问题质量 (0-0.3)
        q_len = len(question)
        if q_len < 10:
            q_score = 0.1
        elif q_len < 50:
            q_score = 0.2
        else:
            q_score = min(0.3, q_len / 300)
        scores.append(q_score)
        
        # 上下文质量 (0-0.3)
        if contexts:
            total_ctx_len = sum(len(c.get("content", "")) for c in contexts)
            ctx_score = min(0.3, total_ctx_len / 1000)
        else:
            ctx_score = 0.0
        scores.append(ctx_score)
        
        # 答案质量 (0-0.2)
        a_len = len(answer)
        if a_len > 50:
            a_score = min(0.2, a_len / 500)
        else:
            a_score = 0.0
        scores.append(a_score)
        
        # 标准答案质量 (0-0.2)
        gt_score = 0.2 if ground_truth and len(ground_truth) > 10 else 0.0
        scores.append(gt_score)
        
        return sum(scores)


# ==================== 3.4 数据集搭建 ====================

class TestSetBuilder:
    """测试集构建器 — 分层采样、去重、导出"""
    
    def __init__(self, db_path: str = "data/rag_data.db"):
        self.db_path = db_path
        self._ensure_registry_table()
    
    def _ensure_registry_table(self):
        """确保测试集注册表存在"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS testset_versions (
                    version_id TEXT PRIMARY KEY,
                    name TEXT,
                    description TEXT,
                    created_at TEXT,
                    size INTEGER,
                    domain_distribution TEXT,
                    difficulty_distribution TEXT,
                    type_distribution TEXT,
                    avg_quality_score REAL,
                    has_ground_truth_ratio REAL,
                    status TEXT,
                    file_paths TEXT
                )
            """)
    
    def build_testset(self,
                      output_prefix: str = "data/testset",
                      golden_size: int = 20,
                      validation_size: int = 50,
                      stress_size: int = 10,
                      min_quality: float = 0.5) -> dict:
        """
        构建分层测试集
        
        Returns:
            {
                "golden": path,
                "validation": path,
                "stress": path,
                "stats": {...}
            }
        """
        # 读取已验证的高质量数据
        records = self._load_validated_records(min_quality)
        
        if not records:
            return {"error": "没有足够的验证数据，请先运行数据解析"}
        
        # 去重
        records = self._deduplicate(records, threshold=0.9)
        
        # 分类
        classified = self._classify_records(records)
        
        # 分层采样
        golden = self._sample_stratified(classified, golden_size, priority="quality")
        validation = self._sample_stratified(classified, validation_size, 
                                                exclude=golden, priority="balanced")
        stress = self._build_stress_set(records, stress_size)
        
        # 导出
        paths = {}
        paths["golden"] = self._export_jsonl(golden, f"{output_prefix}_golden.jsonl")
        paths["validation"] = self._export_jsonl(validation, f"{output_prefix}_validation.jsonl")
        paths["stress"] = self._export_jsonl(stress, f"{output_prefix}_stress.jsonl")
        
        # 标记为测试集
        all_ids = [r["id"] for r in golden + validation + stress]
        self._mark_as_testset(all_ids)
        
        # 计算统计
        stats = self._calculate_stats(golden, validation, stress)
        
        # 注册版本
        version_id = f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._register_version(version_id, stats, paths)
        
        return {
            "version_id": version_id,
            "golden": paths["golden"],
            "validation": paths["validation"],
            "stress": paths["stress"],
            "stats": stats
        }
    
    def _load_validated_records(self, min_quality: float = 0.5) -> List[dict]:
        """加载已验证的高质量数据"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT * FROM processed_data 
                   WHERE processing_stage = 'validated'
                   LIMIT 1000"""
            ).fetchall()
            
            records = []
            for row in rows:
                record = dict(row)
                # 解析 JSON 字段
                for field in ["contexts", "metadata", "evaluation"]:
                    val = record.get(field, "{}")
                    if isinstance(val, str):
                        try:
                            record[field] = json.loads(val)
                        except:
                            record[field] = {}
                records.append(record)
            
            return records
    
    def _deduplicate(self, records: List[dict], threshold: float = 0.9) -> List[dict]:
        """基于问题文本相似度去重"""
        unique = []
        for r in records:
            is_dup = False
            for u in unique:
                sim = self._text_similarity(r["question"], u["question"])
                if sim > threshold:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(r)
        return unique
    
    def _text_similarity(self, a: str, b: str) -> float:
        """Jaccard 相似度"""
        set_a = set(a.lower().split())
        set_b = set(b.lower().split())
        if not set_a or not set_b:
            return 0.0
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union)
    
    def _classify_records(self, records: List[dict]) -> Dict[str, List[dict]]:
        """按领域分类"""
        classified = defaultdict(list)
        for r in records:
            domain = r.get("domain", "其他")
            classified[domain].append(r)
        return dict(classified)
    
    def _sample_stratified(self, classified: Dict[str, List[dict]], 
                           total_size: int, 
                           exclude: Optional[List[dict]] = None,
                           priority: str = "balanced") -> List[dict]:
        """分层采样"""
        exclude_ids = {r["id"] for r in (exclude or [])}
        
        available = {}
        for domain, records in classified.items():
            available[domain] = [r for r in records if r["id"] not in exclude_ids]
        
        domains = list(available.keys())
        if not domains:
            return []
        
        per_domain = max(1, total_size // len(domains))
        
        sampled = []
        for domain in domains:
            pool = available[domain]
            if not pool:
                continue
            
            if priority == "quality":
                # 按质量排序取前N
                pool = sorted(pool, key=lambda r: 
                    r.get("metadata", {}).get("quality_score", 0), reverse=True)
                n = min(per_domain, len(pool))
                sampled.extend(pool[:n])
            else:
                n = min(per_domain, len(pool))
                sampled.extend(random.sample(pool, n))
        
        # 补足数量
        all_available = [r for records in available.values() for r in records]
        remaining = [r for r in all_available if r["id"] not in {s["id"] for s in sampled}]
        
        if len(sampled) < total_size and remaining:
            need = total_size - len(sampled)
            sampled.extend(random.sample(remaining, min(need, len(remaining))))
        
        return sampled[:total_size]
    
    def _build_stress_set(self, records: List[dict], size: int) -> List[dict]:
        """构建压力测试集 — 边界场景"""
        stress_cases = []
        
        for r in records:
            metadata = r.get("metadata", {})
            question = r.get("question", "")
            contexts = r.get("contexts", [])
            answer = r.get("answer", "")
            
            # 边界条件：短问题、长问题、无上下文、无答案
            if (len(question) < 10 or
                len(question) > 500 or
                not contexts or
                not answer):
                stress_cases.append(r)
        
        # 如果没有足够的边界数据，随机选一些补充
        if len(stress_cases) < size:
            other = [r for r in records if r not in stress_cases]
            need = size - len(stress_cases)
            if other:
                stress_cases.extend(random.sample(other, min(need, len(other))))
        
        return stress_cases[:size]
    
    def _export_jsonl(self, records: List[dict], path: str) -> str:
        """导出 JSONL 格式"""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            for r in records:
                export_record = {
                    "id": r["id"],
                    "question": r["question"],
                    "question_type": r.get("question_type", "factual"),
                    "domain": r.get("domain", "其他"),
                    "difficulty": r.get("difficulty", "medium"),
                    "contexts": r.get("contexts", []),
                    "answer": r.get("answer", ""),
                    "ground_truth": r.get("ground_truth", ""),
                    "metadata": r.get("metadata", {})
                }
                f.write(json.dumps(export_record, ensure_ascii=False) + '\n')
        
        return path
    
    def _mark_as_testset(self, ids: List[str]):
        """标记为测试集数据"""
        with sqlite3.connect(self.db_path) as conn:
            for id_ in ids:
                conn.execute(
                    "UPDATE processed_data SET processing_stage = 'testset' WHERE id = ?",
                    (id_,)
                )
            conn.commit()
    
    def _calculate_stats(self, golden: List[dict], validation: List[dict], 
                        stress: List[dict]) -> dict:
        """计算测试集统计"""
        all_records = golden + validation + stress
        
        domains = defaultdict(int)
        difficulties = defaultdict(int)
        types = defaultdict(int)
        quality_scores = []
        has_gt = 0
        
        for r in all_records:
            domains[r.get("domain", "其他")] += 1
            difficulties[r.get("difficulty", "medium")] += 1
            types[r.get("question_type", "factual")] += 1
            
            metadata = r.get("metadata", {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    metadata = {}
            quality_scores.append(metadata.get("quality_score", 0))
            
            if r.get("ground_truth"):
                has_gt += 1
        
        return {
            "total": len(all_records),
            "golden": len(golden),
            "validation": len(validation),
            "stress": len(stress),
            "domain_distribution": dict(domains),
            "difficulty_distribution": dict(difficulties),
            "type_distribution": dict(types),
            "avg_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            "has_ground_truth_ratio": has_gt / len(all_records) if all_records else 0,
        }
    
    def _register_version(self, version_id: str, stats: dict, paths: dict):
        """注册测试集版本"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO testset_versions 
                (version_id, name, description, created_at, size,
                 domain_distribution, difficulty_distribution, type_distribution,
                 avg_quality_score, has_ground_truth_ratio, status, file_paths)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                version_id,
                f"TestSet {version_id}",
                "自动构建的测试集",
                datetime.now().isoformat(),
                stats["total"],
                json.dumps(stats["domain_distribution"]),
                json.dumps(stats["difficulty_distribution"]),
                json.dumps(stats["type_distribution"]),
                stats["avg_quality_score"],
                stats["has_ground_truth_ratio"],
                "active",
                json.dumps(paths)
            ))
            conn.commit()


# ==================== 测试集注册表查询 ====================

class TestSetRegistry:
    """测试集注册表查询"""
    
    def __init__(self, db_path: str = "data/rag_data.db"):
        self.db_path = db_path
    
    def get_active_version(self) -> Optional[dict]:
        """获取当前活跃版本"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM testset_versions WHERE status = 'active' ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            
            if not row:
                return None
            
            return dict(row)
    
    def list_versions(self, limit: int = 10) -> List[dict]:
        """列出所有版本"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM testset_versions ORDER BY created_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [dict(row) for row in rows]


# ==================== 主入口 ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG 测试集搭建工具")
    parser.add_argument("--db", default="data/rag_data.db", help="数据库路径")
    parser.add_argument("--action", required=True, 
                        choices=["import", "parse", "build", "validate", "pipeline"],
                        help="操作类型")
    parser.add_argument("--source", default="all", help="导入来源过滤")
    parser.add_argument("--output", default="data/testset", help="输出前缀")
    parser.add_argument("--golden-size", type=int, default=20, help="Golden 集大小")
    parser.add_argument("--validation-size", type=int, default=50, help="Validation 集大小")
    parser.add_argument("--stress-size", type=int, default=10, help="Stress 集大小")
    
    args = parser.parse_args()
    
    if args.action == "import":
        importer = DataImporter(args.db)
        if args.source == "raw_data":
            result = importer.import_from_raw_data()
        else:
            result = importer.import_from_conversations(source=None if args.source == "all" else args.source)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    
    elif args.action == "parse":
        parser = DataParser()
        result = parser.parse_all(args.db)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    
    elif args.action == "build":
        builder = TestSetBuilder(args.db)
        result = builder.build_testset(
            output_prefix=args.output,
            golden_size=args.golden_size,
            validation_size=args.validation_size,
            stress_size=args.stress_size
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    
    elif args.action == "validate":
        # 验证单条记录
        validator = DataFormatValidator()
        test_record = {
            "question": "什么是对角线论证？",
            "answer": "对角线论证是一种证明方法...",
            "contexts": ["对角线论证由康托尔提出..."],
            "domain": "数学",
            "difficulty": "medium"
        }
        normalized = validator.normalize(test_record)
        is_valid, errors = validator.validate(normalized)
        print(f"验证结果: {is_valid}")
        if errors:
            print(f"错误: {errors}")
    
    elif args.action == "pipeline":
        # 完整流程
        print("=== 1. 数据导入 ===")
        importer = DataImporter(args.db)
        import_result = importer.import_from_conversations()
        print(json.dumps(import_result, ensure_ascii=False, indent=2))
        
        print("\n=== 2. 数据解析 ===")
        parser = DataParser()
        parse_result = parser.parse_all(args.db)
        print(json.dumps(parse_result, ensure_ascii=False, indent=2))
        
        print("\n=== 3. 测试集搭建 ===")
        builder = TestSetBuilder(args.db)
        build_result = builder.build_testset(output_prefix=args.output)
        print(json.dumps(build_result, ensure_ascii=False, indent=2))
