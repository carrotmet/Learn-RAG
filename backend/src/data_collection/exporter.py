"""
RAG 教学项目 — 数据导出模块

参考: /home/ubuntu/.openclaw/workspace/RAG教学/docs/监控指标体系搭建规划.md 2.4 节

功能:
- 从 SQLite 导出数据到 JSONL（标准格式）
- 支持按条件过滤导出
- 支持测试集格式导出（RAGAS 兼容）
- 批量导出工具

使用方式:
    from data_collection.exporter import DataExporter
    
    exporter = DataExporter("data/rag_data.db")
    
    # 导出所有对话
    exporter.export_conversations("data/export_conversations.jsonl")
    
    # 导出测试集格式（RAGAS 兼容）
    exporter.export_testset("data/testset.jsonl", min_quality=0.7)
    
    # 按条件导出
    exporter.export_by_condition("data/export.jsonl", table="conversations", 
                                  conditions="source = 'online_api' AND timestamp > '2026-06-01'")
"""

import sqlite3
import json
import os
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

# 使用绝对导入（支持直接运行和包导入两种模式）
try:
    from sqlite_store import SQLiteCollector
except ImportError:
    from .sqlite_store import SQLiteCollector


class DataExporter:
    """数据导出器 — 支持多种导出格式和条件"""
    
    def __init__(self, db_path: str = "data/rag_data.db"):
        self.db_path = db_path
        self.collector = SQLiteCollector(db_path)
    
    def export_conversations(
        self,
        output_path: str,
        conditions: Optional[str] = None,
        limit: int = 10000,
        include_details: bool = False
    ) -> int:
        """导出对话记录到 JSONL
        
        Args:
            output_path: 输出文件路径
            conditions: SQL WHERE 条件（如 "source = 'online_api'"）
            limit: 最大导出数量
            include_details: 是否包含检索日志和 LLM 调用详情
        
        Returns:
            导出的记录数
        """
        return self.collector.export_to_jsonl(
            output_path=output_path,
            table="conversations",
            conditions=conditions,
            limit=limit
        )
    
    def export_processed_data(
        self,
        output_path: str,
        conditions: Optional[str] = None,
        limit: int = 10000
    ) -> int:
        """导出解析后的标准格式数据到 JSONL
        
        这是测试集搭建的核心输入，字段符合统一数据格式标准。
        """
        return self.collector.export_to_jsonl(
            output_path=output_path,
            table="processed_data",
            conditions=conditions,
            limit=limit
        )
    
    def export_raw_data(
        self,
        output_path: str,
        conditions: Optional[str] = None,
        limit: int = 10000
    ) -> int:
        """导出原始数据到 JSONL"""
        return self.collector.export_to_jsonl(
            output_path=output_path,
            table="raw_data",
            conditions=conditions,
            limit=limit
        )
    
    def export_testset(
        self,
        output_path: str,
        testset_type: str = "validation",
        min_quality: Optional[float] = None,
        domain: Optional[str] = None,
        difficulty: Optional[str] = None,
        limit: int = 1000
    ) -> int:
        """导出测试集（RAGAS 兼容格式）
        
        输出格式符合 RAGAS 评估要求：
        {
            "question": "...",
            "answer": "...",
            "contexts": ["...", "..."],
            "ground_truth": "..."
        }
        
        Args:
            output_path: 输出文件路径
            testset_type: 测试集类型 golden|validation|stress
            min_quality: 最低质量分数
            domain: 按领域过滤
            difficulty: 按难度过滤
            limit: 最大导出数量
        """
        # 构建条件
        conditions_list = []
        if min_quality is not None:
            conditions_list.append(
                f"json_extract(metadata, '$.quality_score') >= {min_quality}"
            )
        if domain:
            conditions_list.append(f"domain = '{domain}'")
        if difficulty:
            conditions_list.append(f"difficulty = '{difficulty}'")
        
        conditions = " AND ".join(conditions_list) if conditions_list else None
        
        # 导出 processed_data 表
        raw_count = self.collector.export_to_jsonl(
            output_path=output_path,
            table="processed_data",
            conditions=conditions,
            limit=limit
        )
        
        if raw_count == 0:
            return 0
        
        # 转换为 RAGAS 兼容格式
        ragas_records = []
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line.strip())
                
                # 解析 contexts
                contexts = []
                if record.get('contexts'):
                    try:
                        ctx_data = json.loads(record['contexts']) if isinstance(record['contexts'], str) else record['contexts']
                        if isinstance(ctx_data, list):
                            contexts = [c.get('content', c) if isinstance(c, dict) else c for c in ctx_data]
                    except:
                        contexts = []
                
                # 构建 RAGAS 格式记录
                ragas_record = {
                    "id": record.get('id', str(uuid.uuid4())),
                    "question": record.get('question', ''),
                    "answer": record.get('answer', ''),
                    "contexts": contexts,
                    "ground_truth": record.get('ground_truth', ''),
                    "metadata": {
                        "domain": record.get('domain', '其他'),
                        "difficulty": record.get('difficulty', 'medium'),
                        "question_type": record.get('question_type', 'factual'),
                        "source": record.get('metadata', {}).get('data_source', 'unknown') if isinstance(record.get('metadata'), dict) else 'unknown',
                        "testset_type": testset_type,
                    }
                }
                ragas_records.append(ragas_record)
        
        # 重写文件为 RAGAS 格式
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in ragas_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        return len(ragas_records)
    
    def export_by_sql(
        self,
        output_path: str,
        query: str,
        params: Optional[tuple] = None
    ) -> int:
        """通过自定义 SQL 查询导出数据
        
        Args:
            output_path: 输出文件路径
            query: SQL SELECT 查询（结果会被序列化为 JSON）
            params: 查询参数
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params or ()).fetchall()
        
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
    
    def get_export_preview(
        self,
        table: str = "conversations",
        conditions: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """获取导出预览（前 N 条记录）
        
        用于在导出前确认数据内容和格式。
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            where = f"WHERE {conditions}" if conditions else ""
            
            # 根据表选择排序字段
            if table in ['conversations', 'retrieval_logs', 'llm_calls', 'user_feedback', 'raw_data']:
                order_by = "ORDER BY timestamp DESC"
            elif table == 'processed_data':
                order_by = "ORDER BY created_at DESC"
            else:
                order_by = ""
            
            query = f"SELECT * FROM {table} {where} {order_by} LIMIT ?"
            rows = conn.execute(query, (limit,)).fetchall()
            
            result = []
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
                result.append(record)
            
            return result
    
    def get_export_summary(
        self,
        table: str = "conversations",
        conditions: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取导出摘要统计
        
        用于在导出前了解数据规模和分布。
        """
        with sqlite3.connect(self.db_path) as conn:
            where = f"WHERE {conditions}" if conditions else ""
            
            # 总记录数
            total = conn.execute(f"SELECT COUNT(*) FROM {table} {where}").fetchone()[0]
            
            # 时间范围（根据表选择时间字段）
            time_field = "timestamp" if table in ['conversations', 'retrieval_logs', 'llm_calls', 'user_feedback', 'raw_data'] else "created_at"
            time_range = conn.execute(
                f"SELECT MIN({time_field}), MAX({time_field}) FROM {table} {where}"
            ).fetchone()
            
            # 字段统计（仅对 conversations 表）
            field_stats = {}
            if table == "conversations":
                # 数据来源分布
                source_dist = conn.execute(
                    f"SELECT source, COUNT(*) FROM {table} {where} GROUP BY source"
                ).fetchall()
                field_stats['source_distribution'] = {row[0]: row[1] for row in source_dist}
                
                # 处理阶段分布
                stage_dist = conn.execute(
                    f"SELECT processing_stage, COUNT(*) FROM {table} {where} GROUP BY processing_stage"
                ).fetchall()
                field_stats['stage_distribution'] = {row[0]: row[1] for row in stage_dist}
            
            return {
                "table": table,
                "total_records": total,
                "time_range": {
                    "min": time_range[0],
                    "max": time_range[1]
                } if time_range else None,
                "conditions": conditions,
                "field_stats": field_stats
            }


# CLI 工具支持
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG 数据导出工具")
    parser.add_argument("--db", default="data/rag_data.db", help="数据库路径")
    parser.add_argument("--output", required=True, help="输出文件路径")
    parser.add_argument("--table", default="conversations", 
                       choices=["conversations", "processed_data", "raw_data", "retrieval_logs", "llm_calls", "user_feedback"],
                       help="要导出的表")
    parser.add_argument("--conditions", help="SQL WHERE 条件")
    parser.add_argument("--limit", type=int, default=10000, help="最大导出数量")
    parser.add_argument("--preview", action="store_true", help="仅预览前 5 条")
    parser.add_argument("--summary", action="store_true", help="仅显示摘要统计")
    parser.add_argument("--testset", action="store_true", help="导出为 RAGAS 测试集格式")
    
    args = parser.parse_args()
    
    exporter = DataExporter(args.db)
    
    if args.preview:
        preview = exporter.get_export_preview(args.table, args.conditions, limit=5)
        print(json.dumps(preview, ensure_ascii=False, indent=2))
    elif args.summary:
        summary = exporter.get_export_summary(args.table, args.conditions)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.testset:
        count = exporter.export_testset(args.output, limit=args.limit)
        print(f"已导出 {count} 条测试集记录到 {args.output}")
    else:
        count = exporter.export_conversations(args.output, args.conditions, args.limit)
        print(f"已导出 {count} 条记录到 {args.output}")
