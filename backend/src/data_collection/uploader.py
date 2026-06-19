"""
RAG 教学项目 — 离线上传解析器

参考: /home/ubuntu/.openclaw/workspace/RAG教学/docs/监控指标体系搭建规划.md 2.2 节

功能:
- 支持 JSON/JSONL/CSV/Excel 多格式解析
- 字段映射（原始字段 → 标准字段）
- 基础数据校验
- 写入 SQLite raw_data 表

使用方式:
    from data_collection.uploader import DataUploader
    
    uploader = DataUploader("data/rag_data.db")
    
    # 上传 JSON 文件
    result = uploader.upload_file("data/questions.json", source_type="offline_upload")
    
    # 上传 JSONL 字符串
    result = uploader.upload_jsonl(jsonl_content, source_type="manual_entry")
"""

import json
import csv
import io
import os
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# 使用绝对导入（支持直接运行和包导入两种模式）
try:
    from sqlite_store import SQLiteCollector
    from config import CollectionConfig
except ImportError:
    from .sqlite_store import SQLiteCollector
    from .config import CollectionConfig


class DataUploader:
    """离线上传解析器 — 支持多格式文件导入"""
    
    # 标准字段映射（原始字段名 → 标准字段名）
    FIELD_MAPPING = {
        # 常见变体 → 标准字段
        'q': 'question',
        'query': 'question',
        '问题': 'question',
        'question': 'question',
        
        'a': 'answer',
        'response': 'answer',
        '回答': 'answer',
        'answer': 'answer',
        '系统回答': 'answer',
        
        'gt': 'ground_truth',
        'truth': 'ground_truth',
        '标准答案': 'ground_truth',
        'ground_truth': 'ground_truth',
        'expected_answer': 'ground_truth',
        
        'ctx': 'contexts',
        'context': 'contexts',
        'docs': 'contexts',
        'documents': 'contexts',
        '上下文': 'contexts',
        'contexts': 'contexts',
        
        'src': 'source',
        '来源': 'source',
        'source': 'source',
        'data_source': 'source',
    }
    
    def __init__(self, db_path: str = "data/rag_data.db", config: Optional[CollectionConfig] = None):
        self.config = config or CollectionConfig()
        self.collector = SQLiteCollector(db_path, config)
        self.upload_stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'errors': []
        }
    
    def upload_file(
        self,
        file_path: str,
        source_type: str = "offline_upload",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """上传文件并解析保存
        
        Args:
            file_path: 文件路径
            source_type: 数据来源类型
            metadata: 上传元数据
        
        Returns:
            上传结果统计
        """
        # 验证文件
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        
        if not self.config.validate_format(filename):
            raise ValueError(f"不支持的文件格式: {filename}")
        
        if not self.config.validate_size(file_size):
            raise ValueError(f"文件过大: {file_size} bytes > {self.config.upload_max_size_mb}MB")
        
        # 生成批次号
        upload_batch = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 根据格式解析
        ext = filename.lower().split('.')[-1]
        if filename.lower().endswith('.xlsx'):
            ext = 'xlsx'
        
        if ext == 'json':
            records = self._parse_json(file_path)
        elif ext == 'jsonl':
            records = self._parse_jsonl(file_path)
        elif ext == 'csv':
            records = self._parse_csv(file_path)
        elif ext == 'xlsx':
            records = self._parse_excel(file_path)
        else:
            raise ValueError(f"不支持的格式: {ext}")
        
        # 保存到 raw_data
        saved_count = 0
        for record in records:
            try:
                # 字段映射
                mapped = self._map_fields(record)
                
                # 基础校验
                self._validate_record(mapped)
                
                # 保存
                self.collector.save_raw_data(
                    raw_content=mapped,
                    source_type=source_type,
                    original_format=ext,
                    upload_batch=upload_batch,
                    metadata={
                        'uploader': metadata.get('uploader', 'unknown') if metadata else 'unknown',
                        'upload_time': datetime.now().isoformat(),
                        'original_filename': filename,
                        **(metadata or {})
                    }
                )
                saved_count += 1
                
            except Exception as e:
                self.upload_stats['errors'].append({
                    'record': record,
                    'error': str(e)
                })
        
        self.upload_stats['total'] += len(records)
        self.upload_stats['success'] += saved_count
        self.upload_stats['failed'] += len(records) - saved_count
        
        return {
            'upload_batch': upload_batch,
            'filename': filename,
            'format': ext,
            'total_records': len(records),
            'saved': saved_count,
            'failed': len(records) - saved_count,
            'errors': self.upload_stats['errors'][-5:]  # 最近 5 条错误
        }
    
    def upload_jsonl(
        self,
        content: str,
        source_type: str = "manual_entry",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """直接上传 JSONL 字符串内容
        
        Args:
            content: JSONL 格式字符串（每行一个 JSON 对象）
            source_type: 数据来源类型
            metadata: 上传元数据
        """
        upload_batch = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        records = []
        for line in content.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                self.upload_stats['errors'].append({
                    'line': line[:100],
                    'error': f"JSON 解析失败: {e}"
                })
        
        saved_count = 0
        for record in records:
            try:
                mapped = self._map_fields(record)
                self._validate_record(mapped)
                
                self.collector.save_raw_data(
                    raw_content=mapped,
                    source_type=source_type,
                    original_format='jsonl',
                    upload_batch=upload_batch,
                    metadata={
                        'uploader': metadata.get('uploader', 'manual') if metadata else 'manual',
                        'upload_time': datetime.now().isoformat(),
                        **(metadata or {})
                    }
                )
                saved_count += 1
                
            except Exception as e:
                self.upload_stats['errors'].append({
                    'record': record,
                    'error': str(e)
                })
        
        self.upload_stats['total'] += len(records)
        self.upload_stats['success'] += saved_count
        self.upload_stats['failed'] += len(records) - saved_count
        
        return {
            'upload_batch': upload_batch,
            'format': 'jsonl',
            'total_records': len(records),
            'saved': saved_count,
            'failed': len(records) - saved_count,
            'errors': self.upload_stats['errors'][-5:]
        }
    
    def _parse_json(self, file_path: str) -> List[Dict[str, Any]]:
        """解析 JSON 文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 支持两种格式：对象列表 或 单个对象
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # 可能是 { "data": [...] } 的包装格式
            if 'data' in data and isinstance(data['data'], list):
                return data['data']
            return [data]
        else:
            raise ValueError(f"JSON 格式不支持: {type(data)}")
    
    def _parse_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """解析 JSONL 文件"""
        records = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    self.upload_stats['errors'].append({
                        'line': line_num,
                        'content': line[:100],
                        'error': f"JSONL 解析失败: {e}"
                    })
        return records
    
    def _parse_csv(self, file_path: str) -> List[Dict[str, Any]]:
        """解析 CSV 文件"""
        records = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(dict(row))
        return records
    
    def _parse_excel(self, file_path: str) -> List[Dict[str, Any]]:
        """解析 Excel 文件（需要 openpyxl）"""
        try:
            import openpyxl
        except ImportError:
            raise ImportError("解析 Excel 需要安装 openpyxl: pip install openpyxl")
        
        workbook = openpyxl.load_workbook(file_path)
        sheet = workbook.active
        
        # 读取表头
        headers = [cell.value for cell in sheet[1]]
        
        records = []
        for row in sheet.iter_rows(min_row=2, values_only=True):
            record = {}
            for header, value in zip(headers, row):
                if value is not None:
                    record[header] = value
            records.append(record)
        
        return records
    
    def _map_fields(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """字段映射：原始字段名 → 标准字段名
        
        保留未映射的原始字段，避免数据丢失。
        """
        mapped = {}
        
        for raw_key, raw_value in record.items():
            # 标准化键名（小写，去除空格）
            normalized_key = str(raw_key).lower().strip().replace(' ', '_')
            
            # 查找映射
            if normalized_key in self.FIELD_MAPPING:
                standard_key = self.FIELD_MAPPING[normalized_key]
                mapped[standard_key] = raw_value
            else:
                # 保留原始字段
                mapped[raw_key] = raw_value
        
        return mapped
    
    def _validate_record(self, record: Dict[str, Any]) -> None:
        """基础数据校验
        
        校验规则:
        - question 必填且长度 >= 5
        - contexts 如果是字符串，尝试解析为列表
        """
        # 检查必填字段
        question = record.get('question')
        if not question:
            raise ValueError("缺少必填字段: question")
        
        if len(str(question).strip()) < self.config.min_question_length:
            raise ValueError(f"question 长度需 >= {self.config.min_question_length} 字符")
        
        if len(str(question)) > self.config.max_question_length:
            raise ValueError(f"question 长度需 <= {self.config.max_question_length} 字符")
        
        # 处理 contexts（可能是字符串或列表）
        contexts = record.get('contexts')
        if contexts and isinstance(contexts, str):
            # 尝试解析 JSON 字符串
            try:
                parsed = json.loads(contexts)
                record['contexts'] = parsed
            except (json.JSONDecodeError, TypeError):
                # 作为单元素列表
                record['contexts'] = [contexts]
        
        return True
