"""
RAG 教学项目 — 数据采集配置

参考: /home/ubuntu/.openclaw/workspace/RAG教学/docs/监控指标体系搭建规划.md 2.5 节
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CollectionConfig:
    """教学项目采集配置 — 轻量简洁
    
    设计原则:
    - 无需 Kafka 等重型中间件
    - SQLite 直写，简单可靠
    - 支持同步/异步写入模式
    """
    
    # SQLite 配置
    db_path: str = "data/rag_data.db"
    
    # 在线采集
    online_enabled: bool = True
    sync_write: bool = True  # True=同步, False=异步（后台线程）
    
    # 离线上传
    upload_max_size_mb: int = 50
    upload_allowed_formats: List[str] = field(default_factory=lambda: [
        "json", "jsonl", "csv", "xlsx"
    ])
    
    # 数据质量校验
    require_question: bool = True
    min_question_length: int = 5
    max_question_length: int = 2000
    
    # 批量导出
    export_batch_size: int = 1000
    
    def validate_format(self, filename: str) -> bool:
        """验证文件格式是否允许"""
        ext = filename.lower().split('.')[-1] if '.' in filename else ''
        # xlsx 特殊处理
        if filename.lower().endswith('.xlsx'):
            ext = 'xlsx'
        return ext in self.upload_allowed_formats
    
    def validate_size(self, size_bytes: int) -> bool:
        """验证文件大小是否超限"""
        max_bytes = self.upload_max_size_mb * 1024 * 1024
        return size_bytes <= max_bytes
