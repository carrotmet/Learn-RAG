"""全文通道：基于 SQLite FTS5"""

import os
import sqlite3
from typing import List, Dict

from hybrid.config import DEFAULT_DB_PATH


class FTSChannel:
    """全文检索通道（SQLite FTS5）
    
    使用 Porter + Unicode61 分词器，支持中文。
    与 DocumentStore 共用同一个 SQLite 数据库文件。
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = DEFAULT_DB_PATH
        self.db_path = db_path
        self._ensure_table()

    def _ensure_table(self):
        """确保 FTS5 虚拟表存在"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS doc_fts USING fts5(
                    content,
                    derivative_id UNINDEXED,
                    parent_chunk_id UNINDEXED,
                    strategy UNINDEXED,
                    tokenize='porter unicode61'
                )
            """)

    def add(self, items: List[Dict]):
        """添加文档到 FTS5 索引
        
        Args:
            items: [{"content": str, "metadata": {derivative_id, parent_chunk_id, strategy}}, ...]
        """
        with sqlite3.connect(self.db_path) as conn:
            for i in items:
                meta = i["metadata"]
                conn.execute("""
                    INSERT INTO doc_fts (content, derivative_id, parent_chunk_id, strategy)
                    VALUES (?, ?, ?, ?)
                """, (
                    i["content"],
                    meta.get("derivative_id", 0),
                    meta.get("parent_chunk_id", 0),
                    meta.get("strategy", "standard"),
                ))

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """FTS5 全文检索
        
        Returns:
            [{"content": str, "score": float, "parent_chunk_id": int,
              "derivative_id": int, "channel": "fts"}, ...]
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            # bm25 越小排名越靠前
            rows = conn.execute("""
                SELECT content, derivative_id, parent_chunk_id, strategy, bm25(doc_fts) as bm25
                FROM doc_fts 
                WHERE doc_fts MATCH ? 
                ORDER BY bm25(doc_fts) ASC 
                LIMIT ?
            """, (query, k)).fetchall()

            if not rows:
                return []

            # 将 bm25 转换为 0-1 分数（越大越好）
            max_bm25 = max(r["bm25"] for r in rows) if rows else 1
            min_bm25 = min(r["bm25"] for r in rows) if rows else 0
            
            results = []
            for r in rows:
                # 归一化：bm25 越小分数越高
                if max_bm25 == min_bm25:
                    score = 1.0
                else:
                    score = (max_bm25 - r["bm25"]) / (max_bm25 - min_bm25 + 1e-6)
                results.append({
                    "content": r["content"],
                    "score": round(score, 4),
                    "parent_chunk_id": r["parent_chunk_id"],
                    "derivative_id": r["derivative_id"],
                    "channel": "fts",
                    "strategy": r["strategy"],
                })
            return results

    def clear(self):
        """清空 FTS5 索引"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM doc_fts")

    def count(self) -> int:
        """索引中文档数量"""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM doc_fts").fetchone()
            return row[0] if row else 0
