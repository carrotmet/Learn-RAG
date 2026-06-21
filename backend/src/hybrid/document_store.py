"""DocumentStore: 统一 Chunk 存储层

负责：
- 文档分块（大chunk 2000字，作为父块）
- 纯数字 ID 管理
- chunks / chunk_derivatives / index_status 表操作
"""

import sqlite3
import json
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from hybrid.config import DEFAULT_DB_PATH


@dataclass
class Chunk:
    """父块（原始chunk）"""
    chunk_id: int
    content: str
    source: str
    page: int = 0
    chunk_index: int = 0


# ── 纯数字 ID 工具 ──────────────────────────────────────────

def parse_derivative_id(did: int) -> Tuple[int, int, int]:
    """解析 derivative_id -> (type_code, chunk_id, seq)
    
    14位格式: {type(1)}{chunk_id(10)}{seq(3)}
    示例: 01000001000 -> (0, 1000001, 0)
    """
    s = str(did).zfill(14)
    return int(s[0]), int(s[1:11]), int(s[11:14])


def make_derivative_id(type_code: int, chunk_id: int, seq: int = 0) -> int:
    """构造 derivative_id
    
    type_code: 0=Summary, 1=ChildChunk, 2=HyDE, 3=Custom
    chunk_id: 父块 ID
    seq: 序号 0-999
    """
    return int(f"{type_code}{chunk_id:010d}{seq:03d}")


# ── DocumentStore ───────────────────────────────────────────

class DocumentStore:
    """统一 Chunk 存储：一次分块（2000字），全策略复用
    
    设计原则：
    - 原始 chunk 永远作为父块（2000字）
    - 子 chunk 是父块内部的二次切分（300字）
    - 派生内容（摘要/HyDE）存储在 chunk_derivatives 表
    """

    def __init__(self, db_path: str = None, chunk_size: int = 2000, chunk_overlap: int = 200):
        if db_path is None:
            db_path = DEFAULT_DB_PATH
        self.db_path = db_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
        )
        self._ensure_tables()

    # ── 表管理 ──────────────────────────────────────────────

    def _ensure_tables(self):
        """确保基础表存在（幂等）"""
        with sqlite3.connect(self.db_path) as conn:
            # 先创建所有业务表
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                    content      TEXT NOT NULL,
                    source       TEXT NOT NULL,
                    page         INTEGER DEFAULT 0,
                    chunk_index  INTEGER,
                    created_at   TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS chunk_derivatives (
                    derivative_id   BIGINT PRIMARY KEY,
                    chunk_id        INTEGER NOT NULL,
                    strategy        TEXT NOT NULL,
                    derivative_type TEXT NOT NULL,
                    content         TEXT NOT NULL,
                    metadata        TEXT,
                    created_at      TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id)
                );

                CREATE TABLE IF NOT EXISTS index_status (
                    derivative_id  BIGINT NOT NULL,
                    strategy       TEXT NOT NULL,
                    channel        TEXT NOT NULL,
                    indexed        INTEGER DEFAULT 0,
                    channel_doc_id TEXT,
                    PRIMARY KEY (derivative_id, strategy, channel)
                );

                CREATE INDEX IF NOT EXISTS idx_derivatives_chunk 
                    ON chunk_derivatives(chunk_id);
                CREATE INDEX IF NOT EXISTS idx_derivatives_strategy 
                    ON chunk_derivatives(strategy);
            """)
            
            # 设置 chunks 表自增 ID 从 1000001 开始
            # sqlite_sequence 表在有 AUTOINCREMENT 的表被写入数据后才会自动创建
            # 策略：插入一条临时记录再删除，触发表创建，然后更新 seq
            conn.execute("INSERT INTO chunks (content, source) VALUES ('__init__', '__init__')")
            conn.execute("DELETE FROM chunks WHERE content = '__init__'")
            
            # 现在 sqlite_sequence 一定存在，更新 seq
            conn.execute("""
                UPDATE sqlite_sequence SET seq = 1000000 WHERE name = 'chunks'
            """)

    # ── 分块与保存 ───────────────────────────────────────────

    def split_and_save(self, docs: List[Document]) -> List[Chunk]:
        """文档 → 分块（父块，2000字）→ 存入 chunks 表
        
        Args:
            docs: LangChain Document 列表，每个包含 page_content 和 metadata
        
        Returns:
            保存后的 Chunk 列表（chunk_id 已填充）
        """
        raw_chunks = self.splitter.split_documents(docs)
        chunks = []

        with sqlite3.connect(self.db_path) as conn:
            for i, doc in enumerate(raw_chunks):
                source = doc.metadata.get("source", "")
                page = doc.metadata.get("page", 0)
                content = doc.page_content

                cur = conn.execute(
                    """INSERT INTO chunks (content, source, page, chunk_index) 
                       VALUES (?, ?, ?, ?)""",
                    (content, source, page, i)
                )
                chunk_id = cur.lastrowid
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    content=content,
                    source=source,
                    page=page,
                    chunk_index=i,
                ))

        return chunks

    # ── 派生内容管理 ─────────────────────────────────────────

    def save_derivative(self, chunk_id: int, strategy: str, dtype: str,
                        derivative_id: int, content: str, metadata: dict = None):
        """保存派生内容到 chunk_derivatives 表
        
        Args:
            chunk_id: 父块 ID
            strategy: "standard"/"summary"/"parent_child"/"hypothetical"
            dtype: "summary"/"child"/"hyde"/"custom"
            derivative_id: 14位纯数字 ID
            content: 派生内容
            metadata: 可选 JSON 元数据
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO chunk_derivatives
                (derivative_id, chunk_id, strategy, derivative_type, content, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (derivative_id, chunk_id, strategy, dtype, content,
                  json.dumps(metadata, ensure_ascii=False) if metadata else None))

    def get_derivatives(self, chunk_id: int, strategy: str = None) -> List[Dict]:
        """获取某 chunk 的派生内容"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if strategy:
                rows = conn.execute(
                    "SELECT * FROM chunk_derivatives WHERE chunk_id=? AND strategy=?",
                    (chunk_id, strategy)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM chunk_derivatives WHERE chunk_id=?",
                    (chunk_id,)
                ).fetchall()
            return [dict(r) for r in rows]

    # ── 父块查询 ─────────────────────────────────────────────

    def get_chunk(self, chunk_id: int) -> Optional[Chunk]:
        """通过 chunk_id 获取父块"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM chunks WHERE chunk_id=?", (chunk_id,)
            ).fetchone()
            if row:
                return Chunk(
                    chunk_id=row["chunk_id"],
                    content=row["content"],
                    source=row["source"],
                    page=row["page"],
                    chunk_index=row["chunk_index"],
                )
            return None

    def get_all_chunks(self, limit: int = None) -> List[Chunk]:
        """获取所有父块（调试用）"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            sql = "SELECT * FROM chunks ORDER BY chunk_id"
            if limit:
                sql += f" LIMIT {limit}"
            rows = conn.execute(sql).fetchall()
            return [
                Chunk(
                    chunk_id=r["chunk_id"],
                    content=r["content"],
                    source=r["source"],
                    page=r["page"],
                    chunk_index=r["chunk_index"],
                )
                for r in rows
            ]

    # ── 索引状态管理 ─────────────────────────────────────────

    def mark_indexed(self, derivative_id: int, strategy: str, channel: str,
                     channel_doc_id: str = None):
        """标记某派生内容已在某通道建立索引"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO index_status
                (derivative_id, strategy, channel, indexed, channel_doc_id)
                VALUES (?, ?, ?, 1, ?)
            """, (derivative_id, strategy, channel, channel_doc_id))

    def is_indexed(self, derivative_id: int, strategy: str, channel: str) -> bool:
        """检查是否已索引"""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """SELECT indexed FROM index_status 
                   WHERE derivative_id=? AND strategy=? AND channel=?""",
                (derivative_id, strategy, channel)
            ).fetchone()
            return bool(row and row[0])

    # ── 统计与调试 ───────────────────────────────────────────

    def get_stats(self) -> Dict:
        """获取存储统计信息"""
        with sqlite3.connect(self.db_path) as conn:
            chunk_count = conn.execute(
                "SELECT COUNT(*) FROM chunks"
            ).fetchone()[0]
            deriv_count = conn.execute(
                "SELECT COUNT(*) FROM chunk_derivatives"
            ).fetchone()[0]
            status_count = conn.execute(
                "SELECT COUNT(*) FROM index_status WHERE indexed=1"
            ).fetchone()[0]
            return {
                "chunks": chunk_count,
                "derivatives": deriv_count,
                "indexed": status_count,
            }

    def clear_all(self, confirm: bool = False):
        """清空所有 chunk 相关表（调试用，危险操作）"""
        if not confirm:
            raise ValueError("必须设置 confirm=True 才能清空")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM index_status")
            conn.execute("DELETE FROM chunk_derivatives")
            conn.execute("DELETE FROM chunks")
            # 重置自增 ID
            conn.execute("DELETE FROM sqlite_sequence WHERE name='chunks'")
            conn.execute("INSERT INTO sqlite_sequence (name, seq) VALUES ('chunks', 1000000)")

    def __repr__(self):
        return f"DocumentStore(db={self.db_path}, chunk_size={self.chunk_size})"
