"""SQLite 向量索引 + 增量同步。

选型说明（面试常问"为什么不用 Milvus/Chroma"）：
- 本系统目标规模为数千 chunk（一个城市的政策库），numpy 暴力余弦单次 <5ms，
  引入独立向量库服务反而增加部署复杂度；
- SQLite 单文件零部署、可随项目分发，embedding 以 BLOB 存储；
- VectorStore 职责收敛在 IndexStore 一处，规模上万后可平滑替换为
  Milvus/Chroma 适配层（检索层只依赖 chunks() 与 matrix() 两个方法）。

增量策略（对应面经"RAG 知识库热更新"）：
- 文件内容 hash 未变 → 跳过；hash 变化 → 删除旧 chunk 后重切重嵌；
- 文件删除 → 级联清库；政策修订（新旧版本并存）通过 status 元数据降权，
  而非物理删除——保留旧版可回答"政策变化"类问题。
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from ..config import settings
from .loader import list_corpus, load_file
from .metadata import PolicyMeta, extract_metadata
from .splitter import Chunk, split_policy_text

logger = logging.getLogger(__name__)

EmbedFn = Callable[[list[str]], "list[list[float]]"]


@dataclass
class IndexedChunk:
    chunk_id: str
    doc_id: str
    seq: int
    text: str
    title: str
    city: str
    domain: str
    status: str
    doc_no: str
    effective_date: str
    embedding: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def is_valid(self) -> bool:
        return "废止" not in self.status


@dataclass
class SyncReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    skipped: int = 0
    total_chunks: int = 0

    def as_dict(self) -> dict:
        return {
            "added": self.added,
            "updated": self.updated,
            "removed": self.removed,
            "skipped": self.skipped,
            "total_chunks": self.total_chunks,
        }


class IndexStore:
    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = str(Path(db_path or settings.db_path).resolve())
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()
        self._chunks: list[IndexedChunk] = []
        self._matrix: np.ndarray = np.zeros((0, 0))
        self.load_cache()  # 构造即加载已有索引，避免"忘记 sync 导致空检索"

    # ---------- schema ----------

    def _create_tables(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS docs (
                    doc_id TEXT PRIMARY KEY,
                    file_name TEXT, title TEXT, doc_no TEXT,
                    effective_date TEXT, status TEXT, city TEXT, domain TEXT,
                    file_hash TEXT, chunk_count INTEGER, updated_at TEXT
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT, seq INTEGER, text TEXT, embedding BLOB
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
                """
            )

    # ---------- 增量同步 ----------

    async def sync(self, corpus_dir: str | None = None, embed_fn: EmbedFn | None = None,
                   llm=None) -> SyncReport:
        """增量同步语料目录。embed_fn 为异步函数；llm 用于扫描件 PDF 的 OCR 兜底。"""
        report = SyncReport()
        embed_fn = embed_fn or self._dummy_embed
        with self._lock:
            known = {
                row[0]: row[1]
                for row in self._conn.execute("SELECT doc_id, file_hash FROM docs")
            }
            seen: set[str] = set()
            for path in list_corpus(corpus_dir or settings.corpus_dir):
                try:
                    content = await load_file(path, llm=llm)
                except ValueError as e:
                    logger.warning("跳过无法解析的文档 %s: %s", path.name, e)
                    continue
                meta = extract_metadata(path.name, content)
                seen.add(meta.doc_id)
                old_hash = known.get(meta.doc_id)
                if old_hash == meta.file_hash:
                    report.skipped += 1
                    continue
                if old_hash is not None:
                    report.updated.append(meta.doc_id)
                else:
                    report.added.append(meta.doc_id)
                await self._ingest(meta, content, embed_fn)
            for gone in set(known) - seen:
                self._delete_doc(gone)
                report.removed.append(gone)
            self._conn.commit()
        self.load_cache()
        report.total_chunks = len(self._chunks)
        return report

    async def _ingest(self, meta: PolicyMeta, content: str, embed_fn: EmbedFn) -> None:
        self._delete_doc(meta.doc_id)
        pieces = split_policy_text(content, meta.title)
        if not pieces:
            return
        vectors = embed_fn([f"{c.title}\n{c.text}" for c in pieces])
        if hasattr(vectors, "__await__"):
            vectors = await vectors
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO docs VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    meta.doc_id, meta.file_name, meta.title, meta.doc_no,
                    meta.effective_date, meta.status, meta.city, meta.domain,
                    meta.file_hash, len(pieces), datetime.now().isoformat(timespec="seconds"),
                ),
            )
            rows = []
            for chunk, vec in zip(pieces, vectors):
                blob = np.asarray(vec, dtype=np.float32).tobytes()
                rows.append((f"{meta.doc_id}:{chunk.seq}", meta.doc_id, chunk.seq,
                             chunk.text, blob))
            self._conn.executemany("INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?)", rows)

    async def add_document(self, file_name: str, content: str, embed_fn: EmbedFn) -> int:
        """API 上传入口：同步摄取单个文档，返回 chunk 数。"""
        meta = extract_metadata(file_name, content)
        with self._lock:
            await self._ingest(meta, content, embed_fn)
            self._conn.commit()
        self.load_cache()
        return len([c for c in self._chunks if c.doc_id == meta.doc_id])

    def remove_doc(self, doc_id: str) -> int:
        with self._lock:
            n = self._delete_doc(doc_id)
            self._conn.commit()
        self.load_cache()
        return n

    def _delete_doc(self, doc_id: str) -> int:
        cur = self._conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        self._conn.execute("DELETE FROM docs WHERE doc_id=?", (doc_id,))
        return cur.rowcount

    @staticmethod
    def _dummy_embed(texts: list[str]) -> list[list[float]]:
        """无 embedding 服务时的降级：零向量（检索退化为纯 BM25，系统仍可用）。"""
        return [[0.0] * 8 for _ in texts]

    # ---------- 读取 / 内存缓存 ----------

    def load_cache(self) -> None:
        rows = self._conn.execute(
            """
            SELECT c.chunk_id, c.doc_id, c.seq, c.text, c.embedding,
                   d.title, d.city, d.domain, d.status, d.doc_no, d.effective_date
            FROM chunks c JOIN docs d ON c.doc_id = d.doc_id
            ORDER BY c.doc_id, c.seq
            """
        ).fetchall()
        chunks: list[IndexedChunk] = []
        vecs: list[np.ndarray] = []
        for r in rows:
            vec = np.frombuffer(r[4], dtype=np.float32)
            chunks.append(
                IndexedChunk(
                    chunk_id=r[0], doc_id=r[1], seq=r[2], text=r[3],
                    title=r[5], city=r[6], domain=r[7], status=r[8],
                    doc_no=r[9], effective_date=r[10], embedding=vec,
                )
            )
            vecs.append(vec)
        with self._lock:
            self._chunks = chunks
            if vecs:
                mat = np.vstack(vecs)
                norms = np.linalg.norm(mat, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                self._matrix = mat / norms
            else:
                self._matrix = np.zeros((0, 1))

    @property
    def chunks(self) -> list[IndexedChunk]:
        return self._chunks

    @property
    def matrix(self) -> np.ndarray:
        """L2 归一化后的 (n, dim) 矩阵，余弦相似度 = 矩阵乘。"""
        return self._matrix

    def doc_count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0])

    def list_docs(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT doc_id, title, doc_no, city, domain, status, effective_date,"
            " chunk_count, updated_at FROM docs ORDER BY doc_id"
        ).fetchall()
        keys = ("doc_id", "title", "doc_no", "city", "domain", "status",
                "effective_date", "chunk_count", "updated_at")
        return [dict(zip(keys, r)) for r in rows]

    def close(self) -> None:
        self._conn.close()
