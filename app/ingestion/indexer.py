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

import json
import logging
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from ..config import settings
from .loader import list_corpus, load_file
from .metadata import PolicyMeta, extract_metadata
from .splitter import split_policy_text

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
    sections: list[str] = field(default_factory=list)  # 条款/章节标签（引用原文定位）
    pages: list[int] = field(default_factory=list)  # 页码（PDF/OCR 语料；网页转载为空）

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
    snapshot_version: int = 0  # 本次同步发布后的快照版本（索引版本化，PC-HR1）
    errors: list[str] = field(default_factory=list)  # 存在但解析失败的文件（保留旧版，PC-HR1）

    def as_dict(self) -> dict:
        return {
            "added": self.added,
            "updated": self.updated,
            "removed": self.removed,
            "skipped": self.skipped,
            "total_chunks": self.total_chunks,
            "snapshot_version": self.snapshot_version,
            "errors": self.errors,
        }


@dataclass(frozen=True)
class IndexSnapshot:
    """不可变检索快照（S3 评审 PC-2）：chunks 元组与矩阵**同版本原子发布**。

    此前检索的三个引用（BM25 命中 / matrix / chunks）跨越 await 各自读取活属性，
    上传/删除触发 load_cache 整体替换后，在飞请求会把旧下标映射到新 chunks——
    越界 500，或界内静默命中位移后的错误 chunk（引用与正文张冠李戴）。
    快照让在飞请求基于同一自洽版本算完；新请求自动取最新版。
    """

    version: int
    chunks: tuple[IndexedChunk, ...]
    matrix: np.ndarray


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
        self._snap_version = 0
        self._snapshot = IndexSnapshot(0, (), self._matrix)
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
            self._migrate_columns()

    def _migrate_columns(self) -> None:
        """旧库升级：chunks 表补 sections/pages 列（引用原文定位，PC-3）。

        CREATE TABLE IF NOT EXISTS 对已存在的旧 schema 不生效，须显式 ALTER；
        旧库已有 chunk 行的 sections/pages 保持 NULL（重摄取后自动回填）。
        """
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(chunks)")}
        if "sections" not in cols:
            self._conn.execute("ALTER TABLE chunks ADD COLUMN sections TEXT")
        if "pages" not in cols:
            self._conn.execute("ALTER TABLE chunks ADD COLUMN pages TEXT")

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
                    # 解析失败 ≠ 文件已删除（PC-HR1）：磁盘上的文件读不出内容可能是
                    # 瞬时 IO 错误/缺视觉模型——旧实现直接 continue 会落入下面的
                    # gone 级联删除，一次解析抖动就静默清掉该文档全部 chunk。
                    # 改为：已入库的保留旧版原样可检，新文件仅记错误；均在报告 errors 留痕。
                    if path.name in known:
                        seen.add(path.name)
                    report.errors.append(path.name)
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
            changed = bool(report.added or report.updated or report.removed)
        if changed:
            self.load_cache()
        # 无变更时不发布新快照（PC-HR1）：版本号保持稳定，在途/新增请求继续用
        # 同一版本，BM25 免重建；内存缓存本就与未变的 DB 一致。
        # （仅当绕过 IndexStore 直接改库时会出现内存滞后——非受支持路径。）
        report.total_chunks = len(self._chunks)
        report.snapshot_version = self._snap_version
        return report

    async def _ingest(self, meta: PolicyMeta, content: str, embed_fn: EmbedFn) -> None:
        self._delete_doc(meta.doc_id)
        pieces = split_policy_text(content, meta.title)
        if not pieces:
            return
        # embed 输入口径（2026-11-06 如实化注释，行为零变更）：
        # splitter._pack 构造 Chunk 时从不赋值 title（恒为默认 ""），故本行的
        # f"{c.title}\n{c.text}" 实际产出 "\n"+正文——标题经 text 首行《标题》进入，
        # 功能无损；全库既有向量（含 FORCE_MOCK 哈希向量）均按此口径计算，
        # 冻结评测数字以其为前提。若要改为纯 c.text（消除冗余换行前缀），
        # 会同时改变哈希向量与真实向量 → 必须捆绑：全量重摄取 + v1/v2 mock/real
        # 全量复测 + 冻结口径回填 + 简历数字同步（消融报告 ablation.md §2 有记录）。
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
                rows.append(
                    (
                        f"{meta.doc_id}:{chunk.seq}", meta.doc_id, chunk.seq,
                        chunk.text, blob,
                        json.dumps(chunk.sections, ensure_ascii=False),
                        json.dumps(chunk.pages),
                    )
                )
            self._conn.executemany(
                "INSERT OR REPLACE INTO chunks"
                " (chunk_id, doc_id, seq, text, embedding, sections, pages)"
                " VALUES (?,?,?,?,?,?,?)",
                rows,
            )

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
                   d.title, d.city, d.domain, d.status, d.doc_no, d.effective_date,
                   c.sections, c.pages
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
                    sections=json.loads(r[11]) if r[11] else [],
                    pages=json.loads(r[12]) if r[12] else [],
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
            # 快照原子发布（S3 评审 PC-2）：单个属性赋值在 CPython 下原子，
            # 在飞检索持有旧 Snapshot 不受影响，新请求取到最新版
            self._snap_version += 1
            self._snapshot = IndexSnapshot(self._snap_version, tuple(chunks), self._matrix)

    @property
    def snapshot(self) -> IndexSnapshot:
        """当前检索快照（chunks 与 matrix 恒同版本）。"""
        return self._snapshot

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
