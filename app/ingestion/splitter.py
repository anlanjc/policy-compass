"""政策条文感知的文本切分器。

通用 RecursiveCharacterTextSplitter 按固定字符窗硬切，会把"第X条"拦腰截断——
政策问答的检索单元应当是完整条文。本切分器策略：

1. 优先按「第X条」条文边界聚合：条文逐条累加，凑满 chunk_size 即成块，
   保证块内条文完整；
2. 无条文结构（说明性文档）时退化为 段落 → 句子 两级滑窗合并；
3. 每块头部注入文档标题（BM25 检索词面信号，成本近零收益显著，见 eval 对比）。

参数（chunk_size=450 / overlap=60）由 eval/runner.py 在 300/450/800 三档扫参得出。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..config import settings

ARTICLE_PAT = re.compile(r"^第[一二三四五六七八九十百\d]+条")
SENT_PAT = re.compile(r"(?<=[。；;！？?\n])")


@dataclass
class Chunk:
    seq: int
    text: str
    doc_id: str = ""
    title: str = ""


def split_policy_text(
    content: str,
    title: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap if overlap is not None else settings.chunk_overlap

    body_lines: list[str] = []
    meta_lines = 0
    for line in content.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("【") and meta_lines < 6:  # 头部元数据行不入正文
            meta_lines += 1
            continue
        if s.startswith("#"):  # 标题行已单独存 metadata
            continue
        body_lines.append(s)
    body = "\n".join(body_lines)

    units = _split_articles(body) if ARTICLE_PAT.search(body) else _split_paragraphs(body)
    return _pack(units, title, chunk_size, overlap)


def _split_articles(body: str) -> list[str]:
    """按条文切分；条内的换行合并。"""
    units: list[str] = []
    current: list[str] = []
    for line in body.splitlines():
        if ARTICLE_PAT.match(line.strip()) and current:
            units.append(" ".join(current).strip())
            current = [line.strip()]
        else:
            s = line.strip()
            if s:
                current.append(s)
    if current:
        units.append(" ".join(current).strip())
    return units


def _split_paragraphs(body: str) -> list[str]:
    """段落 → 句子两级切分（无条文结构的兜底）。"""
    paragraphs = [p.strip() for p in re.split(r"\n+", body) if p.strip()]
    units: list[str] = []
    for p in paragraphs:
        if len(p) <= 200:
            units.append(p)
            continue
        sents = [s for s in SENT_PAT.split(p) if s.strip()]
        units.extend(s.strip() for s in sents if s.strip())
    return units


def _pack(units: list[str], title: str, chunk_size: int, overlap: int) -> list[Chunk]:
    """把单元聚合为块：每块以标题开头，超长单元硬切并滑窗重叠。"""
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_len = 0

    def flush() -> None:
        nonlocal buf, buf_len
        if buf:
            chunks.append(Chunk(seq=len(chunks), text=f"《{title}》\n" + "\n".join(buf)))
            # overlap：保留尾部若干单元作为下块开头，保证条文上下文连续
            tail: list[str] = []
            tail_len = 0
            for u in reversed(buf):
                if tail_len + len(u) > overlap:
                    break
                tail.insert(0, u)
                tail_len += len(u)
            buf, buf_len = list(tail), tail_len

    for u in units:
        if len(u) > chunk_size * 2:  # 单元本身超长：按字符滑窗硬切
            for piece in _hard_slice(u, chunk_size, overlap):
                buf.append(piece)
                buf_len += len(piece)
                if buf_len >= chunk_size:
                    flush()
            continue
        buf.append(u)
        buf_len += len(u)
        if buf_len >= chunk_size:
            flush()
    flush()
    # 纯 tail 重复产生的空块防御
    return [c for c in chunks if len(c.text) > 40]


def _hard_slice(text: str, size: int, overlap: int) -> list[str]:
    step = max(size - overlap, 1)
    return [text[i : i + size] for i in range(0, len(text), step)]
