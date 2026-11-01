"""政策条文感知的文本切分器。

通用 RecursiveCharacterTextSplitter 按固定字符窗硬切，会把"第X条"拦腰截断——
政策问答的检索单元应当是完整条文。本切分器策略：

1. 优先按「第X条」条文边界聚合：条文逐条累加，凑满 chunk_size 即成块，
   保证块内条文完整；
2. 无条文结构（说明性文档）时退化为 段落 → 句子 两级滑窗合并；
3. 每块头部注入文档标题（BM25 检索词面信号，成本近零收益显著，见 eval 对比）。

引用原文定位（PC-3）：切块同时记录每块的
- sections：条款/章节标签（第X条 / 第X章·节·部分 / 一、二、），条文边界即标签；
- pages：页码（PDF/OCR 摄取时在页边界插入【pN】标记行，本切分器剥离标记并记账；
  网页/手写 markdown 语料无页码概念，如实为空）。
设计红线：标签只随单元记账，**不改变任何切分边界**——chunk 文本与引入该特性前
逐字节一致，评测数字零漂移。

参数（chunk_size=450 / overlap=60）由 eval/runner.py 在 300/450/800 三档扫参得出。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..config import settings

ARTICLE_PAT = re.compile(r"^第[一二三四五六七八九十百\d]+条")
CHAP_PAT = re.compile(r"^第[一二三四五六七八九十百\d]+[章节部分]")
NUMHEAD_PAT = re.compile(r"^[一二三四五六七八九十]+、")
PAGE_PAT = re.compile(r"^【p(\d+)】$")
SENT_PAT = re.compile(r"(?<=[。；;！？?\n])")

# 行首标题的长度上限：防止长句开头恰好是"第X条"字样被误记为标题
HEAD_MAX_LEN = 30


@dataclass
class Chunk:
    seq: int
    text: str
    doc_id: str = ""
    title: str = ""
    sections: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)


# 清洗后的一行：(文本, 所在页码 or None)
CleanLine = tuple[str, "int | None"]
# 切分单元：(文本, 所属条款/章节标签 or None, 页码 or None)
Unit = tuple[str, "str | None", "int | None"]


def split_policy_text(
    content: str,
    title: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap if overlap is not None else settings.chunk_overlap

    lines = _clean_lines(content)
    units = _split_units(lines)
    return _pack(units, title, chunk_size, overlap)


def _clean_lines(content: str) -> list[CleanLine]:
    """剥头部元数据行/标题行，识别【pN】页标记行；返回 (行文本, 当前页码)。"""
    out: list[CleanLine] = []
    page: int | None = None
    meta_lines = 0
    for line in content.splitlines():
        s = line.strip()
        if not s:
            continue
        m = PAGE_PAT.match(s)
        if m:  # 页标记行：记账后不入正文
            page = int(m.group(1))
            continue
        if s.startswith("【") and meta_lines < 6:  # 头部元数据行不入正文
            meta_lines += 1
            continue
        if s.startswith("#"):  # 标题行已单独存 metadata
            continue
        out.append((s, page))
    return out


def _heading_label(line: str) -> str | None:
    """行首的条款/章节/数字序号标题，长句不认。"""
    if len(line) > HEAD_MAX_LEN:
        return None
    for pat in (ARTICLE_PAT, CHAP_PAT, NUMHEAD_PAT):
        m = pat.match(line)
        if m:
            return m.group(0)
    return None


def _unit_label(line: str) -> str | None:
    """单元标签：「第X条」不限长（真实公文常见"第七条（常见问题）问：……"长条文行，
    实测限长会整批漏标）；章节/数字序号仍守长度防线防误记。"""
    m = ARTICLE_PAT.match(line)
    if m:
        return m.group(0)
    return _heading_label(line)


def _split_units(lines: list[CleanLine]) -> list[Unit]:
    """统一切分出 (单元文本, 标签, 页码)。

    条文模式：单元标签=开头条文；条文前的引言与条文间的章节行只更新"当前标签"
    上下文，不打断聚合（切块稳定优先）。
    段落模式：逐行成单元，标签=自身标题或最近一次标题（继承）。

    模式判定与旧版逐字对齐：旧 `ARTICLE_PAT.search(body)` 的 `^` 无 MULTILINE，
    实际语义是"正文首行以第X条开头"才进条文模式——若放宽为"任意行命中"，
    十几个文件会从段落模式翻成条文模式，切分边界漂移（实测 68→63 块）。
    """
    if lines and ARTICLE_PAT.match(lines[0][0]):
        units: list[Unit] = []
        current: list[str] = []
        cur_label: str | None = None
        cur_page: int | None = None
        for text, page in lines:
            if ARTICLE_PAT.match(text) and current:
                units.append((" ".join(current).strip(), cur_label, cur_page))
                current, cur_label, cur_page = [text], _unit_label(text), page
            elif not current:
                current, cur_label, cur_page = [text], _unit_label(text), page
            else:
                if len(text) <= HEAD_MAX_LEN and _heading_label(text):
                    cur_label = _heading_label(text)  # 章节行：更新上下文，不另起单元
                current.append(text)
        if current:
            units.append((" ".join(current).strip(), cur_label, cur_page))
        return units

    units = []
    cur_label: str | None = None
    for text, page in lines:
        label = _unit_label(text)
        if label:
            cur_label = label
        shown = label or cur_label
        if len(text) <= 200:
            units.append((text, shown, page))
            continue
        for s in SENT_PAT.split(text):
            if s.strip():
                units.append((s.strip(), shown, page))
    return units


def _pack(units: list[Unit], title: str, chunk_size: int, overlap: int) -> list[Chunk]:
    """把单元聚合为块：每块以标题开头，超长单元硬切并滑窗重叠。"""
    chunks: list[Chunk] = []
    buf: list[Unit] = []
    buf_len = 0

    def flush() -> None:
        nonlocal buf, buf_len
        if not buf:
            return
        sections: list[str] = []
        pages: list[int] = []
        for _, label, page in buf:
            if label and label not in sections:
                sections.append(label)
            if page is not None and page not in pages:
                pages.append(page)
        chunks.append(
            Chunk(
                seq=len(chunks),
                text=f"《{title}》\n" + "\n".join(u for u, _, _ in buf),
                sections=sections,
                pages=pages,
            )
        )
        # overlap：保留尾部若干单元作为下块开头，保证条文上下文连续
        tail: list[Unit] = []
        tail_len = 0
        for u in reversed(buf):
            if tail_len + len(u[0]) > overlap:
                break
            tail.insert(0, u)
            tail_len += len(u[0])
        buf, buf_len = list(tail), tail_len

    for text, label, page in units:
        if len(text) > chunk_size * 2:  # 单元本身超长：按字符滑窗硬切
            for piece in _hard_slice(text, chunk_size, overlap):
                buf.append((piece, label, page))
                buf_len += len(piece)
                if buf_len >= chunk_size:
                    flush()
            continue
        buf.append((text, label, page))
        buf_len += len(text)
        if buf_len >= chunk_size:
            flush()
    flush()
    # 纯 tail 重复产生的空块防御
    return [c for c in chunks if len(c.text) > 40]


def _hard_slice(text: str, size: int, overlap: int) -> list[str]:
    step = max(size - overlap, 1)
    return [text[i : i + size] for i in range(0, len(text), step)]
