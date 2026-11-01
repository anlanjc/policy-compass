"""引用原文定位（PC-3）：chunk 级条款/页码元数据。

红线：标签只随单元记账，不改变任何切分边界——chunk 正文文本与引入该特性前
逐字节一致（页标记必须被完全剥离，见 test_markers_do_not_change_text）。
"""

from __future__ import annotations

import asyncio
import sqlite3

from app.ingestion.indexer import IndexStore
from app.ingestion.splitter import split_policy_text


def _article_doc() -> str:
    return "\n".join(
        [
            "【语料性质】测试语料",
            "# 测试办法",
            "第一条 为规范测试，制定本办法。",
            "第二条 提取额度为每月两千元。",
            "第三条 贷款额度按余额计算。",
            "第四条 本办法自发布之日起施行。",
        ]
    )


def test_article_sections_cover_all_chunks():
    chunks = split_policy_text(_article_doc(), "测试办法")
    assert chunks, "应产出至少一个 chunk"
    labels = [s for c in chunks for s in c.sections]
    assert "第一条" in labels and "第四条" in labels
    # 短文档单块时，块内条款按出现序排列
    if len(chunks) == 1:
        assert chunks[0].sections == ["第一条", "第二条", "第三条", "第四条"]


def test_page_markers_stripped_and_recorded():
    content = (
        "第一条 总则内容，为了规范住房公积金管理和使用而制定本办法。"
        "\n【p1】\n第二条 分页前的内容，提取额度按月计算且不超过规定上限。"
        "\n【p2】\n第三条 分页后的内容，贷款额度按账户余额乘以倍数确定。"
    )
    chunks = split_policy_text(content, "测试")
    assert chunks
    assert all("【p" not in c.text for c in chunks), "页标记不得进入 chunk 正文"
    pages = [p for c in chunks for p in c.pages]
    assert 1 in pages and 2 in pages


def test_markers_do_not_change_text():
    """红线锚点：有页标记与无页标记的语料，切块正文逐字节一致。"""
    with_pages = "第一条 总则。\n【p1】\n第二条 内容甲。\n【p2】\n第三条 内容乙。"
    without = "第一条 总则。\n第二条 内容甲。\n第三条 内容乙。"
    a = "\n".join(c.text for c in split_policy_text(with_pages, "T"))
    b = "\n".join(c.text for c in split_policy_text(without, "T"))
    assert a == b


def test_paragraph_mode_inherits_section_label():
    content = (
        "一、租房提取条件\n申请人及配偶在本地无自有住房且租赁住房的，可以申请提取。\n"
        "二、提取额度\n每月提取金额不超过两千元，年累计不超过两点四万元。"
    )
    chunks = split_policy_text(content, "通知")
    texts = "\n".join(c.text for c in chunks)
    assert "一、" in texts  # 正文不受标签记账影响
    labels = {s for c in chunks for s in c.sections}
    assert "一、" in labels and "二、" in labels


def test_indexer_roundtrip_sections(tmp_path):
    store = IndexStore(str(tmp_path / "idx.db"))
    try:
        n = asyncio.run(
            store.add_document(
                "测试办法.md", _article_doc(),
                lambda texts: [[0.0] * 8 for _ in texts],
            )
        )
        assert n > 0
        cs = [c for c in store.chunks if c.title == "测试办法"]
        assert any("第一条" in c.sections for c in cs)
    finally:
        store.close()


def test_schema_migration_from_old_db(tmp_path):
    """旧 schema（无 sections/pages 列）打开即自动 ALTER，不抛错、不丢数据。"""
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE docs (
            doc_id TEXT PRIMARY KEY,
            file_name TEXT, title TEXT, doc_no TEXT,
            effective_date TEXT, status TEXT, city TEXT, domain TEXT,
            file_hash TEXT, chunk_count INTEGER, updated_at TEXT
        );
        CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT, seq INTEGER, text TEXT, embedding BLOB
        );
        INSERT INTO docs VALUES ('d1','f.md','旧文档','','','','全国','通用','h',0,'');
        INSERT INTO chunks VALUES ('d1:0','d1',0,'旧正文',zeroblob(8));
        """
    )
    conn.commit()
    conn.close()
    store = IndexStore(str(db))
    try:
        cols = {r[1] for r in store._conn.execute("PRAGMA table_info(chunks)")}
        assert "sections" in cols and "pages" in cols
        old = [c for c in store.chunks if c.chunk_id == "d1:0"]
        assert old and old[0].text == "旧正文" and old[0].sections == []
    finally:
        store.close()
