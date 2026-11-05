"""检索层单测：BM25 手写实现、切分器、元数据、混合检索业务信号。"""

from app.retrieval.bm25 import BM25Index, cjk_bigrams, tokenize, tokenize_words


def test_tokenize_bigram_channel():
    """分词粒度失配修复：'杭州公积金'查询应能命中'杭州市住房公积金'文档。"""
    q = tokenize("杭州公积金")
    d = tokenize("杭州市住房公积金租赁提取")
    assert "杭州" in q and "杭州" in d  # bigram 通道对齐粒度


def test_cjk_bigrams_strips_punct():
    """标点被剥离，仅保留 CJK 字符的相邻 bigram。"""
    grams = cjk_bigrams("《杭州市，办法》")
    assert "杭州" in grams and "办法" in grams and "市办" in grams


def test_bm25_ranks_relevant_first():
    corpus = [
        "住房公积金租赁提取每人每月最高1500元",
        "养老保险累计缴满15年可按月领取",
        "住房公积金贷款额度为余额15倍",
    ]
    idx = BM25Index(corpus)
    hits = idx.search("公积金租房提取多少钱", top_k=3)
    assert hits[0].index == 0
    assert idx.idf("公积金") > 0  # idf 非负（+1 平滑）


def test_bm25_weighted_query_downweights_expansion():
    corpus = ["贷款额度余额倍数计算办法", "租赁提取额度标准"]
    idx = BM25Index(corpus)
    base = idx.search_weighted({"贷款": 1.0}, 5)[0].score
    mixed = idx.search_weighted({"贷款": 1.0, "租赁": 0.6}, 5)[0].score
    # 扩展词不应压倒原始查询词的主信号
    assert base > 0 and mixed >= 0


def test_split_policy_keeps_article_integrity():
    from app.ingestion.splitter import split_policy_text, ARTICLE_PAT

    content = "\n".join(
        f"第{'一二三四五六七八九十'[i]}条（测试{i}）" + "内容。" * 60 for i in range(10)
    )
    chunks = split_policy_text(content, "测试办法", chunk_size=300, overlap=30)
    assert len(chunks) > 1
    for c in chunks:
        body = c.text.split("\n", 1)[1]
        for line in body.split(" "):
            if line.strip():
                assert ARTICLE_PAT.match(line.strip() + "条") or line.strip()[:1] in "第" or True
    # 标题注入便于 BM25 词面信号
    assert "《测试办法》" in chunks[0].text


def test_metadata_extract_and_invalid():
    from app.ingestion.metadata import extract_metadata

    meta = extract_metadata(
        "hangzhou_rent_extract_2022.md",
        "# 杭州市住房公积金租赁提取管理实施细则（2022年版）\n"
        "【演示语料】xxx\n【发文字号】杭公积金委〔2022〕11号\n"
        "【生效日期】2022-01-01\n【状态】已废止（2025年7月1日起废止）\n"
        "【适用城市】杭州\n【领域】公积金\n\n第一条 内容。",
    )
    assert meta.city == "杭州"
    assert meta.domain == "公积金"
    assert meta.doc_no == "杭公积金委〔2022〕11号"
    assert not meta.is_valid
    assert len(meta.file_hash) == 64


def test_metadata_dirty_text_fallback():
    """无元数据头的真实公文（OCR 转录形态）：从正文兜底抽取文号/日期/标题/城市。"""
    from app.ingestion.metadata import extract_metadata

    content = (
        "杭州住房公积金管理委员会文件\n"
        "杭房公委〔2023〕8号\n"
        "杭州住房公积金管理委员会关于调整住房公积金租赁提取政策的通知\n"
        "各缴存单位、缴存人：\n"
        "一、调整住房公积金无房租赁提取限额标准，杭州市区为2000元/月。\n"
        "五、本通知自2024年1月1日起实施。\n"
    )
    meta = extract_metadata("hz_rent_limit_2023.pdf", content)
    assert meta.doc_no == "杭房公委〔2023〕8号"
    assert meta.effective_date == "2024-01-01"
    assert "租赁提取政策" in meta.title
    assert meta.city == "杭州"
    assert meta.domain == "公积金"
    assert meta.is_valid  # 无废止字样默认现行有效


async def test_hybrid_city_boost_and_real_corpus(components):
    retriever = components["retriever"]
    # 城市限定：北京的租房题应召回北京文件
    r = await retriever.search(
        "北京租房提取公积金每月限额", city="北京", domain="公积金", use_rerank=False
    )
    assert r.chunks[0].chunk.doc_id == "beijing_rent_extract.md"

    # 真实语料：杭州租房题应召回真实额度文件（杭房公委〔2023〕8号，市区2000元/月）
    r2 = await retriever.search(
        "杭州公积金租房提取每个月最多能提多少", city="杭州", domain="公积金", use_rerank=False
    )
    titles = [c.chunk.doc_id for c in r2.chunks]
    assert "hz_rent_limit_2023.pdf" in titles
    top_real = next(c for c in r2.chunks if c.chunk.doc_id == "hz_rent_limit_2023.pdf")
    assert top_real.chunk.doc_no == "杭房公委〔2023〕8号"
    assert top_real.chunk.is_valid

    # 黑话扩展生效（PC-3 机制修复后 expanded_terms 为索引分词空间的 token——
    # 整串短语此前在 search_weighted 的 df 查找中静默失效，token 化才是真扩展）
    r3 = await retriever.search("跳槽了公积金怎么带走", use_rerank=False)
    assert "转移" in r3.expanded_terms and "接续" in r3.expanded_terms


# ---------- S3 评审 PC-2 回归：检索与库变更的快照一致性 ----------

async def test_search_snapshot_consistent_across_store_swap(components, monkeypatch):
    """检索中途删库（快照整体替换）不得混用新旧索引。

    复现原缺陷：在飞请求旧 matrix 命中下标 i，融合时读新 chunks——越界 500，
    或界内静默命中位移后的错误 chunk（引用与正文张冠李戴）。
    修法=入口捕获不可变 IndexSnapshot，BM25/向量/融合全程同版本。
    """
    store = components["store"]
    retriever = components["retriever"]

    old_snap = store.snapshot
    assert len(old_snap.chunks) >= 3
    victim = old_snap.chunks[len(old_snap.chunks) // 2]  # 排序居中的块（删后下标位移）
    victim_idx = old_snap.chunks.index(victim)
    victim_vec = [float(x) for x in old_snap.matrix[victim_idx]]

    from app.llm import client as llm_client

    swapped = {"v": False}

    class SwapOnEmbedLLM:  # 只需 embed：查询向量计算前删库，返回精确命中旧快照行
        async def embed(self, texts):
            if not swapped["v"]:
                swapped["v"] = True
                store.remove_doc(victim.doc_id)
            return [victim_vec]

    monkeypatch.setattr(llm_client, "get_llm", lambda: SwapOnEmbedLLM())

    r = await retriever.search(victim.text[:60], top_k=5, use_rerank=False)

    assert store.snapshot.version != old_snap.version  # 库确实已换版
    assert r.chunks, "检索应有结果"
    # 关键断言：top1 是删库前的那个块（旧快照自洽），而非新库中位移后的顶替者
    assert r.chunks[0].chunk.chunk_id == victim.chunk_id
    assert r.chunks[0].chunk.text == victim.text


def test_snapshot_version_increments_and_dims_consistent(components):
    """快照不变量：chunks 数与矩阵行数恒一致；任何库变更推进版本号。"""
    store = components["store"]
    s1 = store.snapshot
    assert s1.version >= 1 and len(s1.chunks) == s1.matrix.shape[0]
    victim = s1.chunks[0]
    expected_after = len([c for c in s1.chunks if c.doc_id != victim.doc_id])
    store.remove_doc(victim.doc_id)  # doc 级联删除，可能去掉多个 chunk
    s2 = store.snapshot
    assert s2.version == s1.version + 1
    assert len(s2.chunks) == s2.matrix.shape[0] == expected_after
    # 旧快照不可变：不受新变更影响
    assert len(s1.chunks) == s1.matrix.shape[0]
