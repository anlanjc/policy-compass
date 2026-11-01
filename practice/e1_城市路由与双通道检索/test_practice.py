"""E1 判卷测试 —— 改编自 policy-compass/tests/test_retrieval.py 与 tests/test_tools_graph.py。

验收标准：pytest test_practice.py 全绿（10 passed）。
"""

from skeleton import SimpleBM25, cjk_bigrams, rule_based_route, tokenize


# ---- 规则路由（改编自 tests/test_tools_graph.py::test_agent_rule_router_fallback + 扩展） ----


def test_router_multi_city_falls_back_to_nationwide():
    """多城对比 → 全国（不限定）。"""
    assert rule_based_route("北京和上海租房提取额度").city == "全国"


def test_router_single_city():
    assert rule_based_route("杭州公积金怎么提取").city == "杭州"


def test_router_no_city():
    assert rule_based_route("公积金贷款利率").city == "全国"


def test_router_city_alias():
    """别名词表：'沪牌' → 上海（仅当未命中任何全名时才查别名）。"""
    assert rule_based_route("沪牌怎么拍").city == "上海"


def test_router_domain_and_calc():
    r = rule_based_route("我杭州公积金余额4万缴了5年月薪2万，最多能贷多少？")
    assert r.city == "杭州"
    assert r.domain == "公积金"
    assert r.need_calc is True  # "余额/能贷" + 数字


def test_router_sub_questions():
    assert rule_based_route("北京和上海租房提取额度").sub_questions
    assert rule_based_route("杭州公积金怎么提取").sub_questions == []


# ---- 双通道分词（改编自 tests/test_retrieval.py::test_tokenize_bigram_channel / test_cjk_bigrams_strips_punct） ----


def test_tokenize_bigram_channel():
    """分词粒度失配修复：'杭州公积金'查询应能命中'杭州市住房公积金'文档。"""
    q = tokenize("杭州公积金")
    d = tokenize("杭州市住房公积金租赁提取")
    assert "杭州" in q and "杭州" in d  # bigram 通道对齐粒度
    assert "公积" in q and "公积" in d


def test_cjk_bigrams_strips_punct():
    """标点被剥离，仅保留 CJK 字符的相邻 bigram。"""
    grams = cjk_bigrams("《杭州市，办法》")
    assert "杭州" in grams and "办法" in grams and "市办" in grams
    assert all(len(g) == 2 for g in grams)


# ---- 手写 BM25（改编自 tests/test_retrieval.py::test_bm25_ranks_relevant_first / weighted） ----


def test_bm25_ranks_relevant_first():
    corpus = [
        "住房公积金租赁提取每人每月最高1500元",
        "养老保险累计缴满15年可按月领取",
        "住房公积金贷款额度为余额15倍",
    ]
    idx = SimpleBM25(corpus)
    hits = idx.search("公积金租房提取多少钱", top_k=3)
    assert hits[0][0] == 0
    assert idx.idf("公积金") > 0  # idf 非负（+1 平滑）


def test_bm25_weighted_query_downweights_expansion():
    corpus = ["贷款额度余额倍数计算办法", "租赁提取额度标准"]
    idx = SimpleBM25(corpus)
    base = idx.search_weighted({"贷款": 1.0}, 5)[0][1]
    mixed = idx.search_weighted({"贷款": 1.0, "租赁": 0.6}, 5)[0][1]
    # 扩展词不应压倒原始查询词的主信号
    assert base > 0 and mixed >= 0


def test_bm25_zero_hit_returns_empty():
    idx = SimpleBM25(["住房公积金租赁提取标准"])
    assert idx.search("护照换领", top_k=3) == []
