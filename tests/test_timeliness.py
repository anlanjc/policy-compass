"""政策时效性引擎测试（1B5）：检索日期/状态过滤、date_rank 消费者、对比问答。

零变化守门（兼容性红线）：本文件中的"改前基线"数字/提示词串取自
67cc972（改动前）在 POLICY_COMPASS_FORCE_MOCK=1 下的实测快照——不传新参数时
检索序与生成提示词必须与基线逐字节一致（v1/v2 冻结口径 87.9%/92.2% 依赖此）。
"""

import json
from datetime import date
from pathlib import Path

from app.retrieval.bm25 import BM25Result

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_DEPRECATED_DOC = (
    "# 杭州市住房公积金租赁提取管理实施细则（2022年版）\n"
    "【发文字号】杭公积金委〔2022〕11号\n"
    "【生效日期】2022-01-01\n"
    "【状态】已废止（2025年7月1日起废止）\n"
    "【适用城市】杭州\n"
    "【领域】公积金\n"
    "\n"
    "第一条（提取额度）无房租赁提取住房公积金额度为每人每月1500元，每年提取一次。\n"
    "第二条（申请材料）职工申请租赁提取的，应当提供无房证明与租赁合同。\n"
)

# ---------- 元数据日期助手（date_rank 行为保持 + parse_effective_date 新消费者） ----------


def test_parse_effective_date_and_date_rank_equivalence():
    from app.ingestion.metadata import date_rank, parse_effective_date

    assert parse_effective_date("2024-01-01") == date(2024, 1, 1)
    assert parse_effective_date("") is None
    assert parse_effective_date(None) is None
    assert parse_effective_date("garbage") is None
    # date_rank 语义与重构前逐字节一致：合法日期=序数、非法/空=0
    assert date_rank("2024-01-01") == date(2024, 1, 1).toordinal() * 1.0
    assert date_rank("") == 0.0 and date_rank("bad") == 0.0 and date_rank(None) == 0.0


# ---------- 检索期 as_of / status 过滤（真实语料） ----------


async def test_as_of_filter_boundaries_real_corpus(components):
    """时点过滤边界：effective_date <= as_of 保留；边界日当天保留；早于全部日期则杭州本地块全排除。"""
    retriever = components["retriever"]

    async def docs_at(as_of):
        r = await retriever.search(
            "杭州无房租赁公积金提取限额标准", city="杭州", domain="公积金",
            use_rerank=False, as_of_date=as_of,
        )
        return [c.chunk.doc_id for c in r.chunks]

    # 2023 版生效 2024-01-01：as_of=2023-12-31 排除、边界日 2024-01-01 当天保留
    assert "hz_rent_limit_2023.pdf" not in await docs_at("2023-12-31")
    assert "hz_rent_limit_2023.pdf" in await docs_at("2024-01-01")
    # 2025 版生效 2025-05-21：as_of=2024-06-01 只有 2023 版；as_of=2025-06-30 两版都在
    assert "hz_extract_optimize_2025.pdf" not in await docs_at("2024-06-01")
    assert "hz_rent_limit_2023.pdf" in await docs_at("2024-06-01")
    both = await docs_at("2025-06-30")
    assert "hz_rent_limit_2023.pdf" in both and "hz_extract_optimize_2025.pdf" in both
    # date 入参与 ISO 字符串等价
    r = await retriever.search(
        "杭州无房租赁公积金提取限额标准", city="杭州", domain="公积金",
        use_rerank=False, as_of_date=date(2024, 6, 1),
    )
    assert "hz_extract_optimize_2025.pdf" not in [c.chunk.doc_id for c in r.chunks]


async def test_as_of_rejects_bad_input_and_excludes_undated(components):
    """无生效日期块在 as_of 过滤下一律排除（成都件无日期头）；非法 as_of 显式报错。"""
    retriever = components["retriever"]
    r = await retriever.search(
        "成都租房提取额度标准是多少", city="成都", domain="公积金",
        use_rerank=False, as_of_date="2026-12-31",
    )
    assert "chengdu_rent_extract.md" not in [c.chunk.doc_id for c in r.chunks]
    # 对照：不过滤时成都件可召回
    r2 = await retriever.search(
        "成都租房提取额度标准是多少", city="成都", domain="公积金", use_rerank=False
    )
    assert "chengdu_rent_extract.md" in [c.chunk.doc_id for c in r2.chunks]
    try:
        await retriever.search("测试", as_of_date="garbage")
        raise AssertionError("非法 as_of_date 必须 ValueError")
    except ValueError:
        pass


async def test_status_filter_real_corpus(components):
    """状态过滤：现行有效 → 全部 is_valid；已废止 → 正式语料（23 份全现行）为空。"""
    retriever = components["retriever"]
    r_valid = await retriever.search(
        "租房提取", city="杭州", domain="公积金", use_rerank=False,
        status_filter="现行有效",
    )
    assert r_valid.chunks and all(c.chunk.is_valid for c in r_valid.chunks)
    r_invalid = await retriever.search(
        "租房提取", city="杭州", domain="公积金", use_rerank=False,
        status_filter="已废止",
    )
    assert r_invalid.chunks == []  # 语料盘点结论：0 份已废止文件


async def test_status_filter_with_deprecated_doc(components):
    """合成已废止件：已废止路只回旧版、现行路只回新版；默认（不过滤）旧版×0.3 沉底。"""
    store = components["store"]
    retriever = components["retriever"]
    await store.add_document("hangzhou_rent_2022.md", _DEPRECATED_DOC, components["llm"].embed)
    retriever.refresh()

    r_old = await retriever.search(
        "杭州租房提取额度", city="杭州", domain="公积金", use_rerank=False,
        status_filter="已废止",
    )
    assert [c.chunk.doc_id for c in r_old.chunks] == ["hangzhou_rent_2022.md"]
    assert all(not c.chunk.is_valid for c in r_old.chunks)

    r_cur = await retriever.search(
        "杭州租房提取额度", city="杭州", domain="公积金", use_rerank=False,
        status_filter="现行有效",
    )
    assert r_cur.chunks and all(c.chunk.is_valid for c in r_cur.chunks)
    assert all(c.chunk.doc_id != "hangzhou_rent_2022.md" for c in r_cur.chunks)

    r_all = await retriever.search(
        "杭州租房提取额度", city="杭州", domain="公积金", use_rerank=False, top_k=12
    )
    ids = [c.chunk.doc_id for c in r_all.chunks]
    assert "hangzhou_rent_2022.md" in ids, "默认不过滤时已废止件仍可检索（保留可回答历史）"
    old_pos = ids.index("hangzhou_rent_2022.md")
    assert old_pos >= 1, "已废止件 ×0.3 沉底，不得顶在现行文件之前"
    assert all(c.chunk.is_valid for c in r_all.chunks[:old_pos]), "沉底前不得有现行件被压过"
    assert r_all.chunks[old_pos].chunk.effective_date == "2022-01-01"


# ---------- date_rank 消费者：新旧版本融合分并列时新版本稳定靠前 ----------


def _make_chunk(doc_id, doc_no, title, effective_date, text):
    from app.ingestion.indexer import IndexedChunk

    return IndexedChunk(
        chunk_id=f"{doc_id}:0", doc_id=doc_id, seq=0, text=text, title=title,
        city="杭州", domain="公积金", status="现行有效", doc_no=doc_no,
        effective_date=effective_date,
    )


def _tie_scenario(retriever, old_doc_no, new_doc_no, old_title, new_title,
                  old_date="2022-01-01", new_date="2025-07-01"):
    """构造 old/new 融合分**精确相等**的场景：old 在 BM25 路 rank20（1/(60+20)），
    new 在向量路 rank4（0.8/(60+4)=1/80）——两个有理数在 IEEE754 下位相等；
    其余 19 个 BM25 填充块 + 3 个向量填充块分数严格更高，并列组必然相邻垫底。"""
    query = "测试用查询词"
    chunks = [
        _make_chunk("old.md", old_doc_no, old_title, old_date, "旧版条文内容示例"),
        _make_chunk("new.md", new_doc_no, new_title, new_date, "新版条文内容示例"),
    ]
    chunks += [_make_chunk(f"f{i}.md", "", f"填充文件{i}", "2020-01-01", f"无关填充内容{i}")
               for i in range(19)]
    chunks += [_make_chunk(f"v{i}.md", "", f"向量填充{i}", "2020-01-01", f"无关向量内容{i}")
               for i in range(3)]
    bm25_hits = [BM25Result(index=2 + i, score=1.0) for i in range(19)]
    bm25_hits.append(BM25Result(index=0, score=1.0))  # old → BM25 rank 20 → 1/80
    vec_hits = [(21 + i, 0.9 - 0.1 * i) for i in range(3)]
    vec_hits.append((1, 0.5))  # new → 向量 rank 4 → 0.8/64 == 1/80
    out = retriever._rrf_fuse(tuple(chunks), query, bm25_hits, vec_hits, None, None)
    tail = [sc.chunk.doc_id for sc in out[-2:]]
    assert out[-1].final_score == out[-2].final_score  # 并列成立
    return tail


async def test_date_rank_tiebreak_newer_first(components):
    """同 doc_no 新旧版并列 → date_rank 使新版在前（metadata.date_rank 的消费者）。"""
    retriever = components["retriever"]
    tail = _tie_scenario(
        retriever,
        old_doc_no="杭公积金委〔2022〕11号", new_doc_no="杭公积金委〔2022〕11号",
        old_title="租赁提取管理细则", new_title="租赁提取管理细则修订版",
    )
    assert tail == ["new.md", "old.md"], "融合分并列的同文号新旧版：新版本必须排前"


async def test_date_rank_tiebreak_scoped_no_drift(components):
    """作用域收窄（冻结口径零扰动）：异源并列不重排；同文档多块并列保持原序。"""
    retriever = components["retriever"]
    # 异源（不同 doc_no 且不同标题）：并列时保持融合插入序（old 在前），不被时序重排
    tail = _tie_scenario(
        retriever,
        old_doc_no="杭公积金委〔2022〕11号", new_doc_no="浙公积金发〔2025〕9号",
        old_title="租赁提取管理细则", new_title="租房提取操作指引",
    )
    assert tail == ["old.md", "new.md"]
    # 同文档多块（同 doc_id，date_rank 相同）：稳定排序保持原序
    tail2 = _tie_scenario(
        retriever,
        old_doc_no="杭公积金委〔2022〕11号", new_doc_no="杭公积金委〔2022〕11号",
        old_title="租赁提取管理细则", new_title="租赁提取管理细则",
        old_date="2024-01-01", new_date="2024-01-01",
    )
    assert tail2 == ["old.md", "new.md"]


# ---------- 对比意图检测与路由叠加 ----------


def test_wants_version_compare_keywords():
    from app.agent.router import wants_version_compare

    for q in ("杭州租房提取额度以前是多少", "政策原来怎么规定的", "额度变了没",
              "条例最近一次修订是什么时候", "调整前标准是多少", "老政策还能用吗",
              "旧版办法的额度", "文件废止了吗"):
        assert wants_version_compare(q), q
    for q in ("杭州公积金租房提取每个月最多能提多少", "北京和上海租房提取额度对比",
              "公积金贷款利率是多少", "社保断缴了怎么办"):
        assert not wants_version_compare(q), q


async def test_route_node_merges_version_compare_flag(components):
    """route_node 确定性叠加：对比问题 version_compare=True，中性问题 False（其余路由字段不变）。"""
    agent = components["agent"]
    hit = await agent.nodes["route"]({"question": "杭州公积金租房提取额度以前是多少"})
    assert hit["route"]["version_compare"] is True
    assert hit["route"]["city"] == "杭州" and hit["route"]["domain"] == "公积金"

    miss = await agent.nodes["route"]({"question": "杭州公积金租房提取每个月最多能提多少"})
    assert miss["route"]["version_compare"] is False
    assert miss["route"]["city"] == "杭州"


# ---------- 零变化守门（改前基线：67cc972 @ FORCE_MOCK 实测快照） ----------

# 基线快照（use_rerank=False，无新参数）：[(doc_id, seq, final_score)] 逐字节一致
_BASELINE_ORDERS = [
    ("北京租房提取公积金每月限额", "北京", [
        ("beijing_rent_extract.md", 0), ("beijing_rent_extract.md", 1),
        ("beijing_rent_directpay_2025.md", 0), ("beijing_rent_directpay_2025.md", 1),
        ("beijing_loan_quota.md", 0),
    ]),
    ("杭州公积金租房提取每个月最多能提多少", "杭州", [
        ("hz_extract_optimize_2025.pdf", 0), ("hz_rent_limit_2023.pdf", 1),
        ("hz_extract_optimize_2025.pdf", 1), ("hangzhou_loan_quota.md", 1),
        ("hz_rent_limit_2023.pdf", 0),
    ]),
    ("住房公积金管理条例最近一次修订是什么时候", "全国", [
        ("national_housing_fund_regulation.md", 0), ("national_housing_fund_regulation.md", 3),
        ("national_housing_fund_regulation.md", 2), ("national_housing_fund_regulation.md", 7),
        ("national_housing_fund_regulation.md", 1),
    ]),
    ("上海保障性租赁住房原来的4500元月提取限额还适用吗", "上海", [
        ("shanghai_rent_extract.md", 1), ("shanghai_rent_extract.md", 0),
        ("shanghai_rent_extract.md", 2), ("shanghai_loan_quota.md", 1),
        ("shanghai_settlement.md", 0),
    ]),
    ("深圳现行住房公积金提取规定是哪一版什么时候施行", "深圳", [
        ("shenzhen_extract_rules.md", 0), ("shenzhen_extract_rules.md", 16),
        ("shenzhen_extract_rules.md", 3), ("shenzhen_extract_rules.md", 7),
        ("shenzhen_extract_rules.md", 13),
    ]),
]

_BASELINE_SYSTEM_PROMPT = (
    "你是「政策罗盘」社保公积金政策问答助手。严格遵守：\n"
    "1. 仅依据提供的参考资料回答，资料中没有的数字/结论不得编造；"
    "用户问的是特定城市而资料是其他城市时，须明确说明该城市暂未收录，"
    "不得把其他城市的规定当作该城市的答案；\n"
    "2. 引用来源时使用 [n] 标注对应资料编号；\n"
    "3. 优先引用现行有效文件；若仅找到已废止文件，须明确提示政策已更新；\n"
    "4. 涉及计算的必须采用【计算工具结果】中的数字；\n"
    "5. 资料不足以回答时明确说明知识库未收录，建议咨询 12329（公积金）/12333（人社）热线；\n"
    "6. 回答简洁分条，先给结论再给依据。"
)


async def test_zero_change_retrieval_orders(components):
    """不传新参数时，检索结果与改前基线逐字节一致（doc_id+seq 序）。"""
    retriever = components["retriever"]
    for q, city, expected in _BASELINE_ORDERS:
        r = await retriever.search(q, city=city, domain="公积金", use_rerank=False)
        got = [(c.chunk.doc_id, c.chunk.seq) for c in r.chunks]
        assert got == expected, f"检索序漂移: {q}"


def _handcrafted_state():
    return {
        "question": "杭州租房提取额度是多少",
        "documents": [{
            "chunk_id": "demo.md:0", "title": "测试办法", "doc_no": "杭公〔2024〕1号",
            "city": "杭州", "domain": "公积金", "status": "现行有效",
            "effective_date": "2024-01-01", "sections": ["第一条"], "pages": [],
            "is_valid": True, "text": "第一条（额度）租赁提取每月2000元。", "grade": 0.9,
        }],
    }


def test_zero_change_generation_prompt(components):
    """未命中对比意图时，生成提示词与改前基线逐字节一致（含用户消息）。"""
    agent = components["agent"]
    msgs, docs = agent.build_generation_messages(_handcrafted_state())
    assert msgs[0]["content"] == _BASELINE_SYSTEM_PROMPT
    assert msgs[1]["content"] == (
        "【参考资料】\n[1] （杭州·公积金·杭公〔2024〕1号·生效2024-01-01·现行有效）\n"
        "第一条（额度）租赁提取每月2000元。\n\n【用户问题】杭州租房提取额度是多少"
    )
    assert len(docs) == 1


def test_compare_prompt_adds_dual_version_instructions(components):
    """命中对比意图时：基线 prompt 原样保留 + 追加双版本作答口径段。"""
    agent = components["agent"]
    state = _handcrafted_state()
    state["route"] = {"version_compare": True}
    msgs, _ = agent.build_generation_messages(state)
    assert msgs[0]["content"].startswith(_BASELINE_SYSTEM_PROMPT)
    assert "【新旧版本对比作答口径】" in msgs[0]["content"]
    assert "现行" in msgs[0]["content"] and "生效日期" in msgs[0]["content"]
    # 用户消息（参考资料区）不受对比模式影响
    assert msgs[1]["content"] == (
        "【参考资料】\n[1] （杭州·公积金·杭公〔2024〕1号·生效2024-01-01·现行有效）\n"
        "第一条（额度）租赁提取每月2000元。\n\n【用户问题】杭州租房提取额度是多少"
    )


# ---------- 对比问答：双版本检索与引用卡 ----------


async def test_dual_retrieval_merges_both_versions(components):
    """对比意图命中 → 现行+已废止双路检索，documents 同时含两版（旧版 is_valid=False）。"""
    store = components["store"]
    retriever = components["retriever"]
    agent = components["agent"]
    await store.add_document("hangzhou_rent_2022.md", _DEPRECATED_DOC, components["llm"].embed)
    retriever.refresh()

    state = {
        "question": "杭州租房提取的老政策规定额度是多少，现在变了吗",
        "query": "杭州租房提取的老政策规定额度是多少，现在变了吗",
        "route": {"city": "杭州", "domain": "公积金", "version_compare": True},
        "retrieval_round": 0,
    }
    out = await agent.nodes["retrieve"](state)
    doc_ids = [d["chunk_id"].split(":")[0] for d in out["documents"]]
    assert "hangzhou_rent_2022.md" in doc_ids, "已废止旧版必须进入参考资料"
    assert any(d["is_valid"] for d in out["documents"]), "现行版必须进入参考资料"
    old_doc = next(d for d in out["documents"] if not d["is_valid"])
    assert "废止" in old_doc["status"] and old_doc["effective_date"] == "2022-01-01"
    ev = out["events"][-1]
    assert ev.get("old_version_hits", 0) >= 1

    # 对照：同一问题（去掉对比词）走单路检索——不触发双路（old_version_hits 缺省）
    state2 = {
        "question": "杭州租房提取规定额度是多少", "query": "杭州租房提取规定额度是多少",
        "route": {"city": "杭州", "domain": "公积金", "version_compare": False},
        "retrieval_round": 0,
    }
    out2 = await agent.nodes["retrieve"](state2)
    assert "old_version_hits" not in out2["events"][-1]
    assert all(d["is_valid"] for d in out2["documents"])


async def test_compare_answer_cites_both_versions(components):
    """端到端：对比问题的引用卡自然含两个版本（is_valid/effective_date 透传）。"""
    store = components["store"]
    retriever = components["retriever"]
    agent = components["agent"]
    await store.add_document("hangzhou_rent_2022.md", _DEPRECATED_DOC, components["llm"].embed)
    retriever.refresh()
    state = await agent.run("杭州租房提取的老政策规定额度是多少，现在变了吗")
    sources = state.get("sources", [])
    valid_flags = [s.get("is_valid") for s in sources]
    assert False in valid_flags, "引用卡必须包含已废止旧版（is_valid=False）"
    assert True in valid_flags, "引用卡必须包含现行版"
    old_src = next(s for s in sources if s.get("is_valid") is False)
    assert old_src["effective_date"] == "2022-01-01" and old_src["doc_no"] == "杭公积金委〔2022〕11号"
    assert not state.get("refusal")


# ---------- 评测集与 runner 透传 ----------


async def test_runner_passes_timeliness_filter_fields(components):
    """runner 对 as_of_date 字段透传：过滤生效（2025 版被排除 recall=0），无字段时零变化。"""
    from app.eval.runner import eval_retrieval

    retriever = components["retriever"]
    filtered = [{
        "id": 1, "type": "timeliness", "question": "杭州无房租赁公积金提取限额标准",
        "expect_docs": ["hz_extract_optimize_2025.pdf"], "as_of_date": "2024-06-01",
    }]
    res = await eval_retrieval(retriever, filtered)
    assert res["per_case"][0]["recall"] == 0.0  # 2025 版在 as_of 下被排除 → 过滤真实生效

    unfiltered = [{
        "id": 2, "type": "timeliness", "question": "杭州无房租赁公积金提取限额标准",
        "expect_docs": ["hz_extract_optimize_2025.pdf"],
    }]
    res2 = await eval_retrieval(retriever, unfiltered)
    assert res2["per_case"][0]["recall"] == 1.0  # 无字段 → 无过滤 → 行为与旧版一致


def test_eval_set_timeliness_meta():
    """时效评测集元守门：格式对齐 v2 字段、id 不与冻结集冲突、无库外城市题。"""
    from app.config import settings

    rows = [json.loads(l) for l in
            (DATA_DIR / "eval_set_timeliness.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert 13 <= len(rows) <= 18
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)) and min(ids) >= 201
    frozen_ids = set()
    for name in ("eval_set.jsonl", "eval_set_v2.jsonl"):
        for line in (DATA_DIR / name).read_text(encoding="utf-8").splitlines():
            if line.strip():
                frozen_ids.add(json.loads(line)["id"])
    assert not (set(ids) & frozen_ids), "时效集 id 不得与 v1/v2 冻结集冲突"
    supported = set(settings.cities)
    for r in rows:
        assert r["type"] in ("timeliness", "compare")
        assert r["question"] and "expect_keywords" in r
        assert r["expect_route"]["city"] in supported
        assert not any(c in r["question"] for c in
                       ("天津", "重庆", "苏州", "武汉", "西安")), "禁止出库外城市题"
        if "as_of_date" in r:
            date.fromisoformat(r["as_of_date"])
    types = [r["type"] for r in rows]
    assert types.count("compare") == 2 and types.count("timeliness") == 11
