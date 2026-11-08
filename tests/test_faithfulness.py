"""引用 faithfulness 校验器回归（PC-HR3）：原子抽取、归一化、证据判定与归因。"""

from __future__ import annotations

from app.eval.faithfulness import (
    check_answer,
    cn_to_int,
    extract_atoms,
)


def test_cn_to_int():
    assert cn_to_int("七") == 7
    assert cn_to_int("二十三") == 23
    assert cn_to_int("一百零五") == 105
    assert cn_to_int("10") == 10
    assert cn_to_int("两") == 2
    assert cn_to_int("abc") is None


def test_extract_atoms_kinds_and_exclusions():
    ans = (
        "根据杭房公委〔2025〕4号，租赁提取比例为50%，每人每月最高2000元，"
        "可贷额度按月均余额的20倍计算，期限最长30年。自2026年1月1日起施行。"
    )
    atoms = extract_atoms(ans)
    kinds = [(a.kind, a.text) for a in atoms]
    # 文号原子允许带语流前缀（"根据杭房公委…"）——支持判定用"末两字+〔年〕号"后缀针
    assert any(
        k == "doc_no" and t.endswith("〔2025〕4号") and "杭房公委" in t
        for k, t in kinds
    )
    assert ("amount", "50%") in kinds
    assert ("amount", "2000元") in kinds
    assert ("amount", "20倍") in kinds
    assert ("amount", "30年") in kinds  # 期限年有效
    assert not any(a.text == "2026年" for a in atoms), "公历年份不得成为数量原子"
    assert not any("2025" == a.text for a in atoms), "文号内的年份须被掩码"
    # 条款号
    arts = [a for a in extract_atoms("依据第七条与第10条办理") if a.kind == "article"]
    assert {a.article_no for a in arts} == {7, 10}


def test_amount_forms_normalization():
    """90万元 ↔ 90万 等价；且不得在更长数字内误命中（位数边界）。"""
    ev = "可贷额度为90万，存款利率2.6%"
    assert check_answer("最高90万元", [{"chunk_id": "c1"}], {"c1": ev}).rate == 1.0
    # "2000元" 不得命中 "12000元"
    ev2 = "一次性补贴12000元"
    res = check_answer("补贴为2000元", [{"chunk_id": "c1"}], {"c1": ev2})
    assert res.rate == 0.0 and res.unsupported[0]["atom"] == "2000元"


def test_check_answer_supported_via_metadata_and_tool():
    sources = [
        {"chunk_id": "c1", "doc_no": "杭房公委〔2025〕4号", "sections": ["第七条"]},
        {"chunk_id": "c2", "doc_no": "", "sections": []},
    ]
    texts = {"c1": "租赁提取比例为50%。", "c2": "办理时限为3个工作日。"}
    ans = (
        "依据杭房公委〔2025〕4号第七条，比例为50%，"
        "经计算最高可贷52万，办理需3个工作日。"
    )
    tool = '{"quota_wan": 52.0}\n52\n52万'
    res = check_answer(ans, sources, texts, tool_evidence=tool)
    assert res.total == 5 and res.rate == 1.0  # 文号(元数据)+条款(sections)+50%+52万(工具)+3个工作日


def test_check_answer_flags_and_attribution():
    texts = {
        "c1": "上海首套基本住房公积金贷款个人最高100万元。",
        "c2": "北京租房提取每月最高2000元。",
    }
    all_texts = dict(texts, c3="广州租房提取每人每月最高2000元。")
    # 幻觉：全库未见
    res = check_answer("上海最高可贷250万元。", [{"chunk_id": "c1"}], texts, all_texts=all_texts)
    assert res.unsupported and res.unsupported[0]["attribution"].startswith("生成幻觉")
    # 引用错块：事实在库（c2），但引用的是 c1
    res = check_answer("北京租房提取每月最高2000元。", [{"chunk_id": "c1"}], texts, all_texts=all_texts)
    assert res.unsupported and res.unsupported[0]["attribution"].startswith("引用错块")
    # 正确引用
    res = check_answer(
        "上海最高100万元，北京每月2000元。",
        [{"chunk_id": "c1"}, {"chunk_id": "c2"}], texts, all_texts=all_texts,
    )
    assert res.rate == 1.0
    # 无原子答案：rate 定义为 1.0（vacuous），total=0
    res = check_answer("以上内容摘自现行政策文件。", [], texts)
    assert res.total == 0 and res.rate == 1.0
