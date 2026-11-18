"""mock 抽取式生成（MockLLMClient._generate）回归钉：v2 冻结集 #138 脆性修复。

1B5 车道归因（1B4 冻结口径失败集 #138，检索 recall=1.0 而判分失败）：
_generate 曾只消化生成上下文前 3 个资料块（``list(chunks.items())[:3]``），
对检索排序脆性——#138「广州和深圳租房提取额度规则有什么区别」中广州块排第 5 位，
"2000"额度句永无机会进答案；#131 同根因（深圳块排第 4 位，"百分之六十五"出窗）。
修复=全量块读取（生成上下文已过 grade>=0.4 相关性闸，mock 应与真实模型职责对齐
读完整个参考资料）；对前 3 块的选句逻辑逐字节零变化（新增内容纯追加，见
test_generate_prefix_unchanged / 全量 90 条答案结构对比）。
另顺带删除 _generate 末尾 return 后的死代码（不可达，零行为面，
test_generate_no_dead_code 静态钉防复发）。
"""

from __future__ import annotations

import ast
import inspect
import textwrap

from app.llm.mock import MockLLMClient


def _gen_prompt(blocks: list[str], question: str) -> str:
    """按 build_generation_messages 的上下文形态拼最小 prompt。"""
    body = "\n".join(blocks)
    return f"【参考资料】\n{body}\n\n【用户问题】{question}"


# ---------- 单元级：_generate 抽取行为 ----------


def test_generate_reads_all_context_blocks():
    """修复主钉：关键词句在第 5 块（旧 [:3] 窗外）也必须进答案。

    旧实现只消化前 3 块，第 5 块的"2000"额度句被静默丢弃（#138 失败形态）。
    """
    prompt = _gen_prompt(
        [
            (
                "[1] （杭州·公积金·甲〔2025〕1号·生效2025-01-01·现行有效）\n"
                "《甲办法》\n杭州公积金缴存基数每年调整一次。缴存单位应当按月足额缴存。"
            ),
            (
                "[2] （杭州·公积金·乙〔2025〕2号·生效2025-01-01·现行有效）\n"
                "《乙办法》\n杭州公积金可以用于购买自住住房。提取时需提供身份材料。"
            ),
            (
                "[3] （杭州·公积金·丙〔2025〕3号·生效2025-01-01·现行有效）\n"
                "《丙办法》\n杭州公积金贷款期限最长不超过30年。借款人应当按期偿还本息。"
            ),
            (
                "[4] （杭州·公积金·丁〔2025〕4号·生效2025-01-01·现行有效）\n"
                "《丁办法》\n杭州灵活就业人员可以自愿缴存住房公积金。缴存满一年可申请贷款。"
            ),
            (
                "[5] （广州·公积金·戊〔2025〕5号·生效2025-01-01·现行有效）\n"
                "《戊办法》\n广州租房提取限额提高至每人每月2000元。职工连续足额缴存满三个月可申请。"
            ),
        ],
        "广州和杭州租房提取额度有什么区别",
    )
    out = MockLLMClient()._generate(prompt)
    assert "2000" in out  # 第 5 块关键句必须进答案（#138 形态）
    assert "[5]" in out  # 且标注第 5 块来源编号


def test_generate_prefix_unchanged():
    """前 3 块行为零变化：块序与块内 top-2 选句保持原逻辑，新增块纯追加。"""
    prompt = _gen_prompt(
        [
            (
                "[1] （杭州·公积金·甲〔2025〕1号·生效2025-01-01·现行有效）\n"
                "《甲办法》\n杭州租房提取额度提高至每人每月1500元。本办法自2025年1月1日起施行。"
            ),
            (
                "[2] （杭州·公积金·乙〔2025〕2号·生效2025-01-01·现行有效）\n"
                "《乙办法》\n杭州公积金可以用于购买自住住房。提取时需提供身份材料。"
            ),
            (
                "[3] （杭州·公积金·丙〔2025〕3号·生效2025-01-01·现行有效）\n"
                "《丙办法》\n杭州公积金贷款期限最长不超过30年。借款人应当按期偿还本息。"
            ),
            (
                "[4] （广州·公积金·丁〔2025〕4号·生效2025-01-01·现行有效）\n"
                "《丁办法》\n广州租房提取限额提高至每人每月2000元。职工连续足额缴存满三个月可申请。"
            ),
        ],
        "广州和杭州租房提取额度有什么区别",
    )
    out = MockLLMClient()._generate(prompt)
    # 旧窗口（前 3 块）的选句结果原样保留，且先于新增块出现
    assert out.index("杭州租房提取额度提高至每人每月1500元") < out.index("2000")
    assert out.index("杭州公积金可以用于购买自住住房") < out.index("2000")
    assert out.index("杭州公积金贷款期限最长不超过30年") < out.index("2000")


def test_generate_single_block_golden():
    """既有单城抽取零变化：单块 prompt 的输出与修复前逐字节一致（黄金串）。"""
    prompt = _gen_prompt(
        [
            (
                "[1] （杭州·公积金·甲〔2025〕1号·生效2025-01-01·现行有效）\n"
                "《甲办法》\n杭州租房提取额度提高至每人每月1500元。本办法自2025年1月1日起施行。"
            ),
        ],
        "杭州租房提取额度是多少",
    )
    expected = (
        "根据现行政策文件，为您整理如下：\n"
        "杭州租房提取额度提高至每人每月1500元。本办法自2025年1月1日起施行。[1]\n"
        "以上内容摘自知识库现行有效的政策文件，具体办理以当地公积金/社保中心为准。"
    )
    assert MockLLMClient()._generate(prompt) == expected


def test_generate_excludes_metadata_title_footer_lines():
    """语料元数据行（【开头）、《标题》行、OCR 页脚不进答案（既有过滤不回退）。"""
    prompt = _gen_prompt(
        [
            (
                "[1] （杭州·公积金·甲〔2025〕1号·生效2025-01-01·现行有效）\n"
                "《甲测试办法》\n【来源URL】https://example.com/doc\n【抓取时间】2026-01-01\n"
                "第一条 租房提取额度为每月1500元。\n— 3 —\n杭州住房公积金管理中心\n2025年1月1日"
            ),
        ],
        "租房提取额度是多少",
    )
    out = MockLLMClient()._generate(prompt)
    assert "1500" in out
    assert "《" not in out
    assert "【" not in out
    assert "— 3 —" not in out  # OCR 页脚（真实语料格式，单破折号对）不得进答案


def test_generate_no_relevant_chunks_returns_fallback():
    """无有效资料块（每块均过短）→ 既有兜底句，零变化。"""
    prompt = _gen_prompt(["[1] 短"], "杭州租房提取额度是多少")
    out = MockLLMClient()._generate(prompt)
    assert out == "根据现有政策文件，未找到与该问题直接相关的条文。"


def test_generate_no_dead_code():
    """死代码删除静态钉：_generate 顶层函数体以 return 收尾，且无 return 后语句。

    1B5 车道顺带删除的尾部死代码（return 后重复 append+return，不可达零行为）
    ——防其再度积累。
    """
    src = textwrap.dedent(inspect.getsource(MockLLMClient._generate))
    fn = ast.parse(src).body[0]
    assert isinstance(fn.body[-1], ast.Return)
    assert not any(isinstance(st, ast.Return) for st in fn.body[:-1])


# ---------- 链路级：v2 冻结集真实题目（components 夹具 = 真实语料临时库） ----------


async def test_v2_138_guangzhou_shenzhen_compare(components):
    """#138 端到端复现钉：双城 compare → expect_docs 双命中 + expect_keywords 命中。"""
    agent = components["agent"]
    state = await agent.run("广州和深圳租房提取额度规则有什么区别")
    assert not state["refusal"]
    assert state.get("verified") is True
    answer = state["answer"]
    assert "2000" in answer  # 广州块（第 5 位）"每人每月2000元"必须进答案
    titles = [s["title"] for s in state["sources"]]
    assert "深圳市住房公积金提取管理规定" in titles
    assert "广州住房公积金管理中心关于调整住房公积金政策的通知" in titles


async def test_v2_131_shenzhen_shanghai_compare(components):
    """#131 同根因钉：深圳块排第 4 位出旧窗 → "百分之六十五"须进答案。"""
    agent = components["agent"]
    state = await agent.run("深圳和上海无房租房提取的额度规则有什么区别")
    assert not state["refusal"]
    titles = [s["title"] for s in state["sources"]]
    assert "深圳市住房公积金提取管理规定" in titles
    assert any("沪公积金管委会〔2024〕6号" in t for t in titles)
    assert "百分之六十五" in state["answer"]


async def test_compare_two_city_stable_no_regression(components):
    """双城 compare 零回归：#127（修复前后均通过的存量题）行为不变。"""
    agent = components["agent"]
    state = await agent.run("北京和广州无房租房提取每月限额分别是多少")
    assert not state["refusal"]
    assert "2000" in state["answer"]
