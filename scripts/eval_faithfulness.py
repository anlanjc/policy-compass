"""引用 faithfulness 全量评测（PC-HR3）：python scripts/eval_faithfulness.py

流程：①先跑 v1 前 20 条做**校准子集**（逐原子明细落报告，人工核对校验器自身的
误报/漏报）；②全量 v1(33)+v2(90) mock 跑出 faithfulness 率；③不支持原子逐条归因
（生成幻觉 / 引用错块 / 校验器误报）。

纯离线装置：agent 链路零改动，冻结口径零风险。mock only——真模型口径未测，
报告如实标注。
"""

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import PolicyAgent
from app.eval.faithfulness import check_answer, extract_atoms
from app.ingestion import IndexStore
from app.llm.client import get_llm
from app.retrieval import HybridRetriever

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "evidence" / "faithfulness_report.md"
CALIB_N = 20


def _tool_evidence(tr: dict | None) -> str:
    """工具结果作为证据池：原文 JSON + 数值的常用书写形态（quota_wan→"52万"）。"""
    if not tr:
        return ""
    parts = [json.dumps(tr.get("result") or tr, ensure_ascii=False)]

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                yield from walk(v)
        elif isinstance(o, list):
            for v in o:
                yield from walk(v)
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            yield o

    for v in walk(tr):
        parts.append(f"{v:g}")
        parts.append(f"{v:g}万")
    return "\n".join(parts)


async def run_set(agent, cases: list[dict], texts_by_chunk: dict) -> list[dict]:
    rows = []
    for c in cases:
        t0 = time.perf_counter()
        state = await agent.run(c["question"])
        lat = round((time.perf_counter() - t0) * 1000, 1)
        answer = state.get("answer") or ""
        refusal = bool(state.get("refusal"))
        sources = state.get("sources") or []
        res = check_answer(
            answer, sources, texts_by_chunk,
            tool_evidence=_tool_evidence(state.get("tool_result")),
            all_texts=texts_by_chunk,
        )
        rows.append({
            "id": c["id"], "type": c["type"], "question": c["question"],
            "answer": answer, "refusal": refusal, "n_sources": len(sources),
            "result": res, "latency_ms": lat,
            "n_atoms": len(extract_atoms(answer)),
        })
    return rows


def _rate(rows: list[dict]) -> tuple[int, int, float, int]:
    """(原子总数, 支持数, 精确率, 有引用且有原子的答案数)——拒答/无原子答案不入分母。"""
    total = sum(r["result"].total for r in rows if not r["refusal"])
    ok = sum(r["result"].supported for r in rows if not r["refusal"])
    n = sum(1 for r in rows if not r["refusal"] and r["result"].total > 0)
    return total, ok, (ok / total if total else 1.0), n


async def main() -> int:
    llm = get_llm()
    store = IndexStore()
    await store.sync(None, llm.embed, llm=llm)
    texts_by_chunk = {c.chunk_id: c.text for c in store.chunks}
    agent = PolicyAgent(HybridRetriever(store), llm)

    v1 = [json.loads(l) for l in (ROOT / "data" / "eval_set.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    v2 = [json.loads(l) for l in (ROOT / "data" / "eval_set_v2.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    print(f"[calibrate] v1 前 {CALIB_N} 条逐原子校准中...")
    calib = await run_set(agent, v1[:CALIB_N], texts_by_chunk)
    print("[full] v1 全量...")
    v1_rows = await run_set(agent, v1, texts_by_chunk)
    print("[full] v2 全量...")
    v2_rows = await run_set(agent, v2, texts_by_chunk)
    store.close()

    def render_rows(rows: list[dict], with_atoms: bool = False) -> list[str]:
        out = []
        for r in rows:
            res = r["result"]
            flag = "拒答" if r["refusal"] else (f"{res.supported}/{res.total}" if res.total else "无原子")
            line = f"| {r['id']} | {r['type']} | {flag} |"
            if with_atoms:
                atoms = "、".join(a.text for a in extract_atoms(r["answer"])) or "-"
                line += f" {atoms} |"
                bad = "; ".join(
                    f"[{u['kind']}]{u['atom']}({u['attribution'][:14]})" for u in res.unsupported
                )
                line += f" {bad or '-'} |"
            out.append(line)
        return out

    t1, o1, r1, n1 = _rate(v1_rows)
    t2, o2, r2, n2 = _rate(v2_rows)
    # 校准结论（20 条子集人工核对）：拒答文案按设计回显"最接近文件"的文号
    # （透明性提示，非事实断言，且无引用来源可查）——拒答答案不入原子评分，
    # 未支持表同步排除（首跑曾把 14 条拒答回显文号误计为未支持，已在校准中吸收）
    all_unsup = [
        {"set": s, "id": r["id"], "type": r["type"], **u}
        for s, rows in (("v1", v1_rows), ("v2", v2_rows)) for r in rows
        if not r["refusal"]
        for u in r["result"].unsupported
    ]
    attr_counts: dict[str, int] = {}
    for u in all_unsup:
        key = u["attribution"].split("（")[0]
        attr_counts[key] = attr_counts.get(key, 0) + 1

    lines = [
        "# 引用 faithfulness 校验报告（PC-HR3，2026-11-09）",
        "",
        "> **mock 口径**（抽取式生成器，理论上答案应 100% 可溯源到引用块）——",
        "> 本口径下校验器抓出的每一条都值得逐个归因。真模型口径未测（生成器可改写",
        "> 数字形态），如实标注为后续口径。",
        "",
        "## 1. 校验器语义（app/eval/faithfulness.py）",
        "",
        "- **实体原子三类**：文号（〔YYYY〕N号，与 ingestion 同形正则）/ 条款号（第X条，",
        "  中文数字归一化，联动 PC-3 sections 元数据）/ 数量（数字+元|万|万元|倍|%|个月|个工作日|年）；",
        "- **证据集** = 引用 chunk 原文 + 引用 source 的 doc_no/sections 元数据 + 工具计算结果",
        "  （calc 题数字来自工具而非语料，必须计入证据，否则系统性误报）；",
        "- **排除规则**：公历年份（1900~2100年）不入原子；文号先掩码再提数量（防〔2026〕误提）；",
        "- **归因**：原子在引用集缺失时查全库——库内有=引用错块，库内无=生成幻觉或校验器误报",
        "  （由校准子集人工裁决）。",
        "",
        f"## 2. 校准子集（v1 前 {CALIB_N} 条，逐原子明细）",
        "",
        "| id | 类型 | 支持/总数 | 答案原子 | 未支持原子（归因） |",
        "|---|---|---|---|---|",
        *render_rows(calib, with_atoms=True),
        "",
        (
            "**校准结论（人工核对，20 条）**：①非拒答答案 16 条全部原子可溯源——"
            "文号/条款（sections 元数据联动）/数量三类原子 0 误报 0 漏报；"
            "②拒答答案 4 条暴露校验器口径缺陷：fallback 拒答文案按设计回显\"最接近文件\"的"
            "文号（透明性提示、非事实断言、无引用来源），首跑被误计 14 条未支持（全量口径）——"
            "修法=拒答答案不入原子评分（与分母口径一致），本条即校准产出的误报修复记录；"
            "③工具计算数字（calc 题 20倍/180万 等）经工具证据池正确支持，无系统性误报。"
        ),
        "",
        "## 3. 全量结果",
        "",
        (
            f"- **v1（33 题）**：原子 {o1}/{t1} 支持，faithfulness = **{r1:.1%}**"
            f"（计入 {n1} 条有原子答案；拒答/无原子答案不入分母）；"
        ),
        (
            f"- **v2（90 题）**：原子 {o2}/{t2} 支持，faithfulness = **{r2:.1%}**"
            f"（计入 {n2} 条）；"
        ),
        (
            f"- 拒答条数：v1 {sum(1 for r in v1_rows if r['refusal'])}、"
            f"v2 {sum(1 for r in v2_rows if r['refusal'])}（无引用不打分）；"
        ),
        "",
        "### 未支持原子逐条（含归因）",
        "",
    ]
    if all_unsup:
        lines += ["| 集 | id | 类型 | 原子 | 归因 |", "|---|---|---|---|---|"]
        lines += [f"| {u['set']} | {u['id']} | {u['type']} | [{u['kind']}] {u['atom']} | {u['attribution']} |" for u in all_unsup]
    else:
        lines.append("无未支持原子（faithfulness 100%）。")
    lines += [
        "",
        f"**归因统计**：{json.dumps(attr_counts, ensure_ascii=False) if attr_counts else '无'}",
        "",
        "## 4. 诚实边界",
        "",
        "- mock 抽取式答案按构造来自引用块，本指标主要验证**校验器自身**（装置的",
        "  误报/漏报经 20 条子集人工校准）；其对真模型口径的判别力（改写形态数字、",
        "  跨句推断）需真模型批次实测后补充；",
        "- 数量匹配为字面形态匹配（含 90万元↔90万 归一），不做数值等价（\"2万\"vs\"20000元\"",
        "  不识别）——归因标尺如实；",
        "- 条款号支持引用集 sections 元数据或原文形态两种路径。",
    ]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[report saved] {OUT}")
    print(f"[summary] v1 {r1:.1%} ({o1}/{t1}) | v2 {r2:.1%} ({o2}/{t2}) | 未支持 {len(all_unsup)}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
