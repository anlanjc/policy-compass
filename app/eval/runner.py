"""评估闭环：检索指标 + 端到端行为指标 + 检索配置对比，一键产出 markdown 报告。

指标体系（对齐简历量化叙事）：
- 检索：recall@5 / MRR（确定性计算，不依赖 LLM，隔离评测变量）；
- 端到端：拒答准确率（库外题）、工具调用成功率（计算题，含额度数字核对）、
  答案关键词命中率、引用来源正确率；
- 配置对比：BM25-only / 向量-only / 混合无boost / 混合+业务boost 四组
  recall@5 对比——每个加权信号的效果都可见（消融实验）。

用法：
    python scripts/eval.py                # 全量：检索 + 端到端 + 对比
    python scripts/eval.py --retrieval    # 只跑检索指标（快）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from ..config import settings
from ..ingestion import IndexStore
from ..llm.client import get_llm

EVAL_SET = Path(__file__).resolve().parent.parent.parent / "data" / "eval_set.jsonl"


def load_eval_set() -> list[dict]:
    rows = []
    for line in EVAL_SET.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


# ---------- 检索指标 ----------


def recall_at_k(hits: list[str], expects: list[str], k: int) -> float:
    if not expects:
        return 1.0
    top = set(hits[:k])
    return len(top & set(expects)) / len(set(expects))


def mrr(hits: list[str], expects: list[str]) -> float:
    for rank, doc in enumerate(hits, start=1):
        if doc in expects:
            return 1.0 / rank
    return 0.0


async def eval_retrieval(retriever, cases: list[dict], variant: str = "full") -> dict:
    """对每题：规则路由（隔离 LLM 变量）→ 检索 → 计算 recall@5/MRR。"""
    from ..agent.router import rule_based_route

    recalls, mrrs = [], []
    per_case = []
    for c in cases:
        if c["type"] == "refuse":
            continue  # 库外题无目标文档，不参与检索指标
        route = rule_based_route(c["question"])
        r = await retriever.search(
            c["question"], city=route.city, domain=route.domain,
            top_k=5, use_rerank=False,
        )
        hit_docs = [sc.chunk.doc_id for sc in r.chunks]
        rec = recall_at_k(hit_docs, c["expect_docs"], 5)
        m = mrr(hit_docs, c["expect_docs"])
        # 时效题附加检查：已废止文档不得进入前 5
        stale_leak = bool(c.get("not_expect_docs")) and bool(
            set(hit_docs[:5]) & set(c["not_expect_docs"])
        )
        recalls.append(rec)
        mrrs.append(m)
        per_case.append({"id": c["id"], "type": c["type"], "recall": rec, "mrr": m,
                         "stale_leak": stale_leak})
    return {
        "variant": variant,
        "n": len(recalls),
        "recall@5": round(sum(recalls) / len(recalls), 4) if recalls else 0,
        "mrr": round(sum(mrrs) / len(mrrs), 4) if mrrs else 0,
        "stale_leaks": sum(1 for p in per_case if p["stale_leak"]),
        "per_case": per_case,
    }


# ---------- 消融：检索配置对比 ----------


async def ablation(store, cases: list[dict]) -> list[dict]:
    """四组消融：BM25-only / 向量-only / 混合无boost / 混合+业务boost。"""
    from ..retrieval import HybridRetriever

    results = []

    retriever = HybridRetriever(store)
    city_boost, domain_boost, title_boost, invalid_pen = (
        settings.city_match_boost, settings.domain_match_boost,
        settings.title_match_boost, settings.invalid_doc_penalty,
    )

    async def run_variant(name: str, bm25: bool, vec: bool, boost: bool) -> None:
        if not bm25:
            retriever._bm25 = None
        else:
            retriever.refresh()
        settings.city_match_boost = city_boost if boost else 1.0
        settings.domain_match_boost = domain_boost if boost else 1.0
        settings.title_match_boost = title_boost if boost else 1.0
        settings.invalid_doc_penalty = invalid_pen if boost else 1.0
        orig_vec = retriever._vector_search

        if not vec:
            async def no_vec(q, k):
                return []
            retriever._vector_search = no_vec  # type: ignore
        try:
            res = await eval_retrieval(retriever, cases, variant=name)
        finally:
            retriever._vector_search = orig_vec  # type: ignore
        results.append(res)

    if not settings.embed_enabled:
        # 无 embedding 额度：向量路休眠，消融退化为 BM25 加权前后对比
        await run_variant("bm25_plain", bm25=True, vec=False, boost=False)
        await run_variant("bm25+业务加权", bm25=True, vec=False, boost=True)
    else:
        await run_variant("bm25_only", bm25=True, vec=False, boost=False)
        await run_variant("vector_only", bm25=False, vec=True, boost=False)
        await run_variant("hybrid_no_boost", bm25=True, vec=True, boost=False)
        await run_variant("hybrid_full", bm25=True, vec=True, boost=True)

    settings.city_match_boost, settings.domain_match_boost = city_boost, domain_boost
    settings.title_match_boost, settings.invalid_doc_penalty = title_boost, invalid_pen
    return results


# ---------- 端到端评测 ----------


async def eval_end2end(agent, cases: list[dict]) -> dict:
    per_type: dict[str, list[dict]] = {}
    latencies = []
    for c in cases:
        t0 = time.perf_counter()
        state = await agent.run(c["question"])
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)
        item = {
            "id": c["id"], "type": c["type"], "latency_ms": round(lat, 1),
            "refusal": bool(state.get("refusal")),
            "answer": (state.get("answer") or "")[:200],
            "sources": [s.get("title", "") for s in state.get("sources", [])],
            "tool": (state.get("tool_result") or {}).get("tool"),
            "tool_result": (state.get("tool_result") or {}).get("result"),
        }
        # 判分
        if c["type"] == "refuse":
            item["pass"] = item["refusal"]
        elif c["type"] == "calc":
            tr = state.get("tool_result") or {}
            quota = (tr.get("result") or {}).get("quota_wan")
            item["pass"] = (
                tr.get("tool") == c.get("expect_tool")
                and quota is not None
                and abs(quota - c["expect_quota_wan"]) < 0.6
            )
        else:
            answer = item["answer"]
            item["pass"] = any(k in answer for k in c.get("expect_keywords", [])) or (
                c["type"] == "compare" and all(
                    k in answer for k in c.get("expect_keywords", [])
                )
            )
        per_type.setdefault(c["type"], []).append(item)

    type_stats = {}
    for t, items in per_type.items():
        type_stats[t] = {
            "n": len(items),
            "pass_rate": round(sum(1 for i in items if i["pass"]) / len(items), 4),
            "avg_latency_ms": round(sum(i["latency_ms"] for i in items) / len(items), 1),
            "items": items,
        }
    return {
        "total": len(cases),
        "overall_pass_rate": round(
            sum(1 for items in per_type.values() for i in items if i["pass"])
            / max(sum(len(v) for v in per_type.values()), 1),
            4,
        ),
        "p50_latency_ms": round(sorted(latencies)[len(latencies) // 2], 1) if latencies else 0,
        "by_type": type_stats,
    }


# ---------- 报告 ----------


def render_report(retrieval: dict | None, ablations: list[dict] | None,
                  end2end: dict | None, llm_mode: str, n_cases: int = 0) -> str:
    lines = [
        "# 评测报告",
        "",
        f"- 评测集：`data/eval_set.jsonl`（{n_cases or len(load_eval_set())} 题：事实/对比/计算/拒答/黑话/时效）",
        f"- LLM 模式：**{llm_mode}**（mock 模式用于链路验证，正式数字请以真实 API 运行）",
        f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    if ablations:
        lines += [
        "## 检索配置消融（recall@5 / MRR）",
        "",
        "| 配置 | n | recall@5 | MRR | 已废止文件泄漏进Top5 |",
        "|---|---|---|---|---|",
        ]
        for a in ablations:
            lines.append(
                f"| {a['variant']} | {a['n']} | {a['recall@5']:.1%} | {a['mrr']:.3f} | {a['stale_leaks']} |"
            )
        lines.append("")
    if retrieval:
        lines += [
            "## 当前配置检索明细",
            "",
            f"recall@5 = **{retrieval['recall@5']:.1%}**，MRR = **{retrieval['mrr']:.3f}**，"
            f"废止泄漏 = {retrieval['stale_leaks']} 题",
            "",
        ]
        bad = [p for p in retrieval["per_case"] if p["recall"] < 1.0 or p["stale_leak"]]
        if bad:
            lines += ["未满分题目：", ""]
            for p in bad:
                lines.append(f"- #{p['id']}（{p['type']}）recall={p['recall']:.2f} "
                             f"mrr={p['mrr']:.2f}{' 废止泄漏' if p['stale_leak'] else ''}")
            lines.append("")
    if end2end:
        lines += [
            "## 端到端行为指标",
            "",
            f"总体通过率：**{end2end['overall_pass_rate']:.1%}**，P50 延迟：{end2end['p50_latency_ms']}ms",
            "",
            "| 题型 | n | 通过率 | 平均延迟 |",
            "|---|---|---|---|",
        ]
        for t, s in end2end["by_type"].items():
            lines.append(f"| {t} | {s['n']} | {s['pass_rate']:.1%} | {s['avg_latency_ms']}ms |")
        lines.append("")
        fails = [i for s in end2end["by_type"].values() for i in s["items"] if not i["pass"]]
        if fails:
            lines += ["未通过题目：", ""]
            for i in fails:
                ans = i["answer"][:60].replace("\n", " ")
                lines.append(f"- #{i['id']}（{i['type']}，refusal={i['refusal']}）：{ans}")
            lines.append("")
    return "\n".join(lines)


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PolicyCompass 评测")
    parser.add_argument("--retrieval", action="store_true", help="只跑检索指标与消融")
    parser.add_argument("--out", default="docs/eval_report.md")
    args = parser.parse_args(argv)

    cases = load_eval_set()
    llm = get_llm()
    store = IndexStore()
    # 始终增量同步：语料/词典更新后 hash 变化的文档自动重嵌，保证评测对象最新
    await store.sync(None, llm.embed, llm=llm)

    from ..retrieval import HybridRetriever

    retriever = HybridRetriever(store)
    current_label = "hybrid_full" if settings.embed_enabled else "bm25+业务加权"
    retrieval_res = await eval_retrieval(retriever, cases, variant=current_label)
    abl = await ablation(store, cases)
    e2e = None
    if not args.retrieval:
        from ..agent import PolicyAgent

        agent = PolicyAgent(retriever, llm)
        e2e = await eval_end2end(agent, cases)

    report = render_report(retrieval_res, abl, e2e, settings.effective_llm_mode, len(cases))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[report saved] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
