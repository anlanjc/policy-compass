"""向量腿消融（2026-11-05 S1）：BM25-only vs 混合检索，冻结评测集对比。

三阶段，费用分界明确（chat 全程 FORCE_MOCK，零 chat 花费）：
- bm25   零花费：CI 门禁同环境（POLICY_COMPASS_FORCE_MOCK=1 + EMBED_ENABLED=false）。
         产出 BM25-only 基线：检索消融（纯净/生产加权两组）+ 端到端 mock v1/v2。
- embed  一次性真实重嵌（需主人批准，须显式 --approved 或 AB_APPROVED=1）：
         原地 UPDATE chunks.embedding，输入逐字复刻 ingestion._ingest 的
         f"{title}\n{text}"（sync 对 hash 未变文档跳过，故走直连 UPDATE 而非重摄取，
         零 OCR/零跳过分支风险）。
- hybrid 混合腿：库内真实向量 + 查询向量走真实 embed（app.llm.client.get_llm
         临时替换为"真 embed + mock chat"组合客户端，进程内还原，app/ 零改动）。
         产出五组消融（补生产 BM25 对照）+ 端到端 mock v1/v2。

结果落 evidence/ablation_<phase>.json。口径解读与 CI 门禁基线关系见 docs/ablation.md。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import numpy as np  # noqa: E402

from app.config import settings  # noqa: E402
from app.eval.runner import ablation as runner_ablation  # noqa: E402
from app.eval.runner import eval_end2end, eval_retrieval, load_eval_set  # noqa: E402
from app.ingestion import IndexStore  # noqa: E402

EVAL_V1 = BASE / "data" / "eval_set.jsonl"
EVAL_V2 = BASE / "data" / "eval_set_v2.jsonl"
OUT_DIR = BASE / "evidence"


def _summary(res: dict) -> dict:
    return {k: res[k] for k in ("variant", "n", "recall@5", "mrr", "stale_leaks")}


async def _run_set(store: IndexStore, name: str, path: Path, hybrid: bool) -> dict:
    """单评测集：消融 + 端到端。hybrid=False 走 runner 原生两组（CI 门禁同环境）。"""
    from app.agent import PolicyAgent
    from app.llm.client import get_llm
    from app.retrieval import HybridRetriever

    cases = load_eval_set(path)
    retriever = HybridRetriever(store)
    # 注意口径：FORCE_MOCK 下 MockLLMClient.embed 不检查 embed_enabled，
    # current 标签=BM25+加权+mock 哈希向量腿（CI 门禁/e2e mock 同环境），
    # 与"生产 BM25-only"（real 客户端零向量→向量腿空）不同——docs/ablation.md 展开。
    label = "hybrid_full" if hybrid else "force_mock_current"
    current = await eval_retrieval(retriever, cases, variant=label)

    if not hybrid:
        abl = await runner_ablation(store, cases)  # bm25_plain / bm25+业务加权
    else:
        # 五组：在 runner 四组基础上补"生产 BM25"对照（加权开、向量关），
        # 使"BM25-only（生产）vs hybrid_full（生产）"同一进程内可直比。
        abl = []
        orig_vec = retriever._vector_search

        async def no_vec(q, k):
            return []

        retriever._vector_search = no_vec  # type: ignore[assignment]
        try:
            res = await eval_retrieval(retriever, cases, variant="bm25_production")
        finally:
            retriever._vector_search = orig_vec  # type: ignore[assignment]
        abl.append(res)
        # bm25_only / vector_only / hybrid_no_boost / hybrid_full（内部自管加权还原）
        abl += await runner_ablation(store, cases)

    llm = get_llm()  # FORCE_MOCK 下为 mock；hybrid 阶段为被补丁的组合客户端
    agent = PolicyAgent(retriever, llm)
    e2e = await eval_end2end(agent, cases)
    fail_ids = [
        {"id": i["id"], "type": i["type"], "refusal": i["refusal"]}
        for s in e2e["by_type"].values() for i in s["items"] if not i["pass"]
    ]
    return {
        "eval_set": name,
        "current": _summary(current),
        "ablation": [_summary(r) for r in abl],
        "e2e": {
            "total": e2e["total"],
            "overall_pass_rate": e2e["overall_pass_rate"],
            "p50_latency_ms": e2e["p50_latency_ms"],
            "by_type": {t: {"n": s["n"], "pass_rate": s["pass_rate"]}
                        for t, s in e2e["by_type"].items()},
            "fails": fail_ids,
        },
    }


def _db_vector_state(store: IndexStore) -> tuple[int, int]:
    norms = np.linalg.norm(store.matrix, axis=1)
    return len(store.chunks), int((norms > 0).sum())


async def phase_bm25() -> None:
    assert not settings.embed_enabled, "bm25 阶段要求 EMBED_ENABLED=false（CI 门禁同环境）"
    from app.llm.client import get_llm

    llm = get_llm()
    store = IndexStore()
    await store.sync(None, llm.embed, llm=llm)  # hash 未变全跳过，零副作用
    out = {"phase": "bm25", "env": {"force_mock": True, "embed_enabled": False},
           "sets": [
               await _run_set(store, "v1", EVAL_V1, hybrid=False),
               await _run_set(store, "v2", EVAL_V2, hybrid=False),
           ]}
    _dump("ablation_bm25.json", out)
    _print(out)


async def phase_embed(approved: bool) -> None:
    """一次性真实重嵌全部 chunk（原地 UPDATE，输入复刻 _ingest）。"""
    if not approved:
        sys.exit("拒绝执行：真实 embedding 属现金账，须主人批准后以 --approved（或 AB_APPROVED=1）显式运行")
    from app.llm.client import LLMClient

    settings.embed_enabled = True  # 进程内打开向量路（不改 .env），embed() 才走真实 API
    real = LLMClient()
    store = IndexStore()
    rows = store._conn.execute(
        "SELECT chunk_id, text FROM chunks ORDER BY doc_id, seq"
    ).fetchall()
    # 逐字复刻 _ingest 的 embed 输入：f"{c.title}\n{c.text}"——注意 splitter._pack
    # 构造 Chunk 时从不赋值 title（恒为默认 ""），故生产输入实为 "\n"+正文
    # （标题经 text 首行《标题》进入，功能无损，冗余前缀 quirk 记入 ablation.md）
    inputs = ["\n" + r[1] for r in rows]
    chars = sum(len(t) for t in inputs)
    # DashScope text-embedding-v4 批量上限 10（实测 400 InvalidParameter），
    # config.embed_batch_size=32 是 OpenAI/SiliconFlow 口径——本阶段钳到 10，查询侧单条不受影响
    batch_size = min(settings.embed_batch_size, 10)
    print(f"[embed] chunks={len(rows)} total_chars={chars} model={settings.embed_model}"
          f" base={settings.effective_embed_base} batch={batch_size}")
    t0 = time.perf_counter()
    vecs: list[list[float]] = []
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i : i + batch_size]
        vecs.extend(await real.embed(batch))
        print(f"[embed] batch {i // batch_size + 1}: cum={len(vecs)}")
    dt = time.perf_counter() - t0
    dims = {len(v) for v in vecs}
    assert dims == {settings.embed_dim}, f"维度不符：返回 {dims}，配置 {settings.embed_dim}"
    with store._conn:
        store._conn.executemany(
            "UPDATE chunks SET embedding=? WHERE chunk_id=?",
            [(np.asarray(v, dtype=np.float32).tobytes(), r[0]) for v, r in zip(vecs, rows)],
        )
    store.load_cache()
    n_total, n_nonzero = _db_vector_state(store)
    usage = {"embed_api_calls": (len(inputs) + batch_size - 1) // batch_size,
             "chunks": len(rows), "input_chars": chars, "elapsed_s": round(dt, 1),
             "dim": settings.embed_dim, "nonzero_after": n_nonzero}
    _dump("ablation_embed.json", {
        "phase": "embed", "usage_est": usage,
        "note": "token 用量：/embeddings 响应的 usage 字段 LLMClient 未捕获——以输入字符数上界估 token"
                "（中文 ≈1 token/字），花费以百炼账单为准",
    })
    print(f"[embed] done: {usage}, nonzero rows after = {n_nonzero}/{n_total}")
    await real.close()


async def phase_restore_mock() -> None:
    """把库向量恢复为 FORCE_MOCK 哈希口径（md5 纯函数，逐字节确定）。

    冻结 mock 数字（v2 e2e 91.1% 等）的记录前提是哈希向量库；消融实验把库换成
    真实向量后，mock 模式复现冻结数字需先恢复本口径。生产 real 模式
    （EMBED_ENABLED=false→查询零向量→向量腿空）不受库内容影响。
    """
    from app.llm.mock import MockLLMClient

    mock = MockLLMClient()
    store = IndexStore()
    rows = store._conn.execute(
        "SELECT chunk_id, text FROM chunks ORDER BY doc_id, seq"
    ).fetchall()
    inputs = ["\n" + r[1] for r in rows]  # 逐字复刻 _ingest：Chunk.title 恒空 → "\n"+正文
    vecs = await mock.embed(inputs)  # md5 词袋哈希，无随机、无网络
    with store._conn:
        store._conn.executemany(
            "UPDATE chunks SET embedding=? WHERE chunk_id=?",
            [(np.asarray(v, dtype=np.float32).tobytes(), r[0]) for v, r in zip(vecs, rows)],
        )
    store.load_cache()
    n_total, n_nonzero = _db_vector_state(store)
    print(f"[restore_mock] {len(rows)} chunks -> hash vectors, nonzero {n_nonzero}/{n_total}")
    _dump("ablation_restore_mock.json", {
        "phase": "restore_mock", "chunks": len(rows), "nonzero_after": n_nonzero,
        "note": "恢复 FORCE_MOCK 哈希向量口径（MockLLMClient.embed 确定性重建），"
                "冻结 mock 数字复现前提；hybrid 实验前须重跑 --phase embed",
    })


async def phase_hybrid() -> None:
    from app.llm.client import LLMClient
    from app.llm.mock import MockLLMClient
    import app.llm.client as client_mod

    store = IndexStore()
    n_total, n_nonzero = _db_vector_state(store)
    # 哈希向量（FORCE_MOCK 摄取残留）同样非零，非零计数区分不了真假——
    # 以 embed 阶段产物为闸门：没跑过 --phase embed 就拒绝混合腿
    embed_art = OUT_DIR / "ablation_embed.json"
    assert n_nonzero == n_total and n_total > 0 and embed_art.exists(), (
        f"库内向量为非真实状态（nonzero {n_nonzero}/{n_total}，可能是 FORCE_MOCK 哈希向量），"
        f"且缺 {embed_art.name}——先跑 --phase embed")

    class RealEmbedMockChat(MockLLMClient):
        """chat=mock（确定性判分口径不变），embed=真实 API（向量腿唯一真变量）。"""

        def __init__(self) -> None:
            super().__init__()
            self._real = LLMClient()

        async def embed(self, texts):  # type: ignore[override]
            return await self._real.embed(texts)

    settings.embed_enabled = True
    hybrid = RealEmbedMockChat()
    orig_get_llm = client_mod.get_llm
    client_mod.get_llm = lambda: hybrid  # _vector_search 函数内局部导入按调用时解析，替换生效
    try:
        out = {"phase": "hybrid",
               "env": {"force_mock": True, "embed_enabled": True,
                       "embed_model": settings.embed_model, "vector_weight": settings.vector_weight},
               "sets": [
                   await _run_set(store, "v1", EVAL_V1, hybrid=True),
                   await _run_set(store, "v2", EVAL_V2, hybrid=True),
               ]}
    finally:
        client_mod.get_llm = orig_get_llm
        await hybrid._real.close()
    _dump("ablation_hybrid.json", out)
    _print(out)


def _dump(name: str, data: dict) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    p = OUT_DIR / name
    p.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[saved] {p}")


def _print(out: dict) -> None:
    for s in out["sets"]:
        cur = s["current"]
        print(f"\n== {s['eval_set']} ==  current({cur['variant']}): "
              f"recall@5={cur['recall@5']:.1%} mrr={cur['mrr']:.3f} stale={cur['stale_leaks']}")
        for a in s["ablation"]:
            print(f"   {a['variant']:<22} n={a['n']:<3} recall@5={a['recall@5']:.1%} "
                  f"mrr={a['mrr']:.3f} stale_leaks={a['stale_leaks']}")
        e = s["e2e"]
        fail_ids = [f"#{f['id']}" for f in e["fails"]]
        print(f"   e2e overall={e['overall_pass_rate']:.1%} p50={e['p50_latency_ms']}ms "
              f"fails={fail_ids}")


async def main_async() -> int:
    ap = argparse.ArgumentParser(description="向量腿消融三阶段（bm25/embed/hybrid）")
    ap.add_argument("--phase", choices=["bm25", "embed", "hybrid", "restore_mock"], required=True)
    ap.add_argument("--approved", action="store_true",
                    help="embed 阶段的批准显式开关（等价 AB_APPROVED=1 环境变量）")
    args = ap.parse_args()
    if args.phase == "bm25":
        await phase_bm25()
    elif args.phase == "embed":
        await phase_embed(approved=args.approved or os.environ.get("AB_APPROVED") == "1")
    elif args.phase == "restore_mock":
        await phase_restore_mock()
    else:
        await phase_hybrid()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main_async()))
