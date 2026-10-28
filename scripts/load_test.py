"""mock 压测：httpx + asyncio 并发压 /api/chat，产出 evidence/load_test.md。

两种模式：
- http（默认）：子进程起 uvicorn（强制 mock + 指定模拟延迟），真实 HTTP 栈压测——
  包含 FastAPI/事件循环/SSE 序列化的完整开销；
- inline：进程内 asyncio 并发调 agent.run——无法起服务时的退化路径，
  不含 HTTP 栈开销（报告会注明差异）。

延迟形态：默认 mock 零延迟（测调度开销上限）；--llm-delay 1.5 模拟真实 LLM
每次调用 ~1.5s（链路 8~9 次调用，串行部分构成延迟下限）——报告标注"模拟延迟"。

用法：
    python scripts/load_test.py                          # http 模式 20/50/100 并发
    python scripts/load_test.py --llm-delay 1.5          # 带模拟 LLM 延迟
    python scripts/load_test.py --mode inline            # 进程内退化模式
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

QUESTIONS = [
    "杭州公积金租房提取每个月最多能提多少",
    "北京公积金缴存8年余额9万能贷多少",
    "上海租房提取每月限额是多少",
    "社保断缴有什么影响",
    "公积金异地转移需要什么材料",
]


def _pct(sorted_xs: list[float], p: float) -> float:
    if not sorted_xs:
        return 0.0
    idx = min(len(sorted_xs) - 1, int(len(sorted_xs) * p))
    return sorted_xs[idx]


async def _run_phase(base_url: str, concurrency: int, min_requests: int,
                     questions: list[str]) -> dict:
    """单档并发：直到发满 min_requests 个成功/失败请求为止。"""
    latencies: list[float] = []
    errors = 0
    lock = asyncio.Lock()
    sent = 0
    timeout = httpx.Timeout(120.0)

    async def worker(client: httpx.AsyncClient, wid: int) -> None:
        nonlocal sent, errors
        while True:
            async with lock:
                if sent >= min_requests:
                    return
                sent += 1
                i = sent
            q = questions[(i + wid) % len(questions)]
            t0 = time.perf_counter()
            try:
                r = await client.post(
                    f"{base_url}/api/chat", json={"question": q, "stream": False}
                )
                r.raise_for_status()
                data = r.json()
                if not data.get("answer") and not data.get("refusal"):
                    raise ValueError("空回答")
                lat = (time.perf_counter() - t0) * 1000
                async with lock:
                    latencies.append(lat)
            except Exception:
                lat = (time.perf_counter() - t0) * 1000
                async with lock:
                    errors += 1
                    latencies.append(lat)

    t_start = time.perf_counter()
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        await asyncio.gather(*(worker(client, w) for w in range(concurrency)))
    elapsed = time.perf_counter() - t_start
    lat_sorted = sorted(latencies)
    total = len(latencies)
    return {
        "concurrency": concurrency,
        "requests": total,
        "elapsed_s": round(elapsed, 1),
        "qps": round(total / elapsed, 1),
        "p50_ms": round(_pct(lat_sorted, 0.50), 1),
        "p95_ms": round(_pct(lat_sorted, 0.95), 1),
        "p99_ms": round(_pct(lat_sorted, 0.99), 1),
        "error_rate": round(errors / total, 4) if total else 0,
    }


async def _inline_phase(concurrency: int, min_requests: int, questions: list[str]) -> dict:
    """退化模式：进程内并发调 agent.run（无 HTTP 栈）。"""
    from app.config import settings
    from app.ingestion import IndexStore
    from app.retrieval import HybridRetriever
    from app.agent import PolicyAgent
    from app.llm.mock import MockLLMClient

    llm = MockLLMClient()
    store = IndexStore()
    await store.sync(settings.corpus_dir, llm.embed, llm=llm)
    agent = PolicyAgent(HybridRetriever(store), llm)

    latencies: list[float] = []
    errors = 0
    lock = asyncio.Lock()
    sent = 0

    async def worker(wid: int) -> None:
        nonlocal sent, errors
        while True:
            async with lock:
                if sent >= min_requests:
                    return
                sent += 1
                i = sent
            q = questions[(i + wid) % len(questions)]
            t0 = time.perf_counter()
            try:
                state = await agent.run(q)
                if not state.get("answer") and not state.get("refusal"):
                    raise ValueError("空回答")
                lat = (time.perf_counter() - t0) * 1000
                async with lock:
                    latencies.append(lat)
            except Exception:
                lat = (time.perf_counter() - t0) * 1000
                async with lock:
                    errors += 1
                    latencies.append(lat)

    t_start = time.perf_counter()
    await asyncio.gather(*(worker(w) for w in range(concurrency)))
    elapsed = time.perf_counter() - t_start
    lat_sorted = sorted(latencies)
    total = len(latencies)
    return {
        "concurrency": concurrency,
        "requests": total,
        "elapsed_s": round(elapsed, 1),
        "qps": round(total / elapsed, 1),
        "p50_ms": round(_pct(lat_sorted, 0.50), 1),
        "p95_ms": round(_pct(lat_sorted, 0.95), 1),
        "p99_ms": round(_pct(lat_sorted, 0.99), 1),
        "error_rate": round(errors / total, 4) if total else 0,
    }


async def _bottleneck_probe() -> str:
    """瓶颈归因探针：单请求分解——纯检索 vs 全链路，量化 LLM 串行占比。"""
    from app.config import settings
    from app.ingestion import IndexStore
    from app.retrieval import HybridRetriever
    from app.agent import PolicyAgent
    from app.llm.mock import MockLLMClient

    llm = MockLLMClient()
    store = IndexStore()
    await store.sync(settings.corpus_dir, llm.embed, llm=llm)
    retriever = HybridRetriever(store)
    agent = PolicyAgent(retriever, llm)
    q = QUESTIONS[0]

    t0 = time.perf_counter()
    for _ in range(20):
        await retriever.search(q, city="杭州", domain="公积金", top_k=5, use_rerank=False)
    t_retrieval = (time.perf_counter() - t0) / 20 * 1000

    t0 = time.perf_counter()
    await agent.run(q)
    t_agent = (time.perf_counter() - t0) * 1000

    return (
        f"单请求分解（零模拟延迟，20 次取均值）：纯检索 {t_retrieval:.1f}ms/次，"
        f"全链路 {t_agent:.1f}ms/次——检索占比 {t_retrieval / t_agent:.1%}。"
        "链路含 route→retrieve→grade(5 chunk 并行评分)→(tool)→generate→verify "
        "约 8~9 次 LLM 调用，节点串行推进；设置 --llm-delay 后延迟≈串行调用数×单次延迟，"
        "验证瓶颈在 LLM 调用而非检索（检索为 BM25 内存查询 + SQLite 读，毫秒级）。"
    )


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["http", "inline"], default="http")
    parser.add_argument("--concurrency", default="20,50,100")
    parser.add_argument("--min-requests", type=int, default=500)
    parser.add_argument("--llm-delay", type=float, default=0.0,
                        help="模拟每次 LLM 调用延迟秒数（报告将标注『模拟延迟』）")
    parser.add_argument("--out", default="evidence/load_test.md")
    args = parser.parse_args()

    import os

    os.environ["POLICY_COMPASS_FORCE_MOCK"] = "1"
    os.environ["LLM_MOCK_DELAY"] = str(args.llm_delay)
    from app.config import reset_settings_cache, settings

    reset_settings_cache()
    settings.llm_mock_delay = args.llm_delay

    base_url = "http://127.0.0.1:8765"
    server = None
    if args.mode == "http":
        env = {**os.environ, "POLICY_COMPASS_FORCE_MOCK": "1",
               "LLM_MOCK_DELAY": str(args.llm_delay), "APP_PORT": "8765"}
        server = subprocess.Popen(
            [sys.executable, "scripts/serve.py"],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        # 等服务就绪
        ok = False
        async with httpx.AsyncClient(timeout=2.0, trust_env=False) as c:
            for _ in range(60):
                try:
                    r = await c.get(f"{base_url}/api/health")
                    ok = r.status_code == 200
                    break
                except Exception:
                    await asyncio.sleep(0.5)
        if not ok:
            print("[warn] 服务未就绪，退化为 inline 进程内模式")
            args.mode = "inline"

    rows = []
    try:
        for conc in [int(x) for x in args.concurrency.split(",")]:
            if args.mode == "http":
                row = await _run_phase(base_url, conc, args.min_requests, QUESTIONS)
            else:
                row = await _inline_phase(conc, args.min_requests, QUESTIONS)
            rows.append(row)
            print(f"[phase] {row}")
    finally:
        if server:
            server.terminate()

    probe = await _bottleneck_probe()
    mode_label = "HTTP 全栈（uvicorn 子进程，强制 mock）" if args.mode == "http" else \
        "进程内 asyncio 直调 agent.run（无 HTTP 栈开销，退化模式）"
    delay_label = f"，**模拟延迟 {args.llm_delay}s/次调用**" if args.llm_delay else ""

    lines = [
        "# 压测报告（mock 模式）",
        "",
        f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M')}",
        f"- 模式：{mode_label}{delay_label}",
        f"- 压测对象：POST /api/chat（stream=false，全链路含检索/评分/工具/生成/校验）",
        f"- 每档请求量：≥{args.min_requests} 或打满为止；问题池 5 条（事实/计算/黑话混合）",
        "",
        "## 结果",
        "",
        "| 并发 | 请求数 | 耗时s | QPS | P50(ms) | P95(ms) | P99(ms) | 错误率 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['concurrency']} | {r['requests']} | {r['elapsed_s']} | {r['qps']} | "
            f"{r['p50_ms']} | {r['p95_ms']} | {r['p99_ms']} | {r['error_rate']:.1%} |"
        )
    lines += ["", "## 瓶颈归因", "", probe, ""]
    if args.llm_delay:
        lines.append(
            f"> 注：本轮开启 --llm-delay {args.llm_delay}（每次 LLM 调用注入 {args.llm_delay}s 延迟），"
            "QPS/延迟反映的是「LLM 串行调用主导」的形态，不代表零延迟 mock 的调度上限；"
            "生产真实延迟形态介于两轮之间。"
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report saved] {out}")
    return 0


if __name__ == "__main__":
    asyncio.run(main())
