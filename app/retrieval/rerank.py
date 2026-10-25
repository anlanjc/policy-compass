"""Rerank：调用 bge-reranker 类交叉编码器 API，失败自动降级。

降级策略：网络/服务异常时捕获于 HybridRetriever，跳过重排直接用 RRF 融合序
（stats.rerank_fallbacks 记数）——检索链路任何一环失效都不应导致整体不可用。
"""

from __future__ import annotations

import httpx

from ..config import settings
from ..llm.stats import stats


class Reranker:
    def __init__(self) -> None:
        self._http = httpx.AsyncClient(
            base_url=settings.rerank_api_base.rstrip("/"),
            headers={"Authorization": f"Bearer {settings.rerank_api_key}"},
            timeout=settings.rerank_timeout,
        )

    @property
    def enabled(self) -> bool:
        return settings.rerank_enabled and bool(settings.rerank_api_key)

    async def rerank(
        self, query: str, documents: list[str], top_n: int | None = None
    ) -> list[tuple[int, float]]:
        """返回 [(documents 下标, 相关性分数)]，按分数降序。"""
        payload = {
            "model": settings.rerank_model,
            "query": query,
            "documents": documents,
            "top_n": top_n or len(documents),
        }
        stats.incr("rerank_calls")
        resp = await self._http.post("/rerank", json=payload)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        return [(r["index"], float(r["relevance_score"])) for r in results]

    async def close(self) -> None:
        await self._http.aclose()
