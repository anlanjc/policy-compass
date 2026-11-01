"""混合检索：BM25 + 向量双路召回 → RRF 融合 → 业务元数据加权 → rerank。

三个业务驱动的排序信号（通用 RAG demo 通常缺失）：
1. 城市匹配加权：同名政策各地标准不同，命中路由城市 ×1.8；
2. 领域匹配加权：公积金/社保混检时 ×1.3；
3. 时效降权：已废止政策 ×0.3 保留可检索（可回答"政策变了什么"），
   但正常问答不会排在现行文件之前。

并行设计：查询 embedding（网络 IO）与 BM25 打分（CPU）用 asyncio.gather +
to_thread 并行，检索整体延迟 ≈ max(两路) 而非求和。
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..config import settings
from ..ingestion.indexer import IndexStore, IndexedChunk
from ..llm.stats import stats
from .bm25 import BM25Index, tokenize, tokenize_words
from .rerank import Reranker

logger = logging.getLogger(__name__)


@dataclass
class ScoredChunk:
    chunk: IndexedChunk
    fused_score: float
    bm25_rank: int | None = None  # 1-based，未入榜为 None
    vector_rank: int | None = None
    rerank_score: float | None = None
    final_score: float = 0.0

    def citation(self) -> str:
        parts = [self.chunk.title]
        if self.chunk.doc_no:
            parts.append(self.chunk.doc_no)
        if self.chunk.effective_date:
            parts.append(f"生效 {self.chunk.effective_date}")
        if not self.chunk.is_valid:
            parts.append("已废止")
        return "（".join(parts[:1]) + "）" if len(parts) == 1 else "，".join(parts)


@dataclass
class RetrieveResult:
    query: str
    chunks: list[ScoredChunk] = field(default_factory=list)
    expanded_terms: list[str] = field(default_factory=list)
    rerank_used: bool = False
    latency_ms: float = 0.0


class HybridRetriever:
    def __init__(self, store: IndexStore) -> None:
        self.store = store
        self.reranker = Reranker()
        self._lex = self._load_lexicon()
        self._bm25: BM25Index | None = None
        self.refresh()

    @staticmethod
    def _load_lexicon() -> dict[str, list[str]]:
        path = Path(settings.lexicon_path)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {}

    def refresh(self) -> None:
        """索引变更后重建 BM25（向量矩阵由 IndexStore 内存缓存维护）。"""
        self._bm25 = BM25Index([c.text for c in self.store.chunks])

    # ---------- 查询扩展 ----------

    def expand_query(self, query: str) -> tuple[dict[str, float], list[str]]:
        """词典扩展：黑话命中后以 0.6 权重注入规范词，原始词保持 1.0。

        PC-3 机制修复：注入词必须先经 tokenize_words 归一到索引分词空间。
        此前整串短语直接注入（如"委托提取还贷"/"住房公积金账户余额"），
        在 search_weighted 的 df 查找中静默失效——实测扩展词 df 命中率≈0
        （仅"贷款额度"等恰好是 jieba 原生 token 的词生效），扩展 feature
        自上线起近乎空转，slang 评测项的历史通过属排序巧合。
        """
        weighted = {t: 1.0 for t in set(tokenize(query))}
        expanded: list[str] = []
        for slang, formal in self._lex.items():
            if slang in query:
                for word in formal:
                    for tok in tokenize_words(word):
                        if tok not in weighted:
                            weighted[tok] = 0.6
                            expanded.append(tok)
        return weighted, expanded

    # ---------- 主入口 ----------

    async def search(
        self,
        query: str,
        city: str | None = None,
        domain: str | None = None,
        top_k: int | None = None,
        use_rerank: bool = True,
    ) -> RetrieveResult:
        import time

        start = time.perf_counter()
        top_k = top_k or settings.top_k_final
        result = RetrieveResult(query=query)

        weighted, expanded = self.expand_query(query)
        result.expanded_terms = expanded

        bm25_task = asyncio.get_running_loop().run_in_executor(
            None, lambda: self._bm25.search_weighted(weighted, settings.top_k_recall)
            if self._bm25 else []
        )
        vec_task = asyncio.ensure_future(self._vector_search(query, settings.top_k_recall))
        bm25_hits, vec_hits = await asyncio.gather(bm25_task, vec_task)

        fused = self._rrf_fuse(query, bm25_hits, vec_hits, city, domain)
        candidates = fused[: max(top_k * 3, 12)]

        if use_rerank and self.reranker.enabled and candidates:
            try:
                pairs = await self.reranker.rerank(
                    query, [sc.chunk.text for sc in candidates], top_n=top_k
                )
                by_idx = {i: sc for i, sc in enumerate(candidates)}
                for rank, (idx, score) in enumerate(pairs):
                    sc = by_idx[idx]
                    sc.rerank_score = score
                    sc.final_score = score
                result.chunks = sorted(
                    (by_idx[i] for i, _ in pairs), key=lambda sc: sc.final_score, reverse=True
                )[:top_k]
                result.rerank_used = True
            except Exception as e:  # 降级：rerank 失败不影响主链路
                stats.incr("rerank_fallbacks")
                logger.warning("rerank 降级: %s", e)
                result.chunks = candidates[:top_k]
        else:
            result.chunks = candidates[:top_k]

        result.latency_ms = round((time.perf_counter() - start) * 1000, 1)
        return result

    async def _vector_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        from ..llm.client import get_llm  # 局部导入避免循环依赖

        llm = get_llm()
        matrix = self.store.matrix
        if matrix.size == 0:
            return []
        vec = (await llm.embed([query]))[0]
        import numpy as np

        q = np.asarray(vec, dtype=np.float32)
        norm = float(np.linalg.norm(q))
        if norm == 0:
            return []
        sims = matrix @ (q / norm)
        order = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[i])) for i in order if sims[i] > 0]

    # ---------- RRF 融合 ----------

    def _rrf_fuse(
        self,
        query: str,
        bm25_hits: list,
        vec_hits: list[tuple[int, float]],
        city: str | None,
        domain: str | None,
    ) -> list[ScoredChunk]:
        k = settings.rrf_k
        fused: dict[int, ScoredChunk] = {}

        def ensure(idx: int) -> ScoredChunk:
            if idx not in fused:
                fused[idx] = ScoredChunk(chunk=self.store.chunks[idx], fused_score=0.0)
            return fused[idx]

        for rank, hit in enumerate(bm25_hits, start=1):
            sc = ensure(hit.index)
            sc.bm25_rank = rank
            sc.fused_score += 1.0 / (k + rank)
        for rank, (idx, _) in enumerate(vec_hits, start=1):
            sc = ensure(idx)
            sc.vector_rank = rank
            sc.fused_score += settings.vector_weight / (k + rank)

        # 政策标题=事项名，是比正文更强的主题信号（"租赁提取管理实施细则"）
        from .bm25 import cjk_bigrams

        # 多城问法（"北京和广州…分别是多少"）：路由只能给单城/全国，
        # query 文本里出现的城市名直接对照 chunk.city 加权——与
        # city_match_boost 同一信号源；单城路由时已有上两行覆盖，不重复施加
        mentioned_cities = (
            {c for c in settings.cities if c != "全国" and c in query}
            if not (city and city != "全国")
            else set()
        )

        q_terms = cjk_bigrams(query)
        for sc in fused.values():
            score = sc.fused_score
            if city and city != "全国" and sc.chunk.city == city:
                score *= settings.city_match_boost
            elif city and city != "全国" and sc.chunk.city != "全国":
                score *= 0.5  # 明确指向某城市时，其他城市文件强降权
            elif mentioned_cities and sc.chunk.city in mentioned_cities:
                score *= settings.city_match_boost
            if domain and domain != "通用" and sc.chunk.domain == domain:
                score *= settings.domain_match_boost
            title_terms = cjk_bigrams(sc.chunk.title)
            overlap = len(q_terms & title_terms)
            if q_terms and overlap >= 4 and overlap / len(q_terms) >= 0.25:
                score *= settings.title_match_boost
            if not sc.chunk.is_valid:
                score *= settings.invalid_doc_penalty
            sc.final_score = score if sc.rerank_score is None else sc.final_score

        out = sorted(fused.values(), key=lambda sc: sc.final_score, reverse=True)
        return out
