"""混合检索：BM25 + 向量双路召回 → RRF 融合 → 业务元数据加权 → rerank。

四个业务驱动的排序信号（通用 RAG demo 通常缺失）：
1. 城市匹配加权：同名政策各地标准不同，命中路由城市 ×1.8；
2. 领域匹配加权：公积金/社保混检时 ×1.3；
3. 时效降权：已废止政策 ×0.3 保留可检索（可回答"政策变了什么"），
   但正常问答不会排在现行文件之前；
4. 注入信任降权：命中注入模式库的块 ×injection_trust_penalty 沉底
   （1B3-PC 挤占修复，防关键词堆砌毒块占用 top-k；正式语料零命中，
   冻结口径不受扰动，见 app/agent/injection.py 与 evidence 报告 §4）。

时效性引擎（1B5 收口，以上信号默认全部不变）：
5. 检索期过滤：status_filter（现行有效/已废止）与 as_of_date（时点口径）
   两个可选参数在融合入口裁剪候选——新旧版本对比问答的"已废止路"与
   历史时点问答由此实现；
6. 时序并列消歧：同 doc_no/同标题的新旧版本融合分并列时按 date_rank
   （生效日期序数）降序稳定重排（_newer_version_first_on_ties），新版本
   确定性靠前——date_rank 的首个正式消费者。

并行设计：查询 embedding（网络 IO）与 BM25 打分（CPU）用 asyncio.gather +
to_thread 并行，检索整体延迟 ≈ max(两路) 而非求和。
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from ..config import settings
from ..ingestion.indexer import IndexStore, IndexedChunk
from ..ingestion.metadata import date_rank, parse_effective_date
from ..llm.stats import stats
from .bm25 import BM25Index, tokenize, tokenize_words
from .rerank import Reranker

if TYPE_CHECKING:
    from ..ingestion.indexer import IndexSnapshot

logger = logging.getLogger(__name__)

# 当前检索的快照上下文（S3 评审 PC-2）：search 入口捕获，经 asyncio 上下文
# 传给向量路协程——并发检索各用各的快照，互不串版本
_search_snap: ContextVar["IndexSnapshot | None"] = ContextVar("search_snap", default=None)


def _coerce_as_of(value: date | str | None) -> date | None:
    """as_of_date 入参归一：date 直传，ISO 字符串解析，非法显式报错（调用方 bug 不静默）。"""
    if value is None:
        return None
    if isinstance(value, datetime):  # datetime 是 date 子类，须先判
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        parsed = parse_effective_date(value)
        if parsed is None:
            raise ValueError(f"as_of_date 格式非法: {value!r}（应为 YYYY-MM-DD）")
        return parsed
    raise TypeError(f"as_of_date 类型非法: {type(value).__name__}")


@dataclass
class ScoredChunk:
    chunk: IndexedChunk
    fused_score: float
    bm25_rank: int | None = None  # 1-based，未入榜为 None
    vector_rank: int | None = None
    rerank_score: float | None = None
    final_score: float = 0.0
    # 命中注入模式库（app/agent/injection.py，PC-HR2）：检索融合层信任降权依据，
    # 下游 retrieve_node 的行级消毒与 grade=0 隔离以此同源（1B3-PC 挤占修复）
    injection_flagged: bool = False

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
        self._bm25_built_for = -1  # _bm25 构建时所基于的快照版本（PC-2）
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
        self._bm25_built_for = self.store.snapshot.version

    def _bm25_for(self, snap: IndexSnapshot) -> BM25Index | None:
        """取与快照同版本的 BM25 索引（PC-2）：库变更后首次检索自动按该快照重建。

        `self._bm25 = None`（消融打补丁禁用 BM25）语义保持：显式 None 视为禁用，
        不自动重建；refresh() 恢复并重新盖章版本。
        """
        if self._bm25 is None:
            return None
        if self._bm25_built_for != snap.version:
            self._bm25 = BM25Index([c.text for c in snap.chunks])
            self._bm25_built_for = snap.version
        return self._bm25

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
        status_filter: str | None = None,
        as_of_date: date | str | None = None,
    ) -> RetrieveResult:
        """混合检索主入口。

        时效过滤参数（1B5 政策时效性引擎；默认 None = 不过滤，检索路径与
        旧版逐字节同效——冻结评测口径依赖此）：
        - status_filter: "现行有效" → 只保留 is_valid 块；"已废止" → 只保留
          已废止块（新旧版本对比问答的已废止路由此实现）；其余值不过滤；
        - as_of_date: 只返回 effective_date <= as_of 的块（date 或 ISO 字符串）。
          无生效日期/日期非法的块一律排除——无法证明其在 as_of 时点已生效，
          时点问答宁缺勿错（答错历史口径即事故）；None=不过滤。
        """
        import time

        start = time.perf_counter()
        top_k = top_k or settings.top_k_final
        result = RetrieveResult(query=query)
        as_of = _coerce_as_of(as_of_date)

        weighted, expanded = self.expand_query(query)
        result.expanded_terms = expanded

        # 快照捕获（PC-2）：本请求全程只用这一个自洽版本——库中途变更不影响在飞请求
        snap = self.store.snapshot
        token = _search_snap.set(snap)
        try:
            bm25 = self._bm25_for(snap)
            bm25_task = asyncio.get_running_loop().run_in_executor(
                None, lambda: bm25.search_weighted(weighted, settings.top_k_recall)
                if bm25 else []
            )
            vec_task = asyncio.ensure_future(self._vector_search(query, settings.top_k_recall))
            bm25_hits, vec_hits = await asyncio.gather(bm25_task, vec_task)

            fused = self._rrf_fuse(
                snap.chunks, query, bm25_hits, vec_hits, city, domain,
                status_filter, as_of,
            )
        finally:
            _search_snap.reset(token)
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
                    # rerank 分替代融合分：注入信任降权必须重施，否则重排序
                    # 会洗掉融合层的惩罚、毒块重回 top-k（1B3-PC 挤占修复）
                    sc.final_score = score * (
                        settings.injection_trust_penalty if sc.injection_flagged else 1.0
                    )
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
        # 快照（PC-2）：优先取 search 入口经 ContextVar 传入的版本；
        # 直接调用（旧测试/脚本路径）回退当前快照——仍是同版本自洽对象
        snap = _search_snap.get() or self.store.snapshot
        chunks, matrix = snap.chunks, snap.matrix
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
        chunks: tuple[IndexedChunk, ...],
        query: str,
        bm25_hits: list,
        vec_hits: list[tuple[int, float]],
        city: str | None,
        domain: str | None,
        status_filter: str | None = None,
        as_of_date: date | None = None,
    ) -> list[ScoredChunk]:
        k = settings.rrf_k
        fused: dict[int, ScoredChunk] = {}

        # 时效过滤（1B5）：在融合入口执行——被过滤的块不进候选（而非打 0 分
        # 沉底，0 分块仍会占住 fused 字典与 rerank 候选名额）。
        # status_filter 按 is_valid 语义判定（兼容"现行有效（2018修正）"等变体）；
        # as_of 只保留 effective_date<=as_of 的块，无日期/日期非法的块排除
        # （无法证明其在 as_of 时点已生效，时点问答宁缺勿错）。
        want_valid: bool | None = None
        if status_filter is not None:
            if "废止" in status_filter:
                want_valid = False
            elif "现行" in status_filter or "有效" in status_filter:
                want_valid = True
        as_of_ord = as_of_date.toordinal() if as_of_date is not None else None

        def ensure(idx: int) -> ScoredChunk | None:
            chunk = chunks[idx]
            if want_valid is not None and chunk.is_valid != want_valid:
                return None
            if as_of_ord is not None:
                eff = parse_effective_date(chunk.effective_date)
                if eff is None or eff.toordinal() > as_of_ord:
                    return None
            if idx not in fused:
                fused[idx] = ScoredChunk(chunk=chunk, fused_score=0.0)
            return fused[idx]

        for rank, hit in enumerate(bm25_hits, start=1):
            sc = ensure(hit.index)
            if sc is None:
                continue
            sc.bm25_rank = rank
            sc.fused_score += 1.0 / (k + rank)
        for rank, (idx, _) in enumerate(vec_hits, start=1):
            sc = ensure(idx)
            if sc is None:
                continue
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
        from ..agent.injection import scan_lines  # 局部导入：模式库为 agent/retrieval 共用安全组件

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
            # 注入信任降权（1B3-PC，PC-HR2 挂账修复）：命中注入模式库的块在检索
            # 融合层即沉底——防关键词堆砌毒块挤占 top-k（可用性 DoS，
            # evidence/prompt_injection_report.md §4）。零漂移论证：模式库对正式
            # 语料零命中（tests/test_injection.py 守门），乘数对正式块恒不生效，
            # 冻结口径检索结果不受扰动；此后隔离文档仍走 grade=0 双保险。
            sc.injection_flagged = bool(scan_lines(sc.chunk.text))
            if sc.injection_flagged:
                score *= settings.injection_trust_penalty
                stats.incr("injection_retrieval_downweighted")
            if not sc.chunk.is_valid:
                score *= settings.invalid_doc_penalty
            sc.final_score = score if sc.rerank_score is None else sc.final_score

        out = sorted(fused.values(), key=lambda sc: sc.final_score, reverse=True)
        return _newer_version_first_on_ties(out)


def _same_policy_series(a: IndexedChunk, b: IndexedChunk) -> bool:
    """同一政策系列的判定：同文档 / 同发文字号 / 同标题（"同主题"的保守代理）。"""
    if a.doc_id == b.doc_id:
        return True
    if a.doc_no and a.doc_no == b.doc_no:
        return True
    return bool(a.title) and a.title == b.title


def _newer_version_first_on_ties(items: list[ScoredChunk]) -> list[ScoredChunk]:
    """date_rank 消费者（1B5 政策时效性引擎）——终结 metadata.date_rank 自上线
    以来零消费者的死代码状态。

    业务语义：同一政策的新旧版本并存时（旧版靠 invalid_doc_penalty×0.3 自然
    沉底，不会顶在现行文件之前），两版融合分**恰好并列**时排序此前取决于字典
    插入序（不稳定）。这里对「final_score 相等且属同一政策系列」的相邻组按
    date_rank（生效日期序数）降序**稳定**重排，使新版本确定性地排在旧版之前。

    作用域刻意收窄：仅重排 同 doc_id（同文档多块——date_rank 相同 → 稳定排序
    保持原序）/ 同非空 doc_no（修订版） / 同标题（同名事项新版）三类组合。
    异源并列（不同文件、不同文号、不同标题，如 rerank 打分并列）一律不重排，
    冻结评测口径（87.9%/92.2%）的检索序不受扰动。
    """
    if len(items) < 2:
        return items
    out: list[ScoredChunk] = []
    i = 0
    n = len(items)
    while i < n:
        j = i + 1
        while (
            j < n
            and items[j].final_score == items[i].final_score
            and _same_policy_series(items[i].chunk, items[j].chunk)
        ):
            j += 1
        group = items[i:j]
        if len(group) > 1:
            group = sorted(group, key=lambda sc: -date_rank(sc.chunk.effective_date))
        out.extend(group)
        i = j
    return out
