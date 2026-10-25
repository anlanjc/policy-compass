"""手写 BM25（Okapi BM25， Robertson-Sparck Jones）。

为什么不用 rank_bm25 / LangChain Retriever：BM25 是面试高频考点，逐行实现
才能讲清 idf 饱和、文档长度归一（b 参数）各自解决什么问题；同时暴露
查询词权重入口，支撑黑话扩展词的低权重注入。

公式：score(q,d) = Σ IDF(qi) · f(qi,d)·(k1+1) / (f(qi,d) + k1·(1-b+b·|d|/avgdl))
其中 IDF(qi) = ln(1 + (N - n(qi) + 0.5) / (n(qi) + 0.5))，加 1 保证非负。
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass

from ..config import settings

logger = logging.getLogger(__name__)


def _is_cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def tokenize_words(text: str) -> list[str]:
    """词级分词（不含 bigram 通道），用于标题匹配等需要"真词"的场景。"""
    import jieba

    return [
        t.lower()
        for t in jieba.lcut(text)
        if len(t.strip()) >= 2 or t.strip().isdigit()
    ]


def cjk_bigrams(text: str) -> set[str]:
    """纯 CJK 字符 bigram 集合（跨词边界），用于标题匹配等模糊重叠计算。

    词级匹配对粒度敏感（"杭州"≠"杭州市"，"公积金"≠"住房公积金"），
    字符 bigram 重叠对这些变体鲁棒。
    """
    chars = [c for c in text if _is_cjk(c)]
    return {chars[i] + chars[i + 1] for i in range(len(chars) - 1)}


def tokenize(text: str) -> list[str]:
    """词 + CJK 字符 bigram 双通道分词（Lucene CJK 分析器思路）。

    动机（实测踩坑）：jieba 把文档中"杭州市住房公积金"切成整词，而查询说
    "杭州公积金"——粒度不一致导致词面完全失配、BM25 零命中。对每个 CJK 词
    追加其字符 bigram（"杭州""州市"…），查询与文档同法处理，失配消除；
    bigram 作为普通 token 参与 tf/idf，无需特殊权重。
    """
    words = tokenize_words(text)
    bigrams: list[str] = []
    for w in words:
        if any(_is_cjk(c) for c in w):
            bigrams.extend(w[i : i + 2] for i in range(len(w) - 1))
    return words + bigrams


@dataclass
class BM25Result:
    index: int
    score: float


class BM25Index:
    def __init__(self, corpus: list[str]) -> None:
        self.k1 = settings.bm25_k1
        self.b = settings.bm25_b
        self.doc_tokens: list[list[str]] = []
        self.doc_tf: list[Counter] = []
        self.doc_len: list[int] = []
        self.df: Counter = Counter()
        for text in corpus:
            toks = tokenize(text)
            tf = Counter(toks)
            self.doc_tokens.append(toks)
            self.doc_tf.append(tf)
            self.doc_len.append(len(toks))
            self.df.update(tf.keys())
        self.n_docs = len(corpus)
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0

    def idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def search(self, query: str, top_k: int = 20) -> list[BM25Result]:
        """普通检索：所有查询词等权。"""
        return self.search_weighted({t: 1.0 for t in set(tokenize(query))}, top_k)

    def search_weighted(
        self, weighted_terms: dict[str, float], top_k: int = 20
    ) -> list[BM25Result]:
        """带词权检索：黑话扩展词以 0.6 权重注入，避免淹没原始查询意图。"""
        scores: dict[int, float] = {}
        for term, weight in weighted_terms.items():
            if term not in self.df:
                continue
            idf = self.idf(term)
            for idx in range(self.n_docs):
                f = self.doc_tf[idx].get(term, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (
                    1 - self.b + self.b * self.doc_len[idx] / (self.avgdl or 1)
                )
                scores[idx] = scores.get(idx, 0.0) + weight * idf * f * (self.k1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [BM25Result(index=i, score=s) for i, s in ranked]
