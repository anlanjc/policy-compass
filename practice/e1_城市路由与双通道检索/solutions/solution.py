"""E1 参考实现 —— 与 app/agent/router.py::rule_based_route、app/retrieval/bm25.py 逐行为对应（先自己写，跑不过再看）。"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

CITIES = ["杭州", "上海", "北京", "深圳", "广州", "全国"]
CITY_ALIASES = {"沪": "上海", "京": "北京", "杭": "杭州", "穗": "广州", "深": "深圳"}
DOMAIN_KEYWORDS = {
    "公积金": ["公积金", "房贷", "贷款额度", "租房提取", "缴存"],
    "社保": ["社保", "五险", "养老", "断缴", "补缴", "落户", "居转户", "缴费基数"],
    "医保": ["医保", "医疗", "报销", "大病"],
}
CALC_PAT = ("余额", "月薪", "缴了", "缴存", "能贷", "贷多少", "额度", "多少", "几万", "月供")

K1 = 1.5
B = 0.75


@dataclass
class RouteResult:
    city: str = "全国"
    domain: str = "通用"
    need_calc: bool = False
    sub_questions: list[str] = field(default_factory=list)

    def validated(self) -> "RouteResult":
        if self.city not in CITIES:
            self.city = "全国"
        if self.domain not in ("公积金", "社保", "医保", "通用"):
            self.domain = "通用"
        return self


def _is_cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def tokenize_words(text: str) -> list[str]:
    import jieba

    return [t.lower() for t in jieba.lcut(text) if len(t.strip()) >= 2 or t.strip().isdigit()]


def rule_based_route(question: str) -> RouteResult:
    """确定性规则路由：LLM 不可用时的兜底，也是路由正确性的对照组。"""
    hits = [name for name in CITIES if name != "全国" and name in question]
    city = hits[0] if len(hits) == 1 else "全国"  # 多城对比/零城 → 全国（不限定）
    if not hits:
        for alias, full in CITY_ALIASES.items():
            if alias in question:
                city = full
                break
    domain = "通用"
    for dom, words in DOMAIN_KEYWORDS.items():
        if any(w in question for w in words):
            domain = dom
            break
    need_calc = any(w in question for w in CALC_PAT) and any(
        ch.isdigit() for ch in question
    )
    sub_questions: list[str] = []
    if any(w in question for w in ("和", "对比", "区别", "哪个")):
        sub_questions = [question]
    return RouteResult(city=city, domain=domain, need_calc=need_calc,
                       sub_questions=sub_questions).validated()


def cjk_bigrams(text: str) -> set[str]:
    """纯 CJK 字符 bigram 集合（跨词边界），用于标题匹配等模糊重叠计算。"""
    chars = [c for c in text if _is_cjk(c)]
    return {chars[i] + chars[i + 1] for i in range(len(chars) - 1)}


def tokenize(text: str) -> list[str]:
    """词 + CJK 字符 bigram 双通道分词（Lucene CJK 分析器思路）。"""
    words = tokenize_words(text)
    bigrams: list[str] = []
    for w in words:
        if any(_is_cjk(c) for c in w):
            bigrams.extend(w[i : i + 2] for i in range(len(w) - 1))
    return words + bigrams


class SimpleBM25:
    def __init__(self, corpus: list[str]) -> None:
        self.k1 = K1
        self.b = B
        self.doc_tf: list[Counter] = []
        self.doc_len: list[int] = []
        self.df: Counter = Counter()
        for text in corpus:
            toks = tokenize(text)
            tf = Counter(toks)
            self.doc_tf.append(tf)
            self.doc_len.append(len(toks))
            self.df.update(tf.keys())
        self.n_docs = len(corpus)
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0

    def idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """普通检索：所有查询词等权。"""
        return self.search_weighted({t: 1.0 for t in set(tokenize(query))}, top_k)

    def search_weighted(
        self, weighted_terms: dict[str, float], top_k: int = 20
    ) -> list[tuple[int, float]]:
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
        return ranked
