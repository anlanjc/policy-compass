"""E1 骨架 —— 城市路由与双通道检索（对应 app/agent/router.py 与 app/retrieval/bm25.py）。

做题流程：读 README 与讲义 → 只按签名和 docstring 规格自己实现 → 跑
`python -m pytest test_practice.py -q` → 全绿后对照 solutions/solution.py。
禁止先把 solution 抄进来（那只是自欺）。

提示：
- 先实现 rule_based_route（确定性路由），再做分词双通道，最后 SimpleBM25。
- token 顺序：词通道在前、bigram 通道在后（测试对顺序敏感的部分已用集合断言）。
"""

from __future__ import annotations

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
        # TODO: 非法城市/领域回退默认值（防 LLM 输出越界）
        raise NotImplementedError


def _is_cjk(ch: str) -> bool:
    raise NotImplementedError


def tokenize_words(text: str) -> list[str]:
    """jieba 词通道：长度 >=2 的词 + 数字串，全部小写。"""
    raise NotImplementedError


def rule_based_route(question: str) -> RouteResult:
    """确定性规则路由。

    规格提示：
    - 命中恰好 1 个城市才限定；命中 0 个或多城对比 → 全国（不限定）
    - 无城市命中时查别名（沪/京/杭/穗/深）
    - 领域按 DOMAIN_KEYWORDS 顺序取第一个命中
    - need_calc：出现计算类关键词 且 问题里含数字
    - 含"和/对比/区别/哪个" → 拆为子问题（保持原句）
    """
    raise NotImplementedError


def cjk_bigrams(text: str) -> set[str]:
    """纯 CJK 字符 bigram 集合（跨词边界），用于标题模糊重叠。"""
    raise NotImplementedError


def tokenize(text: str) -> list[str]:
    """词 + CJK 字符 bigram 双通道分词（Lucene CJK 分析器思路）。

    bigram 只在含 CJK 的词内生成（跨词边界的由 cjk_bigrams 场景另行处理）。
    """
    raise NotImplementedError


class SimpleBM25:
    """Okapi BM25，支持带词权检索（黑话扩展词 0.6 权重注入）。"""

    def __init__(self, corpus: list[str]) -> None:
        # TODO: 预计算每篇的 token Counter、文档长度、df、avgdl
        raise NotImplementedError

    def idf(self, term: str) -> float:
        # TODO: Okapi IDF：log(1 + (N - n + 0.5) / (n + 0.5))
        raise NotImplementedError

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        # TODO: 复用 search_weighted，所有词等权 1.0
        raise NotImplementedError

    def search_weighted(
        self, weighted_terms: dict[str, float], top_k: int = 20
    ) -> list[tuple[int, float]]:
        # TODO: BM25 打分公式，权重乘进每一项；按分数降序取 top_k
        raise NotImplementedError
