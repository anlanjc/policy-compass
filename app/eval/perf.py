"""性能画像的分桶与分位数（PC-HR4，纯函数）。

分桶优先级（互斥归属，文档化）：计算类 > 黑话 > 单城 > 全国——
计算类走工具路径（结构不同必须单列）；黑话以词典扩展触发为判据
（retrieve 事件 expanded_terms 非空）；其余按路由城市二分。
"""

from __future__ import annotations

import math

BUCKETS = ("calc", "slang", "single_city", "national")
BUCKET_LABELS = {
    "calc": "计算类（工具路径）",
    "slang": "黑话（词典扩展触发）",
    "single_city": "单城",
    "national": "全国",
}


def classify_bucket(route: dict, case_type: str, expanded_terms: list[str] | None) -> str:
    if case_type == "calc" or route.get("need_calc"):
        return "calc"
    if case_type == "slang" or expanded_terms:
        return "slang"
    city = route.get("city")
    if city and city != "全国":
        return "single_city"
    return "national"


def percentile(values: list[float], q: float) -> float:
    """最近秩分位数（nearest-rank：ceil(q·n)，q∈[0,1]）；空列表返回 0.0。"""
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))
    return s[idx]
