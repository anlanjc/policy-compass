"""性能画像分桶与分位数回归（PC-HR4）。"""

from __future__ import annotations

from app.eval.perf import classify_bucket, percentile


def test_classify_bucket_precedence():
    # 计算类最高优先级（工具路径，即便命中黑话/城市也单列）
    assert classify_bucket({"city": "杭州", "need_calc": True}, "fact", ["租赁提取"]) == "calc"
    assert classify_bucket({"city": "全国", "need_calc": False}, "calc", []) == "calc"
    # 黑话次之（词典扩展触发判据）
    assert classify_bucket({"city": "杭州", "need_calc": False}, "fact", ["租赁提取"]) == "slang"
    assert classify_bucket({"city": "全国", "need_calc": False}, "slang", []) == "slang"
    # 单城/全国按路由
    assert classify_bucket({"city": "杭州", "need_calc": False}, "fact", []) == "single_city"
    assert classify_bucket({"city": "全国", "need_calc": False}, "refuse", []) == "national"
    assert classify_bucket({}, "fact", []) == "national"


def test_percentile_nearest_rank():
    assert percentile([], 0.5) == 0.0
    assert percentile([7.0], 0.95) == 7.0
    vals = [float(i) for i in range(1, 101)]  # 1..100
    assert percentile(vals, 0.5) == 50.0
    assert percentile(vals, 0.95) == 95.0
    assert percentile(vals, 0.0) == 1.0
    assert percentile(vals, 1.0) == 100.0
