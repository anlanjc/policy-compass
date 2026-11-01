# PolicyCompass 白板重写练习场（practice/）

> **先自己写，跑不过再看 `solutions/`。** 参考实现是判卷后的对照答案，不是起点。

## 练习清单

| 练习 | 对应讲义 | 重写对象 | 判卷测试来源 |
|---|---|---|---|
| `e1_城市路由与双通道检索/` | `docs/lectures/L1_混合检索双通道.md` | `app/agent/router.py::rule_based_route` + `app/retrieval/bm25.py` 的 `cjk_bigrams`/`tokenize`/`BM25Index` | `tests/test_retrieval.py`、`tests/test_tools_graph.py` 核心用例改编 |

## 做题流程

1. **读讲义**：合上讲义能复述"零命中踩坑因果链 + BM25 公式三要素"再动笔。
2. **看签名**：`skeleton.py` 的 docstring 就是行为规格，逐条满足。
3. **自己写**：不打开 `app/` 对着抄；卡住回讲义。
4. **跑测试**：在本目录 `python -m pytest test_practice.py -v`（系统 Python 已装 jieba/pytest；本项目无 .venv，用系统 python 直接跑即可）。
5. **对照 solution**：全绿后 diff `solutions/solution.py`，找自己漏掉的分支与"为什么"。
6. **写一句"我理解了什么"**：填进下面完成记录表。

## 验证记录（判卷有效性）

| 练习 | solution 放入 skeleton | 还原 NotImplementedError 骨架 | 结论 |
|---|---|---|---|
| e1_城市路由与双通道检索 | 11 passed in 0.74s | 11 failed in 0.49s | 测试真的能判卷 |

## 环境说明

- 本项目无独立 .venv，用系统 Python（已验证 jieba 0.42.1 / pytest 9.1.1 可用，原 `tests/test_retrieval.py` 8 passed 基线通过）。
- `practice/conftest.py` 负责把项目根加入 `sys.path` 并强制 mock 模式（本练习不调 LLM，纯确定性函数）。
- `test_practice.py` 的 import 全部指向 `skeleton.py`——你写的代码就是被测对象。

## 完成记录

| 日期 | 练习 | 一句"我理解了什么" |
|---|---|---|
| | | |
