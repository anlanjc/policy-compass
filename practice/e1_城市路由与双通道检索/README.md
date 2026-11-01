# E1 · 白板重写：城市规则路由 + BM25 双通道检索

> 讲义：`docs/lectures/L1_混合检索双通道.md`（双通道/RRF/加权部分）；路由对应 `app/agent/router.py` ｜ 判卷改编自 `tests/test_retrieval.py` 与 `tests/test_tools_graph.py::test_agent_rule_router_fallback`

## 任务

蒙住 `app/retrieval/bm25.py` 与 `app/agent/router.py`，在 `skeleton.py` 里重写两块最可测的确定性逻辑（练习自包含，不 import app 检索/路由模块）：

1. **规则路由 `rule_based_route`**：城市命中（1 个→该城；多个/0 个→全国；0 个才查别名词表）、领域关键词按声明顺序、need_calc（计算词+数字）、对比型问题拆 sub_questions；
2. **双通道分词**：`cjk_bigrams`（纯 CJK 相邻字符对，标点剥离）、`tokenize`（词通道+每个 CJK 词追加字符 bigram）；
3. **手写 BM25 `SimpleBM25`**：`__init__` 建 tf/df/doc_len/avgdl、`idf = ln(1 + (N-n+0.5)/(n+0.5))`、`search_weighted` 按 `weight·idf·f·(k1+1) / (f + k1·(1-b+b·|d|/avgdl))` 打分。

给定不许改：词表常量（CITIES/CITY_ALIASES/DOMAIN_KEYWORDS/CALC_PAT）、`tokenize_words`（jieba）、`RouteResult.validated()`、`SimpleBM25.search` 语法糖。

## 提示

- 路由的坑在**城市判定顺序**：多城对比时不能再去查别名（想想"京沪"这类写法）；need_calc 必须"计算词**且**数字"两个条件同时成立。
- bigram 的意义是修复"杭州市住房公积金 vs 杭州公积金"的分词粒度失配——讲义 L1 第 1 节有完整因果链。bigram 是普通 token，不加特殊权重。
- BM25 公式默写三要素：idf 加 1 非负、k1 词频饱和、b 文档长度归一；查询词不在 df 里直接跳过。
- 测试采用真实 jieba 分词（与 `tests/test_retrieval.py` 同一运行环境），判卷与真实实现行为逐位一致。

## 验收标准（哪些测试跑绿）

`pytest test_practice.py` 全部 11 条：

- 路由 6 条（改编自 `test_tools_graph.py` 规则路由用例 + 别名/领域/计算/sub_questions 扩展）：多城→全国、单城→该城、零城→全国、"沪"→上海、domain+need_calc、sub_questions 有无；
- 双通道 2 条（改编自 `test_retrieval.py` 前两条）：bigram 通道对齐粒度、标点剥离；
- BM25 3 条（改编自 `test_retrieval.py` 两条 + 零命中扩展）：相关文档排第一、idf 非负、扩展词低权重不压主信号、零命中返回空列表。

## 验证记录（判卷有效性）

- 参考实现（`solutions/solution.py`）复制进 `skeleton.py` 后：**11 passed in 0.74s**（全绿）；
- 还原 NotImplementedError 版骨架后：**11 failed in 0.49s**（全红，证明测试真的在判卷）。

## 做完后

对照 `solutions/solution.py` 与 `app/retrieval/bm25.py`、`app/agent/router.py` diff，自问三题（讲义 L1 第 7 节有答案）：
1. RRF 里向量路为什么 0.8 权重？（进阶，不在本练习判卷范围）
2. bigram 为什么不加特殊权重？
3. 路由为什么"多城→全国"而不是报错或随机选一城？
然后在 `practice/README.md` 的完成记录表里写一句"我理解了什么"。
