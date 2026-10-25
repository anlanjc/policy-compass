"""政策罗盘 PolicyCompass：跨城市社保/公积金政策的 Agentic RAG 问答服务。

在 Datawhale《All in RAG》教程"四步 RAG"之上的业务化演进：
- 意图路由（城市/领域/是否需要计算）→ 元数据过滤检索
- 手写混合检索（BM25 + 向量 + RRF 融合）+ rerank
- CRAG 式自纠错：文档评分 → 黑话改写重检索 → 生成 → 忠实度校验 → 诚实拒答
- Function Calling 计算工具：LLM 提参数，公式硬编码计算
- FastAPI SSE 流式接口 + 全链路 trace
- 5 类 30 题评测集与一键对比报告

语料为按真实政策结构编写的演示数据（非真实政策条文），见 data/corpus。
"""

__version__ = "0.1.0"
