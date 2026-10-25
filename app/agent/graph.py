"""LangGraph CRAG 编排：把自纠错检索（Corrective RAG）落成可运行的状态机。

图结构：

            ┌────────────────────────────┐
            ↓                            │
START → route → retrieve → grade ──不相关──→ rewrite（≤max 次）→ retrieve
                      │  相关                        │ 改写耗尽
                      ↓                              ↓
                need_calc? ──是──→ tool           fallback（诚实拒答）
                      │否                            ↑
                      ↓                              │ verify 两次不过
                  generate → verify ──通过──→ END   │
                      ↑        └────不通过（重生成≤max）┘
                      └────────────（重试时强化提示）

条件边决策全部依据量化信号（grade 分数、attempt 计数），不是模型自由发挥；
循环边界由 max_query_rewrites 与 recursion_limit 双重保证。
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from ..config import settings
from .nodes import make_nodes
from .state import AgentState

logger = logging.getLogger(__name__)


def _after_grade(state: AgentState) -> Literal["tool", "generate", "rewrite", "fallback"]:
    docs = [d for d in state.get("documents", []) if d.get("grade") is not None]
    max_grade = max((d["grade"] for d in docs), default=0.0)
    if max_grade >= settings.grade_threshold:
        if state["route"].get("need_calc"):
            return "tool"
        return "generate"
    if state.get("retrieval_round", 0) < settings.max_query_rewrites:
        return "rewrite"
    return "fallback"


def _after_verify(state: AgentState) -> Literal["generate", "fallback", "end"]:
    if state.get("verified"):
        return "end"
    if state.get("generation_attempts", 0) <= settings.max_generate_retries:
        return "generate"  # 携带强化提示重生成
    return "fallback"


class PolicyAgent:
    """封装图构建与调用（retriever / llm 通过构造注入，便于测试替换）。"""

    def __init__(self, retriever, llm) -> None:
        self.llm = llm
        self.nodes = make_nodes(retriever, llm)
        builder = StateGraph(AgentState)
        node_names = {"route": "route_intent", "retrieve": "retrieve", "grade": "grade",
                      "rewrite": "rewrite", "tool": "tool", "generate": "generate",
                      "verify": "verify", "fallback": "fallback"}
        for key, name in node_names.items():
            builder.add_node(name, self.nodes[key])
        builder.add_edge(START, "route_intent")
        builder.add_edge("route_intent", "retrieve")
        builder.add_edge("retrieve", "grade")
        builder.add_conditional_edges(
            "grade",
            _after_grade,
            {"tool": "tool", "generate": "generate", "rewrite": "rewrite", "fallback": "fallback"},
        )
        builder.add_edge("rewrite", "retrieve")
        builder.add_edge("tool", "generate")
        builder.add_edge("generate", "verify")
        builder.add_conditional_edges(
            "verify",
            _after_verify,
            {"generate": "generate", "fallback": "fallback", "end": END},
        )
        builder.add_edge("fallback", END)
        self.graph = builder.compile(checkpointer=MemorySaver())

    # ---------- 调用入口 ----------

    async def run(self, question: str, thread_id: str | None = None, history: list[dict] | None = None) -> AgentState:
        """完整执行图（非流式），返回最终 state（answer/sources/events）。

        thread_id：多轮对话共享同一 id 以复用 checkpointer 记忆；单次调用
        省略时自动生成，避免跨问题 state 泄漏。
        """
        import uuid

        if thread_id is None:
            thread_id = str(uuid.uuid4())
        messages = [{"role": "user", "content": question}]
        if history:
            messages = history + messages
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": settings.graph_recursion_limit,
        }
        final = await self.graph.ainvoke(
            {"question": question, "messages": messages, "events": []}, config
        )
        return final

    def build_generation_messages(self, state: AgentState) -> tuple[list[dict], list[dict]]:
        """供 API 流式模式复用：图内生成与流式生成使用同一 prompt 组装。"""
        return self.nodes["build_generation_messages"](state)
