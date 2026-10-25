"""LangGraph 全局状态定义。

events 使用 Annotated[list, operator.add] reducer：各节点 append 的 trace 事件
自动合并，API 层可随图执行逐步推送给前端（SSE），实现"看得见的 agent 思考过程"。
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class AgentState(TypedDict, total=False):
    # 输入
    thread_id: str
    messages: list[dict]  # 多轮历史（已按 history_window 截断）
    question: str

    # 意图路由
    route: dict[str, Any]  # {city, domain, need_calc, sub_questions}

    # 检索（含改写循环）
    query: str  # 当前检索词（可能已被改写）
    retrieval_round: int
    expanded_terms: list[str]
    documents: list[dict[str, Any]]  # [{chunk_id, text, title, city, ..., grade}]

    # 工具调用（Function Calling）
    tool_result: dict[str, Any] | None

    # 生成与校验
    answer: str
    generation_attempts: int
    verified: bool
    refusal: bool
    sources: list[dict[str, Any]]

    # 轨迹
    events: Annotated[list[dict[str, Any]], operator.add]
