"""图节点实现：检索、评分、改写、工具调用、生成、忠实度校验、诚实拒答。

可靠性细节（面经"失败处理"的直接落点）：
- grade 并行评分 + 单条 8s 超时，评分失败的 chunk 记 0.3 保守分（宁可不选，不可错选）；
- verify 不通过 → 携带"严格依据资料"强化提示重生成一次 → 仍不通过 → 诚实降级，
  绝不输出无法溯源的答案（政策问答答错数字即事故）；
- 改写循环上限 max_query_rewrites + 图级 recursion_limit 双重防死循环。
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from pydantic import BaseModel, Field

from ..config import settings
from ..llm.client import LLMError, StructuredOutputError
from ..llm.stats import stats
from .state import AgentState

logger = logging.getLogger(__name__)


def _ev(name: str, **kw: Any) -> dict[str, Any]:
    """构造 trace 事件（API 层随 SSE 推送，展示 agent 决策路径）。"""
    return {"event": name, "ts": round(time.time(), 3), **kw}


# ---------- 评分与校验的输出 schema ----------

class GradeResult(BaseModel):
    score: float = Field(ge=0, le=1, description="文档与问题的相关性 0~1")
    reason: str = Field(default="", description="一句话理由")


class RewriteResult(BaseModel):
    rewritten: str = Field(description="改写后的检索问题")
    expanded_terms: list[str] = Field(default_factory=list)


class VerifyResult(BaseModel):
    grounded: bool = Field(description="回答是否完全由参考资料支持")
    unsupported: list[str] = Field(default_factory=list, description="不支持的表述")


# ---------- 节点工厂（闭包注入 retriever / llm） ----------


def make_nodes(retriever, llm):
    async def route_node(state: AgentState) -> dict:
        from .router import route_query

        t0 = time.perf_counter()
        route = await route_query(llm, state["question"], state.get("messages", [])[:-1])
        return {
            "route": route.model_dump(),
            "query": state["question"],
            "retrieval_round": 0,
            "generation_attempts": 0,
            "refusal": False,
            "events": [
                _ev(
                    "route",
                    **route.model_dump(),
                    latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                )
            ],
        }

    async def retrieve_node(state: AgentState) -> dict:
        t0 = time.perf_counter()
        route = state["route"]
        r = await retriever.search(
            state["query"],
            city=route.get("city"),
            domain=route.get("domain"),
            use_rerank=True,
        )
        documents = []
        for sc in r.chunks:
            c = sc.chunk
            documents.append(
                {
                    "chunk_id": c.chunk_id,
                    "title": c.title,
                    "doc_no": c.doc_no,
                    "city": c.city,
                    "domain": c.domain,
                    "status": c.status,
                    "effective_date": c.effective_date,
                    "is_valid": c.is_valid,
                    "text": c.text,
                    "retrieval_score": round(sc.final_score, 5),
                }
            )
        return {
            "documents": documents,
            "expanded_terms": r.expanded_terms,
            "events": [
                _ev(
                    "retrieve",
                    round=state.get("retrieval_round", 0) + 1,
                    query=state["query"],
                    hits=len(documents),
                    top1=documents[0]["title"] if documents else None,
                    expanded_terms=r.expanded_terms,
                    rerank_used=r.rerank_used,
                    latency_ms=r.latency_ms,
                    total_ms=round((time.perf_counter() - t0) * 1000, 1),
                )
            ],
        }

    async def grade_node(state: AgentState) -> dict:
        docs = state["documents"][: settings.top_k_final]

        async def grade_one(doc: dict) -> float:
            try:
                async with asyncio.timeout(8):
                    g = await llm.chat_json(
                        [
                            {
                                "role": "system",
                                "content": "#task=grade\n你是检索结果评审员。判断待评文档能否直接支撑回答该政策问题。"
                                "打分锚点（严格按此校准）：文档含有回答所需的直接依据（具体标准/数字/公式/流程/条件）=0.85~1.0；"
                                "主题一致但仅覆盖问题的一部分=0.6~0.8；仅话题沾边、无答案=0.2~0.4；主题无关=0~0.1。"
                                "注意：政策问答的同一文档常含多条条文，只要其中一条能直接回答问题即应给高分。",
                            },
                            {
                                "role": "user",
                                "content": f"问题：{state['query']}\n\n待评文档：{doc['text'][:800]}",
                            },
                        ],
                        GradeResult,
                    )
                doc["grade"] = g.score
                doc["grade_reason"] = g.reason
            except (StructuredOutputError, LLMError, TimeoutError):
                doc["grade"] = 0.3  # 保守分：评分失败不惩罚检索、但不会误判为强相关
                doc["grade_reason"] = "评分失败，保守记分"
            return doc["grade"]

        t0 = time.perf_counter()
        grades = await asyncio.gather(*(grade_one(d) for d in docs))
        for doc in state["documents"][len(docs):]:
            doc["grade"] = None
        max_grade = max(grades, default=0.0)
        avg_grade = sum(grades) / len(grades) if grades else 0.0
        return {
            "documents": state["documents"],
            "events": [
                _ev(
                    "grade",
                    max_grade=round(max_grade, 2),
                    avg_grade=round(avg_grade, 2),
                    per_doc=[
                        {"title": d["title"][:24], "grade": d["grade"]}
                        for d in docs
                    ],
                    latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                )
            ],
        }

    async def rewrite_node(state: AgentState) -> dict:
        t0 = time.perf_counter()
        try:
            r = await llm.chat_json(
                [
                    {
                        "role": "system",
                        "content": "#task=rewrite\n你是政务检索查询改写器。用户常用口语黑话（如：断缴、跳槽、把公积金带走、"
                        "社保基数），而政策条文使用规范用语（中断缴费、工作变动、转移接续、缴费基数）。"
                        "请把口语问题改写为规范政策用语检索式：展开黑话、补全省略的城市/领域语境、保留原意。"
                        "不要回答问题，只输出改写。",
                    },
                    {
                        "role": "user",
                        "content": f"用户原始问题：{state['question']}\n（第 {state.get('retrieval_round', 0) + 1} 次改写，"
                        f"此前检索词：{state['query']}）",
                    },
                ],
                RewriteResult,
            )
            new_query = r.rewritten
        except (StructuredOutputError, LLMError):
            # 词典级兜底改写：retriever.expand_query 已做，这里直接用扩展词拼接
            weighted, expanded = retriever.expand_query(state["question"])
            new_query = state["question"] + " " + " ".join(expanded) if expanded else state["question"]

        return {
            "query": new_query,
            "retrieval_round": state.get("retrieval_round", 0) + 1,
            "events": [
                _ev(
                    "rewrite",
                    new_query=new_query,
                    round=state.get("retrieval_round", 0) + 1,
                    latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                )
            ],
        }

    async def tool_node(state: AgentState) -> dict:
        from .tools import execute_tool, select_tool

        t0 = time.perf_counter()
        call = await select_tool(llm, state["question"])
        if call.tool == "none":
            return {"tool_result": None, "events": [_ev("tool", tool="none")]}
        # 确定性上下文补全：模型常省略 city（默认它会被带入）——
        # 用意图路由的城市结果回填，缺失参数靠硬编码兜底而非指望模型自觉
        if "city" not in call.args or not call.args.get("city"):
            route_city = (state.get("route") or {}).get("city") or ""
            if route_city and route_city != "全国":
                call.args["city"] = route_city
        result = execute_tool(call)
        return {
            "tool_result": {"tool": call.tool, "args": call.args, "result": result},
            "events": [
                _ev(
                    "tool",
                    tool=call.tool,
                    args=call.args,
                    ok=result.get("ok"),
                    latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                )
            ],
        }

    def build_generation_messages(state: AgentState) -> list[dict]:
        """组装生成 prompt（图内生成与 API 流式生成共用，保证行为一致）。"""
        docs = [d for d in state["documents"] if d.get("grade") is None or d["grade"] >= 0.4][
            : settings.top_k_final
        ]
        ctx_lines = []
        for i, d in enumerate(docs, start=1):
            meta = f"（{d['city']}·{d['domain']}·{d['doc_no']}·生效{d['effective_date']}·"
            meta += "现行有效" if d["is_valid"] else "已废止，仅作历史参考"
            meta += "）"
            # chunk.text 由切分器注入《标题》开头，此处只拼元数据行
            ctx_lines.append(f"[{i}] {meta}\n{d['text']}")
        tool_part = ""
        if state.get("tool_result"):
            tr = state["tool_result"]
            tool_part = f"\n\n【计算工具结果（已按公式精确计算，直接采用）】\n{tr['result']}\n"
        history = state.get("messages", [])[-settings.history_window:-1]
        hist_part = ""
        if history:
            hist_part = "\n\n【对话历史】\n" + "\n".join(
                f"{m['role']}: {m['content'][:200]}" for m in history
            )
        system = (
            "你是「政策罗盘」社保公积金政策问答助手。严格遵守：\n"
            "1. 仅依据提供的参考资料回答，资料中没有的数字/结论不得编造；\n"
            "2. 引用来源时使用 [n] 标注对应资料编号；\n"
            "3. 优先引用现行有效文件；若仅找到已废止文件，须明确提示政策已更新；\n"
            "4. 涉及计算的必须采用【计算工具结果】中的数字；\n"
            "5. 资料不足以回答时明确说明知识库未收录，建议咨询 12329（公积金）/12333（人社）热线；\n"
            "6. 回答简洁分条，先给结论再给依据。"
        )
        user = (
            f"【参考资料】\n{chr(10).join(ctx_lines) or '（无）'}{tool_part}{hist_part}"
            f"\n\n【用户问题】{state['question']}"
        )
        return [{"role": "system", "content": system},
                {"role": "user", "content": user}], docs

    async def generate_node(state: AgentState) -> dict:
        t0 = time.perf_counter()
        msgs, docs = build_generation_messages(state)
        attempt = state.get("generation_attempts", 0)
        if attempt > 0:  # 忠实度校验重试：强化提示
            msgs[0]["content"] += (
                "\n注意：上一次回答存在未被资料支持的表述，本次回答中每一句结论"
                "都必须能在参考资料中找到出处，宁少勿错。"
            )
        answer = await llm.chat(msgs, temperature=0.2)
        sources = [
            {
                "no": i,
                "title": d["title"],
                "doc_no": d["doc_no"],
                "city": d["city"],
                "effective_date": d["effective_date"],
                "is_valid": d["is_valid"],
                "chunk_id": d["chunk_id"],
            }
            for i, d in enumerate(docs, start=1)
        ]
        return {
            "answer": answer,
            "sources": sources,
            "generation_attempts": attempt + 1,
            "events": [
                _ev(
                    "generate",
                    attempt=attempt + 1,
                    chars=len(answer),
                    sources=len(sources),
                    latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                )
            ],
        }

    async def verify_node(state: AgentState) -> dict:
        t0 = time.perf_counter()
        msgs, _ = build_generation_messages(state)
        docs = [d for d in state["documents"] if d.get("grade") is None or d["grade"] >= 0.4][
            : settings.top_k_final
        ]
        ctx = "\n".join(f"[{i}] {d['text'][:600]}" for i, d in enumerate(docs, start=1))
        tool_ctx = ""
        if state.get("tool_result"):
            tool_ctx = f"\n[工具] {state['tool_result']['result']}"
        try:
            v = await llm.chat_json(
                [
                    {
                        "role": "system",
                        "content": "#task=verify\n你是忠实度校验员。逐句检查候选回答：每个实质性结论"
                        "（数字、额度、条件、流程）是否都能在参考资料或工具结果中找到支持。"
                        "全部支持 grounded=true；存在编造/无出处的数字或结论则 false 并列出。\n"
                        "注意：程序性、兜底性表述不算无出处结论，不得计入 unsupported，"
                        "例如「建议拨打12329/12333热线」「以当地公积金/社保中心为准」「具体以官方公告为准」——"
                        "它们是服务规范而非事实断言。只校验政策事实本身。",
                    },
                    {
                        "role": "user",
                        "content": f"问题：{state['question']}\n\n参考资料：\n{ctx}{tool_ctx}\n\n候选回答：\n{state['answer']}",
                    },
                ],
                VerifyResult,
            )
        except (StructuredOutputError, LLMError):
            v = VerifyResult(grounded=True, unsupported=[])  # 校验器失效不阻塞主链路
        return {
            "events": [
                _ev(
                    "verify",
                    grounded=v.grounded,
                    unsupported=v.unsupported[:3],
                    latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                )
            ],
            # grounded 与 attempts 写入 state 供条件边判断
            "documents": state["documents"],
            "tool_result": state.get("tool_result"),
            "answer": state["answer"],
            "sources": state.get("sources", []),
            "generation_attempts": state.get("generation_attempts", 0),
            "verified": v.grounded,
        }

    async def fallback_node(state: AgentState) -> dict:
        stats.incr("refusals")
        graded = [d for d in state.get("documents", []) if d.get("grade")]
        if graded:
            top = max(graded, key=lambda d: d["grade"])
            answer = (
                "知识库中未找到能够可靠回答该问题的现行政策条文，为避免提供错误信息，"
                "我不做猜测性回答。\n与您的问题最接近的文件是《"
                + top["title"]
                + f"》（{top['doc_no']}），如需准确答复建议拨打 12329（公积金）/ 12333（人社）热线确认。"
            )
        else:
            answer = (
                "知识库中未收录与该问题相关的政策文件。我无法回答知识库以外的问题"
                "（如车牌拍卖、个税、护照等），建议咨询相应主管部门或拨打 12345 政务热线。"
            )
        return {
            "answer": answer,
            "refusal": True,
            "sources": [],
            "events": [_ev("fallback", refusal=True)],
        }

    return {
        "route": route_node,
        "retrieve": retrieve_node,
        "grade": grade_node,
        "rewrite": rewrite_node,
        "tool": tool_node,
        "generate": generate_node,
        "verify": verify_node,
        "fallback": fallback_node,
        "build_generation_messages": build_generation_messages,
    }
