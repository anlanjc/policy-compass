"""API 路由：SSE 流式问答、文档管理、健康与统计。

流式实现（token 级）与 LangGraph 的关系——两套入口共用同一组节点函数：
- POST /api/chat（stream=false）：完整执行 LangGraph 图（含 verify 重试循环），
  适合评测与结构化消费；
- POST /api/chat（stream=true）：手动驱动 route→retrieve→grade→(rewrite 循环)
  →tool 的同一批节点函数，随后 llm.chat_stream 逐 token 推送、流毕执行 verify
  （不通过则推送 correction 事件替换答案）。图内生成与流式生成共用
  build_generation_messages，行为一致。
  这是"框架服务需求"的取舍：LangGraph 原生流式只到节点粒度，token 级流式
  需要在图外接管生成节点。
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import AsyncIterator

import secrets
from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse

from ..agent.state import AgentState
from ..config import settings
from ..llm.stats import stats
from .schemas import (
    ChatRequest,
    ChatResponse,
    DocumentUpload,
    DocumentUploadResponse,
    Source,
)

router = APIRouter(prefix="/api")


def get_app_state(request=None):
    """从 FastAPI app 取全局单例（store/retriever/agent/recorder）。"""
    from .main import app_state

    return app_state


# ---------- 问答 ----------


async def _run_stream(ctx, req: ChatRequest) -> AsyncIterator[str]:
    """token 级 SSE：事件用 type=trace 推送，答案用 type=token 推送。"""
    agent_nodes = ctx["agent"].nodes
    llm = ctx["llm"]
    start = time.perf_counter()

    def sse(payload: dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    yield sse({"type": "meta", "trace_id": req.thread_id or str(uuid.uuid4()),
               "llm_mode": settings.effective_llm_mode})

    state: AgentState = {
        "question": req.question,
        "messages": (req.history or []) + [{"role": "user", "content": req.question}],
        "events": [],
    }

    try:
        # --- 前置阶段：与图相同的节点函数 ---
        state.update(await agent_nodes["route"](state))
        for ev in state["events"]:
            yield sse({"type": "trace", "event": ev})

        state = dict(state)
        state["events"] = []
        state.update(await agent_nodes["retrieve"](state))
        for ev in state["events"]:
            yield sse({"type": "trace", "event": ev})

        graded = await agent_nodes["grade"](state)
        state.update(graded)
        for ev in graded.get("events", []):
            yield sse({"type": "trace", "event": ev})

        from ..agent.graph import _after_grade, _after_verify  # 复用条件边逻辑

        decision = _after_grade(state)
        rewrite_round = 0
        while decision == "rewrite" and rewrite_round < settings.max_query_rewrites:
            state["events"] = []
            state.update(await agent_nodes["rewrite"](state))
            for ev in state["events"]:
                yield sse({"type": "trace", "event": ev})
            state["events"] = []
            state.update(await agent_nodes["retrieve"](state))
            for ev in state["events"]:
                yield sse({"type": "trace", "event": ev})
            graded = await agent_nodes["grade"](state)
            state.update(graded)
            for ev in graded.get("events", []):
                yield sse({"type": "trace", "event": ev})
            decision = _after_grade(state)
            rewrite_round += 1

        if decision == "fallback" or (decision == "rewrite"):
            state.update(await agent_nodes["fallback"](state))
            yield sse({"type": "token", "text": state["answer"]})
            yield sse({"type": "done", "answer": state["answer"], "refusal": True,
                       "sources": [], "tool_result": None,
                       "latency_ms": round((time.perf_counter() - start) * 1000, 1)})
            return

        if decision == "tool":
            state["events"] = []
            state.update(await agent_nodes["tool"](state))
            for ev in state["events"]:
                yield sse({"type": "trace", "event": ev})

        # --- 生成：token 级流式（图外接管） ---
        msgs, docs = ctx["agent"].build_generation_messages(state)
        collected: list[str] = []
        async for delta in llm.chat_stream(msgs, temperature=0.2):
            collected.append(delta)
            yield sse({"type": "token", "text": delta})
        answer = "".join(collected)
        state["answer"] = answer
        state["sources"] = [
            {
                "no": i, "title": d["title"], "doc_no": d["doc_no"], "city": d["city"],
                "effective_date": d["effective_date"], "sections": d.get("sections", []),
                "pages": d.get("pages", []), "is_valid": d["is_valid"],
                "chunk_id": d["chunk_id"],
            }
            for i, d in enumerate(docs, start=1)
        ]

        # --- 流后忠实度校验 ---
        v_state = await agent_nodes["verify"](state)
        if not v_state.get("verified"):
            yield sse({"type": "trace", "event": {"event": "verify", "grounded": False,
                        "note": "流式回答未通过忠实度校验，触发重写"}})
            state["generation_attempts"] = 1
            state["answer"] = await llm.chat(msgs, temperature=0.1)
            yield sse({"type": "correction", "answer": state["answer"]})

        yield sse({
            "type": "done",
            "answer": state["answer"],
            "refusal": False,
            "sources": state["sources"],
            "tool_result": state.get("tool_result"),
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        })
    except Exception as e:  # 全局兜底：任何异常都以 error 事件结束，绝不悬挂连接
        yield sse({"type": "error", "detail": str(e)[:300]})


@router.post("/chat")
async def chat(req: ChatRequest):
    ctx = get_app_state()
    stats.incr("queries_total")
    if req.stream:
        return StreamingResponse(
            _run_stream(ctx, req),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    start = time.perf_counter()
    trace_id = req.thread_id or str(uuid.uuid4())
    final = await ctx["agent"].run(
        req.question, thread_id=trace_id, history=req.history
    )
    latency = round((time.perf_counter() - start) * 1000, 1)
    resp = ChatResponse(
        trace_id=trace_id,
        question=req.question,
        answer=final.get("answer", ""),
        refusal=bool(final.get("refusal")),
        sources=[Source(**s) for s in final.get("sources", [])],
        tool_result=final.get("tool_result"),
        events=final.get("events", []),
        latency_ms=latency,
    )
    ctx["recorder"].record(
        trace_id, req.question, resp.events, resp.answer, resp.refusal,
        [s.model_dump() for s in resp.sources], latency,
    )
    return resp


# ---------- 文档管理（知识库热更新） ----------


async def require_admin_key(x_admin_key: str | None = Header(default=None)) -> None:
    """知识库写接口鉴权（S3 评审 PC-3）：公网部署下匿名上传=可投毒假政策、
    匿名删除=可清空语料。默认拒绝：未配置 POLICY_COMPASS_ADMIN_KEY 时写接口
    整体禁用；配置后校验 X-Admin-Key（常数时间比较）。读接口不受影响。
    """
    key = settings.admin_key
    if not key:
        raise HTTPException(
            403,
            "知识库写接口已禁用：配置 POLICY_COMPASS_ADMIN_KEY 后方可调用"
            "（公网部署防投毒/清空，见 DEPLOY.md）",
        )
    if not x_admin_key or not secrets.compare_digest(x_admin_key, key):
        raise HTTPException(401, "X-Admin-Key 缺失或不正确")


@router.get("/documents")
async def list_documents():
    ctx = get_app_state()
    return {"documents": ctx["store"].list_docs()}


@router.post("/documents", response_model=DocumentUploadResponse,
             dependencies=[Depends(require_admin_key)])
async def upload_document(body: DocumentUpload):
    ctx = get_app_state()
    try:
        n = await ctx["store"].add_document(body.filename, body.content, ctx["llm"].embed)
    except ValueError as e:
        raise HTTPException(400, str(e))
    ctx["retriever"].refresh()
    return DocumentUploadResponse(doc_id=body.filename, chunks=n)


@router.delete("/documents/{doc_id}", dependencies=[Depends(require_admin_key)])
async def delete_document(doc_id: str):
    ctx = get_app_state()
    n = ctx["store"].remove_doc(doc_id)
    if n == 0 and not any(d["doc_id"] == doc_id for d in ctx["store"].list_docs()):
        raise HTTPException(404, f"文档不存在: {doc_id}")
    ctx["retriever"].refresh()
    return {"removed": doc_id, "chunks_deleted": n}


@router.post("/ingest/sync", dependencies=[Depends(require_admin_key)])
async def ingest_sync():
    """重新扫描 corpus 目录（增量）：政策文件更新后的一键热更新入口。"""
    ctx = get_app_state()
    report = await ctx["store"].sync(None, ctx["llm"].embed)
    ctx["retriever"].refresh()
    return report.as_dict()


# ---------- 运维 ----------


@router.get("/health")
async def health():
    ctx = get_app_state()
    return {
        "status": "ok",
        "llm_mode": settings.effective_llm_mode,
        "docs": ctx["store"].doc_count(),
        "chunks": len(ctx["store"].chunks),
    }


@router.get("/stats")
async def get_stats():
    return stats.snapshot()
