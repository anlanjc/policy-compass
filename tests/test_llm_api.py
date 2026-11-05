"""LLM 客户端可靠性 + FastAPI 接口测试。"""

import json

import pytest


def test_extract_json_variants():
    from app.llm.client import extract_json

    assert extract_json('{"a": 1}') == '{"a": 1}'
    assert extract_json('说明```json\n{"a": 1}\n```完') == '{"a": 1}'
    assert extract_json('前置 {"a": {"b": "}"}} 后置') == '{"a": {"b": "}"}}'
    assert extract_json('没有 JSON') is None


async def test_chat_json_schema_repair(components):
    """结构化输出自校验：mock 返回合法 JSON 时一次通过并完成 pydantic 校验。"""
    from app.agent.nodes import GradeResult

    llm = components["llm"]
    g = await llm.chat_json(
        [
            {"role": "system", "content": "#task=grade"},
            {"role": "user", "content": "问题：杭州租房提取额度\n\n待评文档：《杭州市住房公积金租赁提取管理实施细则》第二条 每人每月最高可提取额度为1500元"},
        ],
        GradeResult,
    )
    assert 0 <= g.score <= 1


async def test_api_endpoints(temp_env, monkeypatch):
    import httpx

    from app.api.main import app
    from app.config import settings

    # S3 评审 PC-3：写接口须配管理密钥；本测试启用并携带 X-Admin-Key
    monkeypatch.setattr(settings, "admin_key", "test-admin-key")

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test", timeout=60,
            headers={"X-Admin-Key": "test-admin-key"},
        ) as c:
            health = (await c.get("/api/health")).json()
            assert health["status"] == "ok" and health["docs"] > 0

            chat = (await c.post("/api/chat", json={"question": "杭州公积金租房提取每月最多能提多少"})).json()
            assert chat["answer"] and not chat["refusal"]
            # 真实语料（杭房公委〔2023〕8号）的事实流入答案：额度数字或按实提取规则
            assert ("2000" in chat["answer"] or "实际房租" in chat["answer"]) and chat["sources"]
            assert chat["latency_ms"] >= 0

            # SSE 流式：meta → trace* → token* → done
            kinds = []
            async with c.stream(
                "POST", "/api/chat",
                json={"question": "我杭州公积金余额4万能贷多少", "stream": True},
            ) as resp:
                assert resp.headers["content-type"].startswith("text/event-stream")
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        obj = json.loads(line[6:])
                        kinds.append(obj["type"])
                        if obj["type"] == "done":
                            assert obj["answer"]
            assert kinds[0] == "meta" and "trace" in kinds and "token" in kinds
            assert kinds[-1] == "done"

            stats = (await c.get("/api/stats")).json()
            assert stats["queries_total"] >= 2

            docs = (await c.get("/api/documents")).json()["documents"]
            assert len(docs) >= 15

            # 知识库热更新：上传 → 可检索 → 删除
            demo = (
                "# 深圳市住房公积金租赁提取暂行办法（演示）\n"
                "【发文字号】深公积金〔2026〕1号\n【生效日期】2026-01-01\n"
                "【状态】现行有效\n【适用城市】深圳\n【领域】公积金\n\n"
                "第一条 深圳无房职工租赁提取每人每月最高2600元。\n"
            )
            up = (await c.post("/api/documents", json={"filename": "shenzhen_demo.md", "content": demo})).json()
            assert up["chunks"] >= 1
            # 查询锚定上传文档标题（暂行办法）：真实深圳语料（65%比例制）入库后，
            # 泛化的"每月能提多少"不再唯一命中演示文档，标题锚定保证测的是热更新链路
            chat2 = (await c.post("/api/chat", json={"question": "深圳住房公积金租赁提取暂行办法规定每人每月最高提取多少"})).json()
            assert "2600" in chat2["answer"]
            await c.delete("/api/documents/shenzhen_demo.md")
            gone = (await c.get("/api/documents")).json()["documents"]
            assert all(d["doc_id"] != "shenzhen_demo.md" for d in gone)


# ---------- S3 评审 PC-3 回归：知识库写接口鉴权 ----------

async def test_write_endpoints_auth(temp_env, monkeypatch):
    """默认禁用（403）→ 配 key 后无头 401 / 错头 401 / 对头 200；读接口不受影响。"""
    import httpx

    from app.api.main import app
    from app.config import settings

    demo = (
        "# 深圳租房提取测试办法（演示）\n【发文字号】测〔2026〕1号\n【状态】现行有效\n"
        "【适用城市】深圳\n【领域】公积金\n\n"
        "第一条 深圳无房职工租赁提取每人每月最高2600元，未办理租赁备案的按实际租金核算。\n"
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test", timeout=60
        ) as c:
            # 1) 未配置 key：写接口整体禁用
            monkeypatch.setattr(settings, "admin_key", "")
            assert (await c.post("/api/documents", json={"filename": "t.md", "content": demo})).status_code == 403
            assert (await c.delete("/api/documents/t.md")).status_code == 403
            assert (await c.post("/api/ingest/sync")).status_code == 403
            # 读接口不受影响
            assert (await c.get("/api/documents")).status_code == 200

            # 2) 配置 key：无头/错头 401
            monkeypatch.setattr(settings, "admin_key", "secret-key")
            assert (await c.post("/api/documents", json={"filename": "t.md", "content": demo})).status_code == 401
            assert (await c.delete("/api/documents/t.md", headers={"X-Admin-Key": "wrong"})).status_code == 401

            # 3) 正确 key：放行，且上传内容可检索、可删除
            h = {"X-Admin-Key": "secret-key"}
            up = await c.post("/api/documents", json={"filename": "t.md", "content": demo}, headers=h)
            assert up.status_code == 200 and up.json()["chunks"] >= 1
            chat = (await c.post("/api/chat", json={"question": "测试文档规定每月最高提取多少"})).json()
            assert "2600" in chat["answer"]
            assert (await c.delete("/api/documents/t.md", headers=h)).status_code == 200


# ---------- S3 评审 PC-8/PC-9 回归 ----------

async def test_grade_fallback_on_unlisted_exception(components, monkeypatch):
    """PC-8：未列名异常（如 RuntimeError）在评分环节记 0.3 保守分，不炸整请求。"""
    from app.agent.nodes import make_nodes

    llm = components["llm"]
    async def boom(*a, **k):
        raise RuntimeError("上游 200+HTML 页")
    monkeypatch.setattr(llm, "chat_json", boom)

    nodes = make_nodes(components["retriever"], llm)
    state = {"query": "杭州租房提取额度", "documents": [
        {"chunk_id": "c1", "title": "杭州市住房公积金租赁提取管理实施细则",
         "text": "《杭州市住房公积金租赁提取管理实施细则》第二条 每人每月最高可提取额度为2000元"},
    ]}
    out = await nodes["grade"](state)
    assert out["documents"][0]["grade"] == 0.3
    assert "RuntimeError" in out["documents"][0]["grade_reason"]


async def test_llm_non_json_body_raises_llmerror(temp_env, monkeypatch):
    """PC-8：200+非 JSON 体按可重试 LLMError 处理，JSONDecodeError 不得穿透。"""
    import httpx

    from app.config import settings
    from app.llm.client import LLMClient, LLMError

    monkeypatch.setattr(settings, "llm_max_retries", 0)
    c = LLMClient()
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(200, text="<html>代理错误页</html>")

    await c._http.aclose()
    c._http = httpx.AsyncClient(transport=httpx.MockTransport(handler),
                                base_url=settings.llm_api_base)
    try:
        try:
            await c.chat([{"role": "user", "content": "hi"}])
            raised = ""
        except LLMError as e:
            raised = str(e)
        assert "响应非 JSON" in raised
        assert calls["n"] == 1  # max_retries=0：单次尝试
    finally:
        await c._http.aclose()


async def test_stream_error_frame_sanitized(temp_env, monkeypatch):
    """PC-9：流式兜底 error 帧不外发异常明文（上游响应体/内部 URL）。"""
    import httpx

    from app.api import routes
    from app.api.main import app

    async with app.router.lifespan_context(app):
        ctx = routes.get_app_state()
        async def boom(state):
            raise RuntimeError("HTTP 401: sk-secret-key LeakedBody https://internal-dashscope/x")
        monkeypatch.setattr(ctx["agent"], "nodes",
                            {**ctx["agent"].nodes, "route": boom})
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test", timeout=60
        ) as c:
            async with c.stream("POST", "/api/chat",
                                json={"question": "杭州租房提取", "stream": True}) as resp:
                body = ""
                async for line in resp.aiter_lines():
                    body += line + "\n"
        assert "LeakedBody" not in body and "sk-secret" not in body
        assert "internal-dashscope" not in body
        assert '"type": "error"' in body.replace(" ", " ") or '"type":"error"' in body


def test_metadata_title_sanitized():
    """PC-6 后端纵深：上传标题/文号剥离 HTML 危险字符。"""
    from app.ingestion.metadata import extract_metadata

    meta = extract_metadata(
        "x.md",
        "# <img src=x onerror=alert(1)>深圳租房提取办法（演示）\n"
        "【发文字号】深公积金〔2026〕1号\n【状态】现行有效\n【适用城市】深圳\n\n"
        "第一条 深圳无房职工租赁提取每人每月最高2600元，按实际租金核算执行。\n",
    )
    assert "<" not in meta.title and ">" not in meta.title and '"' not in meta.title
    assert "深圳租房提取办法" in meta.title


async def test_stream_records_trace_jsonl(temp_env):
    """PC-10：stream 路径与非流式同构留档（此前主路径全程无档可复盘）。"""
    import httpx

    from app.api.main import app
    from app.config import settings
    from pathlib import Path

    trace_file = Path(settings.trace_dir) / "traces.jsonl"
    n0 = len(trace_file.read_text(encoding="utf-8").splitlines()) if trace_file.exists() else 0
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test", timeout=60
        ) as c:
            async with c.stream("POST", "/api/chat",
                                json={"question": "杭州租房提取每月最多能提多少", "stream": True}) as resp:
                async for _ in resp.aiter_lines():
                    pass
    n1 = len(trace_file.read_text(encoding="utf-8").splitlines()) if trace_file.exists() else 0
    assert n1 == n0 + 1  # 流式请求恰好落一条与 JSONL
