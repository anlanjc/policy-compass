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


async def test_get_llm_singleton_lifecycle(temp_env, monkeypatch):
    """get_llm 进程级缓存单例（POLICY-1，评审 4.0）。

    修复前：每次 get_llm() 新建 LLMClient（构造含独立 httpx.AsyncClient）且
    无人 aclose——检索热路径每查询泄漏连接池（真实 embedding 部署 FD 随累计
    请求增长）。修复=配置签名缓存：同签名同实例、签名失配自动重建、
    reset_llm_cache() 强制丢弃（lifespan 关闭后调用）。
    """
    from app.config import settings
    from app.llm.client import get_llm, reset_llm_cache

    reset_llm_cache()  # 隔离其他测试的缓存状态
    try:
        a, b = get_llm(), get_llm()
        assert a is b, "同签名 get_llm 必须返回同一实例（每调用新建=连接池泄漏）"
        if settings.effective_llm_mode == "mock":
            assert not hasattr(a, "_http"), "mock 客户端不应持有 httpx 连接池"
        # 签名失配（进程内改 embed 开关）→ 自动重建，不返回已缓存旧实例
        old = a
        saved = settings.embed_enabled
        try:
            settings.embed_enabled = not saved
            assert get_llm() is not old, "配置签名变更后必须重建客户端"
        finally:
            settings.embed_enabled = saved
        assert get_llm() is not old  # 恢复后签名再次失配，仍是新实例
    finally:
        reset_llm_cache()


async def test_embed_independent_endpoint(temp_env, monkeypatch):
    """embed 独立端点/密钥生效（POLICY-7，评审 4.0）。

    修复前：EMBED_API_BASE/EMBED_API_KEY 是死配置（effective_embed_* 无运行时
    消费者），/embeddings 永远打 LLM 端点用 LLM key 鉴权——README/.env.example/
    compose 教配的独立 embedding 供应商配了不生效（当时可用纯因两端点恰好相同）。
    """
    import httpx

    from app.config import settings
    from app.llm.client import LLMClient

    saved = (settings.embed_enabled, settings.embed_api_base, settings.embed_api_key)
    try:
        # 1) embed 端点/key 与 LLM 相同 → 复用主客户端（零额外连接池）
        monkeypatch.setattr(settings, "embed_enabled", True)
        monkeypatch.delenv("EMBED_API_BASE", raising=False)
        monkeypatch.delenv("EMBED_API_KEY", raising=False)
        c_same = LLMClient()
        assert c_same._embed_http is None

        # 2) 独立端点+key → 建独立客户端，请求打 /embeddings 且用 embed key 鉴权
        monkeypatch.setattr(settings, "embed_api_base", "https://embed.example.com/v1")
        monkeypatch.setattr(settings, "embed_api_key", "sk-embed-test")
        c_diff = LLMClient()
        assert c_diff._embed_http is not None
        assert str(c_diff._embed_http.base_url).rstrip("/") == "https://embed.example.com/v1"
        assert "sk-embed-test" in c_diff._embed_http.headers["Authorization"]

        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["auth"] = request.headers.get("Authorization", "")
            return httpx.Response(
                200, json={"data": [{"index": 0, "embedding": [0.1, 0.2]}]}
            )

        c_diff._embed_http = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://embed.example.com/v1",
            headers={"Authorization": "Bearer sk-embed-test"},
        )
        vecs = await c_diff.embed(["文本"])
        assert seen["url"] == "https://embed.example.com/v1/embeddings"
        assert seen["auth"] == "Bearer sk-embed-test"
        assert vecs == [[0.1, 0.2]]
        await c_diff.close()  # 独立 embed 客户端一并关闭（不悬挂）
    finally:
        settings.embed_enabled, settings.embed_api_base, settings.embed_api_key = saved


# ---------- POLICY-10（评审 4.0）：流式首字节前重试 + 统一失败计数/包装 ----------


def _sse_body(deltas: list[str]) -> str:
    """构造 OpenAI 兼容 SSE 响应体。"""
    frames = [
        "data: " + json.dumps({"choices": [{"delta": {"content": d}}]},
                              ensure_ascii=False) + "\n\n"
        for d in deltas
    ]
    frames.append("data: [DONE]\n\n")
    return "".join(frames)


async def _stream_client(monkeypatch, handler, max_retries: int):
    """真实 LLMClient + MockTransport 替身（与 PC-8 测试同法，零真实网络）。"""
    import httpx

    from app.config import settings
    from app.llm.client import LLMClient

    monkeypatch.setattr(settings, "llm_max_retries", max_retries)
    c = LLMClient()
    await c._http.aclose()
    c._http = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url=settings.llm_api_base
    )
    return c


async def test_stream_retry_before_first_byte(temp_env, monkeypatch):
    """POLICY-10：首字节前的 retryable 状态按 _post_with_retry 同口径退避重试。

    修复前：429/502/503 直接抛 LLMError 终止整个 SSE（前端唯一路径
    static/index.html stream:true），非流式同窗口有 3 次退避——瞬时抖动
    直接杀死整条流式回答。修复=首字节前同口径有限退避（llm_retries 计数）。
    """
    import httpx

    from app.llm.stats import stats

    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        if attempts["n"] <= 2:
            return httpx.Response(503, text="upstream overloaded")
        return httpx.Response(
            200, text=_sse_body(["你好", "，世界"]),
            headers={"content-type": "text/event-stream"},
        )

    c = await _stream_client(monkeypatch, handler, max_retries=2)
    try:
        before = stats.snapshot()
        deltas = [d async for d in c.chat_stream([{"role": "user", "content": "hi"}])]
        after = stats.snapshot()
        assert "".join(deltas) == "你好，世界", "重试后应产出完整回答"
        assert attempts["n"] == 3, "两次 503 + 一次成功，退避重试必须发生"
        assert after["llm_retries"] - before["llm_retries"] == 2, "重试计数缺失"
    finally:
        await c._http.aclose()


async def test_stream_exhaustion_counted_and_wrapped(temp_env, monkeypatch):
    """POLICY-10：流式耗尽重试后 LLMError + llm_failures 计数。

    修复前：retryable 分支既不重试也不记 llm_failures（对比 _post_with_retry
    的统一失败计数）——/api/stats 对流式路径的失败率观测失真。
    """
    import httpx

    from app.llm.client import LLMError
    from app.llm.stats import stats

    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        return httpx.Response(429, text="rate limited")

    c = await _stream_client(monkeypatch, handler, max_retries=1)
    try:
        before = stats.snapshot()
        raised = ""
        try:
            async for _ in c.chat_stream([{"role": "user", "content": "hi"}]):
                pass
        except LLMError as e:
            raised = str(e)
        after = stats.snapshot()
        assert raised, "耗尽重试必须抛 LLMError"
        assert "429" in raised or "已重试" in raised
        assert attempts["n"] == 2, "max_retries=1 → 1 次尝试 + 1 次重试"
        assert after["llm_retries"] - before["llm_retries"] == 1
        assert after["llm_failures"] - before["llm_failures"] >= 1, (
            "llm_failures 计数缺失（retryable 分支零计数）"
        )
    finally:
        await c._http.aclose()


async def test_stream_http_error_wrapped_as_llmerror(temp_env, monkeypatch):
    """POLICY-10：非重试型 HTTP 错误（401）统一包装为 LLMError。

    修复前：raise_for_status() 的 httpx.HTTPStatusError 不在 except 列表内
    裸穿透——违反"HTTP 错误统一包装为 LLMError"契约（_post_with_retry 同契约）。
    """
    import httpx

    from app.llm.client import LLMError

    def handler(request):
        return httpx.Response(401, text='{"error": "invalid api key"}')

    c = await _stream_client(monkeypatch, handler, max_retries=0)
    try:
        exc: Exception | None = None
        try:
            async for _ in c.chat_stream([{"role": "user", "content": "hi"}]):
                pass
        except Exception as e:  # noqa: BLE001 —— 本测试就是要验异常类型
            exc = e
        assert isinstance(exc, LLMError), (
            f"应为 LLMError，实际裸抛 {type(exc).__name__}（违反统一包装契约）"
        )
        assert "401" in str(exc)
    finally:
        await c._http.aclose()


async def test_stream_no_retry_after_first_byte(temp_env, monkeypatch):
    """POLICY-10 诚实边界：已产出增量后中途断流不得重试。

    重试会向客户端重复已写内容——只能以 LLMError 终止本次流（沿用既有
    中途失败语义：llm_failures 计数 + '流式请求失败'）。此测试在修复前后
    都必须绿：守门修复不得把重试越界到首字节之后。
    """
    import httpx

    from app.llm.client import LLMError
    from app.llm.stats import stats

    attempts = {"n": 0}

    async def broken_stream():
        # 只给一个增量帧（不给 [DONE]），随后连接中断——模拟"首字节已写出
        # 后流被掐断"；若带 [DONE]，流会正常结束而非中断（测试装置需如实模拟）
        frame = "data: " + json.dumps(
            {"choices": [{"delta": {"content": "第一"}}]}, ensure_ascii=False
        ) + "\n\n"
        yield frame.encode("utf-8")
        raise httpx.ReadError("连接中断")

    def handler(request):
        attempts["n"] += 1
        return httpx.Response(
            200, content=broken_stream(),
            headers={"content-type": "text/event-stream"},
        )

    c = await _stream_client(monkeypatch, handler, max_retries=3)
    try:
        before = stats.snapshot()
        collected = []
        raised = ""
        try:
            async for d in c.chat_stream([{"role": "user", "content": "hi"}]):
                collected.append(d)
        except LLMError as e:
            raised = str(e)
        after = stats.snapshot()
        assert "".join(collected) == "第一", "首字节必须已产出"
        assert raised, "中途断流以 LLMError 终止"
        assert attempts["n"] == 1, "首字节后绝不重试（重复内容）"
        assert after["llm_retries"] - before["llm_retries"] == 0
        assert after["llm_failures"] - before["llm_failures"] >= 1
    finally:
        await c._http.aclose()
