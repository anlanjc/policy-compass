"""知识库热更新回归（PC-HR1）：指纹增量、内容反映、删除级联、在途快照隔离、后台重建端点。

冻结口径联动：未变更文件零重嵌入（embed 调用数不变）、零快照推进（version 不变）。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest


def _write_doc(path: Path, title: str, marker_sentence: str, doc_no: str = "热更〔2026〕1号") -> None:
    path.write_text(
        f"# {title}\n【发文字号】{doc_no}\n【生效日期】2026-01-01\n"
        f"【状态】现行有效\n【适用城市】深圳\n【领域】公积金\n\n"
        f"第一条 深圳无房职工租赁提取每人每月最高2600元。\n"
        f"第二条 {marker_sentence}\n",
        encoding="utf-8",
    )


@dataclass
class HR:
    store: object
    corpus: Path
    llm: object


@pytest.fixture()
async def hr(temp_env, tmp_path):
    """独立小语料的 store（不走 components 的全量真实语料，便于增删改可控）。"""
    from app.ingestion import IndexStore
    from app.llm.client import get_llm

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    store = IndexStore(temp_env.db_path)
    yield HR(store=store, corpus=corpus, llm=get_llm())
    store.close()


def _counting_embed(llm):
    calls = {"texts": 0}
    orig = llm.embed

    async def wrapped(texts):
        calls["texts"] += len(texts)
        return await orig(texts)

    return wrapped, calls


# ---------- 指纹未变 → 零重摄（不重嵌入、不重建快照） ----------


async def test_fingerprint_unchanged_zero_reembed(hr):
    _write_doc(hr.corpus / "doc_a.md", "深圳市租赁提取办法甲", "传统窗口办理需提前预约。")
    _write_doc(hr.corpus / "doc_b.md", "深圳市租赁提取办法乙", "线下网点周末不受理。", "热更〔2026〕2号")
    embed, calls = _counting_embed(hr.llm)

    r1 = await hr.store.sync(str(hr.corpus), embed)
    assert len(r1.added) == 2 and calls["texts"] > 0
    v1 = hr.store.snapshot.version
    texts_after_first = calls["texts"]

    r2 = await hr.store.sync(str(hr.corpus), embed)
    assert r2.skipped == 2 and not r2.added and not r2.updated and not r2.removed
    assert calls["texts"] == texts_after_first, "指纹未变不得重嵌入"
    assert hr.store.snapshot.version == v1, "零变更不得推进快照版本"
    assert r2.snapshot_version == v1
    assert r2.total_chunks == r1.total_chunks


# ---------- 修改语料文件 → 检索反映新内容 ----------


async def test_modified_file_reflected_in_retrieval(hr):
    from app.retrieval import HybridRetriever

    path = hr.corpus / "doc_a.md"
    _write_doc(path, "深圳市租赁提取办法甲", "传统窗口办理需提前预约。")
    await hr.store.sync(str(hr.corpus), hr.llm.embed)
    retriever = HybridRetriever(hr.store)

    r_old = await retriever.search("传统窗口办理要不要预约", use_rerank=False)
    assert r_old.chunks and r_old.chunks[0].chunk.doc_id == "doc_a.md"
    assert "传统窗口" in r_old.chunks[0].chunk.text

    _write_doc(path, "深圳市租赁提取办法甲", "线上小程序办理免预约即时到账。")
    report = await hr.store.sync(str(hr.corpus), hr.llm.embed)
    assert report.updated == ["doc_a.md"]

    r_gone = await retriever.search("传统窗口办理要不要预约", use_rerank=False)
    assert not any(c.chunk.doc_id == "doc_a.md" and "传统窗口" in c.chunk.text for c in r_gone.chunks)

    r_new = await retriever.search("线上小程序办理怎么预约", use_rerank=False)
    assert r_new.chunks and r_new.chunks[0].chunk.doc_id == "doc_a.md"
    assert "线上小程序" in r_new.chunks[0].chunk.text


# ---------- 删除语料文件 → 对应 chunk 级联消失 ----------


async def test_deleted_file_removes_chunks(hr):
    from app.retrieval import HybridRetriever

    _write_doc(hr.corpus / "doc_a.md", "深圳市租赁提取办法甲", "传统窗口办理需提前预约。")
    _write_doc(hr.corpus / "doc_b.md", "深圳市租赁提取办法乙", "线下网点周末不受理。", "热更〔2026〕2号")
    await hr.store.sync(str(hr.corpus), hr.llm.embed)
    retriever = HybridRetriever(hr.store)
    assert any(c.doc_id == "doc_b.md" for c in hr.store.chunks)

    (hr.corpus / "doc_b.md").unlink()
    report = await hr.store.sync(str(hr.corpus), hr.llm.embed)
    assert report.removed == ["doc_b.md"]
    assert not any(c.doc_id == "doc_b.md" for c in hr.store.chunks)

    r = await retriever.search("线下网点周末受理吗", use_rerank=False)
    assert not any(c.chunk.doc_id == "doc_b.md" for c in r.chunks)


# ---------- 在途请求旧快照隔离（PC-2 同款回归，作用于热更新路径） ----------


async def test_inflight_search_uses_old_snapshot_during_reindex(hr, monkeypatch):
    """检索中途整库热更新（sync 重摄+原子发布），在飞请求必须全程旧快照自洽。"""
    from app.llm import client as llm_client
    from app.retrieval import HybridRetriever

    path = hr.corpus / "doc_a.md"
    _write_doc(path, "深圳市租赁提取办法甲", "传统窗口办理需提前预约。")
    await hr.store.sync(str(hr.corpus), hr.llm.embed)
    retriever = HybridRetriever(hr.store)

    old_snap = hr.store.snapshot
    victim = next(c for c in old_snap.chunks if c.doc_id == "doc_a.md" and "传统窗口" in c.text)
    victim_vec = [float(x) for x in old_snap.matrix[old_snap.chunks.index(victim)]]

    real_llm = hr.llm

    class SyncDuringEmbedLLM:
        """首次 embed 时改写语料并整库 sync（新内容用同一 mock embed 空间重建），
        本次查询向量返回旧快照 victim 行——若实现串了新索引，旧文本查询的
        精确命中将位移。后续调用委托真实 mock embed（同空间，后置检索可用）。"""

        def __init__(self):
            self.triggered = False

        async def embed(self, texts):
            if not self.triggered:
                self.triggered = True
                _write_doc(path, "深圳市租赁提取办法甲", "线上小程序办理免预约即时到账。")
                await hr.store.sync(str(hr.corpus), real_llm.embed)
                return [victim_vec]
            return await real_llm.embed(texts)

    fake = SyncDuringEmbedLLM()  # 单实例：get_llm 每次返回同一对象，触发器只烧一次
    monkeypatch.setattr(llm_client, "get_llm", lambda: fake)

    r = await retriever.search(victim.text[:60], top_k=5, use_rerank=False)

    assert hr.store.snapshot.version != old_snap.version  # 库确实已热更新换版
    assert r.chunks, "检索应有结果"
    # 关键断言：top1 仍是旧库的块（旧快照自洽跑完），而非新库中的顶替内容
    assert r.chunks[0].chunk.chunk_id == victim.chunk_id
    assert "传统窗口" in r.chunks[0].chunk.text

    # 热更新完成后，新请求自动取新快照：新内容可检，旧内容消失
    r_new = await retriever.search("线上小程序办理免预约吗", use_rerank=False)
    assert r_new.chunks and r_new.chunks[0].chunk.doc_id == "doc_a.md"
    assert "线上小程序" in r_new.chunks[0].chunk.text
    assert not any("传统窗口" in c.chunk.text for c in r_new.chunks)


# ---------- 解析失败 ≠ 删除（PC-HR1 真缺陷修复回归） ----------


async def test_parse_failure_keeps_existing_doc(hr, monkeypatch):
    """磁盘上存在但瞬时读不出的文件，不得被当作"已删除"级联清库。

    缺陷背景：/api/ingest/sync 未传 llm → 扫描件 PDF 解析 ValueError → 旧实现
    落入 gone 分支被静默删除。修后语义：已入库文档保留旧版原样可检，errors 留痕。
    """
    _write_doc(hr.corpus / "doc_a.md", "深圳市租赁提取办法甲", "传统窗口办理需提前预约。")
    await hr.store.sync(str(hr.corpus), hr.llm.embed)
    v1 = hr.store.snapshot.version
    chunks_before = [c for c in hr.store.chunks if c.doc_id == "doc_a.md"]
    assert chunks_before

    from app.ingestion import indexer as indexer_mod

    orig_load = indexer_mod.load_file

    async def flaky_load(path, llm=None):
        if path.name == "doc_a.md":
            raise ValueError("模拟瞬时 IO 抖动/缺视觉模型")
        return await orig_load(path, llm=llm)

    monkeypatch.setattr(indexer_mod, "load_file", flaky_load)
    report = await hr.store.sync(str(hr.corpus), hr.llm.embed)
    assert report.errors == ["doc_a.md"]
    assert report.removed == [] and report.updated == []
    assert hr.store.snapshot.version == v1, "解析失败未变更库，不得推进快照"
    assert [c for c in hr.store.chunks if c.doc_id == "doc_a.md"] == chunks_before


# ---------- 管理端点：后台重建 + singleflight + 鉴权（PC-3 门禁） ----------


async def test_reindex_endpoint_auth(temp_env, monkeypatch):
    import httpx

    from app.api.main import app
    from app.config import settings

    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test", timeout=60
    ) as c:
        # 未配置 key：与既有写接口同一道闸（默认禁用 403）
        monkeypatch.setattr(settings, "admin_key", "")
        assert (await c.post("/api/ingest/reindex")).status_code == 403
        assert (await c.get("/api/ingest/reindex/status")).status_code == 403

        # 配置 key：无头/错头 401
        monkeypatch.setattr(settings, "admin_key", "secret-key")
        assert (await c.post("/api/ingest/reindex")).status_code == 401
        assert (
            await c.get("/api/ingest/reindex/status", headers={"X-Admin-Key": "wrong"})
        ).status_code == 401


async def test_reindex_endpoint_background_and_singleflight(temp_env, monkeypatch):
    import httpx

    from app.api import routes
    from app.api.main import app
    from app.config import settings

    monkeypatch.setattr(settings, "admin_key", "secret-key")
    h = {"X-Admin-Key": "secret-key"}

    async with app.router.lifespan_context(app):
        ctx = routes.get_app_state()
        release = asyncio.Event()
        orig_sync = ctx["store"].sync

        async def slow_sync(*a, **k):
            await release.wait()
            return await orig_sync(*a, **k)

        monkeypatch.setattr(ctx["store"], "sync", slow_sync)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test", timeout=60
        ) as c:
            before = ctx["store"].snapshot.version

            r1 = await c.post("/api/ingest/reindex", headers=h)
            assert r1.status_code == 202
            assert r1.json()["status"] == "started"
            assert r1.json()["snapshot_version"] == before

            # singleflight：构建进行中重复触发 409（不排队不叠跑）
            r2 = await c.post("/api/ingest/reindex", headers=h)
            assert r2.status_code == 409

            st = (await c.get("/api/ingest/reindex/status", headers=h)).json()
            assert st["running"] is True

            release.set()
            for _ in range(200):
                st = (await c.get("/api/ingest/reindex/status", headers=h)).json()
                if not st["running"]:
                    break
                await asyncio.sleep(0.01)
            assert st["running"] is False and st["error"] is None
            assert st["last_report"] and st["last_report"]["total_chunks"] > 0
            # 真实语料全量命中指纹 → 快照版本不推进（零变更零发布）；
            # llm 已传 → 扫描件正常解析，无删除无解析错误（PC-HR1 级联删除缺陷回归）
            assert st["last_report"]["skipped"] >= 20
            assert st["last_report"]["removed"] == [] and st["last_report"]["errors"] == []
            assert st["last_report"]["snapshot_version"] == before
            assert st["snapshot_version"] == before

            # 完成后可再次触发（singleflight 已释放）
            monkeypatch.setattr(ctx["store"], "sync", orig_sync)
            r3 = await c.post("/api/ingest/reindex", headers=h)
            assert r3.status_code == 202
            for _ in range(200):
                st = (await c.get("/api/ingest/reindex/status", headers=h)).json()
                if not st["running"]:
                    break
                await asyncio.sleep(0.01)
            assert st["running"] is False and st["error"] is None


async def test_reindex_error_does_not_stuck_singleflight(temp_env, monkeypatch):
    import httpx

    from app.api import routes
    from app.api.main import app
    from app.config import settings

    monkeypatch.setattr(settings, "admin_key", "secret-key")
    h = {"X-Admin-Key": "secret-key"}

    async with app.router.lifespan_context(app):
        ctx = routes.get_app_state()

        async def broken_sync(*a, **k):
            raise RuntimeError("boom")

        monkeypatch.setattr(ctx["store"], "sync", broken_sync)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test", timeout=60
        ) as c:
            r = await c.post("/api/ingest/reindex", headers=h)
            assert r.status_code == 202
            for _ in range(200):
                st = (await c.get("/api/ingest/reindex/status", headers=h)).json()
                if not st["running"]:
                    break
                await asyncio.sleep(0.01)
            assert st["running"] is False
            assert st["error"] and "boom" not in st["error"]  # PC-9：异常明文不外发
            assert st["last_report"] is None

            # 失败释放 singleflight：可重新触发
            r2 = await c.post("/api/ingest/reindex", headers=h)
            assert r2.status_code == 202  # 仍走 broken_sync，但证明 409 已解除
            for _ in range(200):
                st = (await c.get("/api/ingest/reindex/status", headers=h)).json()
                if not st["running"]:
                    break
                await asyncio.sleep(0.01)
            assert st["running"] is False
