"""测试全局夹具：强制 mock 模式 + 临时数据库 + 共享组件。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["POLICY_COMPASS_FORCE_MOCK"] = "1"


@pytest.fixture()
def temp_env(tmp_path, monkeypatch):
    """每测试独立 db/trace 目录，互不污染。"""
    from app.config import get_settings, reset_settings_cache

    monkeypatch.setenv("DB_PATH", str(tmp_path / "index.db"))
    monkeypatch.setenv("TRACE_DIR", str(tmp_path / "traces"))
    reset_settings_cache()
    yield get_settings()
    reset_settings_cache()


@pytest.fixture()
async def components(temp_env):
    """store（已摄取演示语料）+ retriever + llm + agent。"""
    import asyncio

    from app.agent import PolicyAgent
    from app.ingestion import IndexStore
    from app.llm.client import get_llm
    from app.retrieval import HybridRetriever

    llm = get_llm()
    store = IndexStore(temp_env.db_path)
    await store.sync(temp_env.corpus_dir, llm.embed)
    retriever = HybridRetriever(store)
    agent = PolicyAgent(retriever, llm)
    yield {"llm": llm, "store": store, "retriever": retriever, "agent": agent}
    store.close()
