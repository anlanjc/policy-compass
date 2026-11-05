"""FastAPI 应用入口：lifespan 内构建 store/retriever/agent 单例。

启动流程：加载（或增量同步）索引 → 构建 BM25/向量混合检索 → 编译 LangGraph。
首次启动自动摄取 data/corpus（增量，未变化文件跳过），随后可经
POST /api/documents 与 POST /api/ingest/sync 热更新。
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .. import __version__
from ..config import settings
from ..llm.client import get_llm
from ..obs.trace import TraceRecorder
from ..retrieval import HybridRetriever
from ..ingestion import IndexStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    llm = get_llm()
    store = IndexStore()
    if not store.chunks:
        report = await store.sync(None, llm.embed, llm=llm)
        logger.info("语料同步完成: %s", report.as_dict())
    retriever = HybridRetriever(store)

    from ..agent import PolicyAgent

    agent = PolicyAgent(retriever, llm)
    app_state.update(
        {
            "llm": llm,
            "store": store,
            "retriever": retriever,
            "agent": agent,
            "recorder": TraceRecorder(),
        }
    )
    logger.info(
        "PolicyCompass 就绪 | llm_mode=%s | docs=%d chunks=%d",
        settings.effective_llm_mode, store.doc_count(), len(store.chunks),
    )
    yield
    await llm.close()
    await retriever.reranker.close()
    store.close()


app = FastAPI(
    title="政策罗盘 PolicyCompass",
    description="跨城市社保/公积金政策 Agentic RAG 问答服务"
    "（CRAG 自纠错检索 + Function Calling 计算 + 忠实度校验）",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    # 跨域白名单（S3 评审 PC-3）：默认空=仅同源（演示前端由本服务托管，天然同源）；
    # 跨域演示前端用 POLICY_COMPASS_CORS_ORIGINS 显式列出，不再放 *（drive-by 防护）
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

from .routes import router  # noqa: E402  (需在 lifespan 定义后导入避免循环)

app.include_router(router)

# 演示前端（static/index.html）：SSE 打字机 + 实时 agent 决策路径
_static_dir = __import__("pathlib").Path(__file__).resolve().parent.parent.parent / "static"
if _static_dir.exists():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="static")
