"""增量摄取语料目录（知识库热更新 CLI 入口）。

用法：
    python scripts/ingest.py --reindex-incremental   # 增量热更新（推荐写法）
    python scripts/ingest.py                          # 等价（向后兼容的历史入口）

带 hash 的增量同步：未变化文件跳过（零重嵌入零重建），变更文件自动重切重嵌，
删除文件级联清理；完成后一次性原子发布新快照（PC-2 机制），在途请求不受影响。
管理端点形态见 POST /api/ingest/reindex（后台构建 + singleflight）。
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.ingestion import IndexStore
from app.llm.client import get_llm


async def main() -> None:
    ap = argparse.ArgumentParser(description="policy-compass 语料增量热更新")
    ap.add_argument(
        "--reindex-incremental",
        action="store_true",
        help="执行增量重建（指纹未变跳过/变更重嵌/删除清理）；缺省时行为相同，"
        "此旗标用于脚本与文档中的显式触发面",
    )
    ap.parse_args()

    llm = get_llm()
    store = IndexStore()
    before = store.snapshot.version
    report = await store.sync(settings.corpus_dir, llm.embed, llm=llm)
    print(
        f"docs: +{len(report.added)} ~{len(report.updated)} -{len(report.removed)} "
        f"| skipped(hash 未变): {report.skipped} | total chunks: {report.total_chunks}"
    )
    print(f"snapshot: v{before} -> v{report.snapshot_version}")
    store.close()


if __name__ == "__main__":
    asyncio.run(main())
