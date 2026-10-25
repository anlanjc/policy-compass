"""增量摄取语料目录：python scripts/ingest.py

带 hash 的增量同步：未变化文件跳过，变更文件自动重切重嵌，删除文件级联清理。
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.ingestion import IndexStore
from app.llm.client import get_llm


async def main() -> None:
    llm = get_llm()
    store = IndexStore()
    report = await store.sync(settings.corpus_dir, llm.embed, llm=llm)
    print(
        f"docs: +{len(report.added)} ~{len(report.updated)} -{len(report.removed)} "
        f"| skipped(hash 未变): {report.skipped} | total chunks: {report.total_chunks}"
    )
    store.close()


if __name__ == "__main__":
    asyncio.run(main())
