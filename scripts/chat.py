"""CLI 问答演示：python scripts/chat.py "你的问题"

打印 agent 决策路径（trace）与最终答案，适合录屏/截图演示。
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


async def main() -> None:
    if len(sys.argv) < 2:
        print('用法: python scripts/chat.py "杭州公积金租房提取每月最多能提多少"')
        return
    question = sys.argv[1]

    from app.agent import PolicyAgent
    from app.config import settings
    from app.ingestion import IndexStore
    from app.llm.client import get_llm
    from app.retrieval import HybridRetriever

    llm = get_llm()
    store = IndexStore()
    if not store.chunks:
        print("(首次运行，摄取语料…)")
        await store.sync(settings.corpus_dir, llm.embed)
    agent = PolicyAgent(HybridRetriever(store), llm)

    print(f"\n问题：{question}\n{'-' * 56}")
    state = await agent.run(question)
    print("决策路径：")
    for e in state.get("events", []):
        label = {
            "route": lambda: f"路由 → {e.get('city')}/{e.get('domain')} 需计算={e.get('need_calc')}",
            "retrieve": lambda: f"检索(第{e.get('round')}轮) → top1《{str(e.get('top1'))[:20]}》",
            "grade": lambda: f"评分 → max={e.get('max_grade')} avg={e.get('avg_grade')}",
            "rewrite": lambda: f"改写 → {str(e.get('new_query'))[:40]}",
            "tool": lambda f=None: f"工具 → {e.get('tool')}({e.get('args')}) ok={e.get('ok')}",
            "generate": lambda: f"生成(第{e.get('attempt')}次) → {e.get('chars')}字",
            "verify": lambda: f"忠实度校验 → grounded={e.get('grounded')}",
            "fallback": lambda: "诚实拒答分支",
        }.get(e["event"], lambda: e["event"])
        try:
            text = label()
        except TypeError:
            text = label(None)
        print(f"  · {text}")
    print("-" * 56)
    print(f"回答：\n{state.get('answer', '')}")
    if state.get("sources"):
        print("引用：")
        for s in state["sources"]:
            mark = "现行有效" if s["is_valid"] else "已废止"
            print(f"  [{s['no']}] 《{s['title']}》{s.get('doc_no', '')}（{mark}）")
    store.close()


if __name__ == "__main__":
    asyncio.run(main())
