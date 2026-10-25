"""请求级 trace 落盘（JSONL）：每次问答的决策路径、耗时、引用来源。

可观测性设计：agent 的"黑盒感"是演示与调试的最大敌人——每步决策
（路由、检索、评分、改写、工具、校验）都记录为事件，SSE 实时推送 + JSONL
留档，事后可用 jq/pandas 复盘任何一次回答的完整决策链。
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from ..config import settings


class TraceRecorder:
    def __init__(self, trace_dir: str | None = None) -> None:
        self.path = Path(trace_dir or settings.trace_dir)
        self.path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(
        self,
        trace_id: str,
        question: str,
        events: list[dict],
        answer: str,
        refusal: bool,
        sources: list[dict],
        latency_ms: float,
    ) -> None:
        entry = {
            "trace_id": trace_id,
            "ts": round(time.time(), 3),
            "question": question,
            "events": events,
            "answer": answer,
            "refusal": refusal,
            "sources": sources,
            "latency_ms": latency_ms,
        }
        line = json.dumps(entry, ensure_ascii=False)
        with self._lock, open(self.path / "traces.jsonl", "a", encoding="utf-8") as f:
            f.write(line + "\n")
