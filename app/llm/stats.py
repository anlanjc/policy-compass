"""运行时统计：请求量 / token / 重试 / JSON 解析失败率。

这些计数直接暴露在 GET /api/stats，是"结构化输出自校验把解析失败率从 X% 降到 Y%"
这类简历数字的观测来源。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class Stats:
    started_at: float = field(default_factory=time.time)

    llm_requests: int = 0
    llm_retries: int = 0
    llm_failures: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    json_calls: int = 0
    json_parse_failures: int = 0  # 首次解析失败（触发纠错重试）
    json_final_failures: int = 0  # 纠错后仍失败（上层降级）

    embed_calls: int = 0
    rerank_calls: int = 0
    rerank_fallbacks: int = 0
    queries_total: int = 0
    refusals: int = 0  # 走到诚实拒答分支的次数
    tool_calls: int = 0

    _lock = threading.Lock()

    def incr(self, name: str, n: int = 1) -> None:
        with self._lock:
            setattr(self, name, getattr(self, name) + n)

    def snapshot(self) -> dict:
        with self._lock:
            uptime = round(time.time() - self.started_at, 1)
            data = {
                k: getattr(self, k)
                for k in self.__dataclass_fields__
                if not k.startswith("_")
            }
            data["uptime_seconds"] = uptime
            if self.json_calls:
                data["json_first_pass_failure_rate"] = round(
                    self.json_parse_failures / self.json_calls, 4
                )
            return data


stats = Stats()
