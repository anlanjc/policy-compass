"""LLM 客户端：OpenAI 兼容协议的异步薄封装（httpx）。

刻意不依赖 openai SDK，目的：
1. 讲清协议本质——一次 POST /chat/completions，流式即 SSE `data:` 行流；
2. 超时、指数退避重试、结构化输出自校验全部自主可控（面试常问的可靠性细节）。

可靠性设计：
- 仅对"可重试错误"退避重试：网络异常、HTTP 429/5xx；4xx 参数错误直接抛出；
- chat_json 结构化输出：解析失败时把错误信息回灌给模型纠错重试（默认 2 轮），
  仍失败抛 StructuredOutputError，由编排层降级（拒答/跳过），绝不让异常裸露给用户。
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from typing import AsyncIterator, Sequence, Type, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from ..config import settings
from .stats import stats

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class LLMError(Exception):
    """LLM 调用最终失败（已耗尽重试）。"""


class StructuredOutputError(LLMError):
    """结构化输出解析/校验失败（已耗尽纠错轮次）。"""


def extract_json(text: str) -> str | None:
    """从模型输出中抽取第一个平衡的 JSON 对象。

    兼容：裸 JSON、markdown 代码围栏、前后夹杂解释文字。
    扫描时忽略字符串字面量内部的大括号，避免被字符串里的 '}' 截断。
    """
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        start = text.find("{", start + 1)
    return None


class LLMClient:
    """真实后端：OpenAI 兼容 /chat/completions 与 /embeddings。"""

    def __init__(self) -> None:
        self._http = httpx.AsyncClient(
            base_url=settings.llm_api_base.rstrip("/"),
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            timeout=settings.llm_timeout,
        )

    # ---------- 底层请求 ----------

    async def _post_with_retry(self, url: str, payload: dict) -> dict:
        delay = 0.5
        last_exc: Exception | None = None
        for attempt in range(settings.llm_max_retries + 1):
            try:
                stats.incr("llm_requests")
                resp = await self._http.post(url, json=payload)
                if resp.status_code in RETRYABLE_STATUS:
                    raise LLMError(f"retryable status {resp.status_code}")
                resp.raise_for_status()
                data = resp.json()
                usage = data.get("usage") or {}
                stats.incr("prompt_tokens", usage.get("prompt_tokens", 0))
                stats.incr("completion_tokens", usage.get("completion_tokens", 0))
                return data
            except (httpx.TimeoutException, httpx.TransportError, LLMError) as e:
                last_exc = e
                if attempt < settings.llm_max_retries:
                    stats.incr("llm_retries")
                    jitter = random.uniform(0, 0.3)
                    await asyncio.sleep(delay + jitter)
                    delay *= 2
        stats.incr("llm_failures")
        raise LLMError(f"LLM 请求失败（已重试 {settings.llm_max_retries} 次）: {last_exc}")

    # ---------- 对话 ----------

    async def chat(
        self,
        messages: Sequence[dict],
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> str:
        payload: dict = {
            "model": settings.llm_model,
            "messages": list(messages),
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        data = await self._post_with_retry("/chat/completions", payload)
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as e:
            raise LLMError(f"响应结构异常: {e}: {data}") from e

    async def chat_stream(
        self,
        messages: Sequence[dict],
        temperature: float = 0.3,
    ) -> AsyncIterator[str]:
        """SSE 流式对话：逐 token 产出增量文本。

        协议要点：响应为 text/event-stream，每帧 `data: {...}`，
        以 `data: [DONE]` 结束；增量在 choices[0].delta.content。
        """
        payload = {
            "model": settings.llm_model,
            "messages": list(messages),
            "temperature": temperature,
            "stream": True,
        }
        stats.incr("llm_requests")
        try:
            async with self._http.stream(
                "POST", "/chat/completions", json=payload
            ) as resp:
                if resp.status_code in RETRYABLE_STATUS:
                    raise LLMError(f"retryable status {resp.status_code}")
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = obj.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta", {}).get("content")
                        if delta:
                            stats.incr("completion_tokens", 0)  # 流式无 usage，token 见估算
                            yield delta
        except (httpx.TimeoutException, httpx.TransportError) as e:
            stats.incr("llm_failures")
            raise LLMError(f"流式请求失败: {e}") from e

    # ---------- 结构化输出（带自校验纠错） ----------

    async def chat_json(
        self,
        messages: Sequence[dict],
        schema: Type[T],
        temperature: float = 0.0,
    ) -> T:
        stats.incr("json_calls")
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        msgs = [
            {
                "role": "system",
                "content": f"你是严谨的策略执行器。只输出一个 JSON 对象，"
                f"符合此 JSON Schema（不要输出任何解释文字或代码围栏）：\n{schema_json}",
            },
            *messages,
        ]
        last_err = ""
        for round_ in range(settings.json_repair_rounds + 1):
            content = await self.chat(msgs, temperature=temperature)
            raw = extract_json(content)
            if raw is None:
                last_err = "输出中未找到 JSON 对象"
            else:
                try:
                    return schema.model_validate_json(raw)
                except ValidationError as e:
                    last_err = str(e)[:500]
            stats.incr("json_parse_failures")
            logger.warning("结构化输出第 %d 轮解析失败: %s", round_, last_err)
            msgs = msgs[:2] + [
                msgs[1],
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": f"你的输出无法解析：{last_err}。"
                    f"请重新只输出符合 Schema 的 JSON 对象。",
                },
            ]
        stats.incr("json_final_failures")
        raise StructuredOutputError(f"结构化输出解析失败: {last_err}")

    # ---------- Embedding ----------

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not settings.embed_enabled:
            # 零向量降级：向量检索相似度恒为 0 被过滤，检索退化为纯 BM25（架构预留路径）
            return [[0.0] * settings.embed_dim for _ in texts]
        results: list[list[float]] = []
        for i in range(0, len(texts), settings.embed_batch_size):
            batch = texts[i : i + settings.embed_batch_size]
            stats.incr("embed_calls")
            data = await self._post_with_retry(
                "/embeddings", {"model": settings.embed_model, "input": batch}
            )
            items = sorted(data["data"], key=lambda x: x["index"])
            results.extend(item["embedding"] for item in items)
        return results

    async def close(self) -> None:
        await self._http.aclose()


def get_llm() -> LLMClient:
    """按配置返回真实或 mock 实现（延迟导入避免测试路径依赖）。"""
    if settings.effective_llm_mode == "mock":
        from .mock import MockLLMClient

        return MockLLMClient()  # type: ignore[return-value]
    return LLMClient()
