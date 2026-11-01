"""API 请求/响应模型（pydantic v2）。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000, description="用户问题")
    thread_id: str | None = Field(default=None, description="多轮对话线程 id")
    history: list[dict] | None = Field(
        default=None, description="历史对话 [{role, content}]，服务端按窗口截断"
    )
    stream: bool = Field(default=False, description="true 时返回 SSE 流")


class Source(BaseModel):
    no: int
    title: str
    doc_no: str = ""
    city: str = ""
    effective_date: str = ""
    sections: list[str] = []
    pages: list[int] = []
    is_valid: bool = True
    chunk_id: str = ""


class ChatResponse(BaseModel):
    trace_id: str
    question: str
    answer: str
    refusal: bool
    sources: list[Source]
    tool_result: dict[str, Any] | None = None
    events: list[dict]
    latency_ms: float


class DocumentUpload(BaseModel):
    filename: str = Field(description="文件名（需含政策元数据头）")
    content: str = Field(min_length=10, description="文件内容（markdown/txt）")


class DocumentUploadResponse(BaseModel):
    doc_id: str
    chunks: int
    retriever_refreshed: bool = True
