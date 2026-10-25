"""扫描件 PDF 的视觉模型 OCR 兜底。

真实政务 PDF 的常态是红头文件扫描件——pypdf 对其抽出的文本近于零（实测
4 页仅 3 字符）。本模块提供第二条解析路径：pymupdf 渲染页面为图片 →
视觉 LLM（qwen-vl 系列，OpenAI-compatible vision 格式）逐页转录 → 拼接文本。

摄取管线的调用约定见 loader.load_file：pypdf 文本密度过低时自动走此路径，
OCR 不可用（无视觉模型/调用失败）时抛出明确异常由上层跳过该文档并记录。
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

from ..config import settings
from ..llm.client import LLMClient, LLMError

logger = logging.getLogger(__name__)

OCR_PROMPT = (
    "你是专业的公文 OCR 引擎。将图片中的公文逐字转录为纯文本：\n"
    "1. 红头标题、发文字号（如 杭房公委〔2025〕4号）、正文、落款日期全部转录；\n"
    "2. 保留原有段落与条款结构，不要合并或改写；\n"
    "3. 只输出转录文本，不要任何解释、标注或 markdown 符号。"
)

# 每页抽取字符数低于该阈值判定为扫描件（正文页正常应远高于此）
TEXT_DENSITY_THRESHOLD = 30


def is_scanned_pdf(text: str, pages: int) -> bool:
    """pypdf 抽取结果判定：整体密度过低即视为扫描件。"""
    if pages == 0:
        return False
    return len(text.strip()) / pages < TEXT_DENSITY_THRESHOLD


def render_pages(pdf_path: str | Path, dpi: int = 150) -> list[bytes]:
    """PDF 页面渲染为 PNG 字节（pymupdf，无外部依赖）。"""
    import pymupdf

    doc = pymupdf.open(str(pdf_path))
    out = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        out.append(pix.tobytes("png"))
    doc.close()
    return out


async def ocr_pdf(llm: LLMClient, pdf_path: str | Path, max_pages: int = 10) -> str:
    """视觉模型逐页 OCR，返回拼接文本。

    结果按文件内容哈希缓存到磁盘：OCR 昂贵（每页一次视觉模型调用），
    重复摄取/重建索引/离线测试都应复用缓存；缓存未命中且当前为 mock
    模式（无真实视觉模型）时抛错，由上层跳过该文档——保证测试可离线。
    """
    pdf_path = Path(pdf_path)
    cache_dir = Path(settings.ocr_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    import hashlib

    file_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:16]
    cache_file = cache_dir / f"{pdf_path.stem}_{file_hash}.txt"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")
    if settings.effective_llm_mode == "mock":
        raise ValueError(
            f"{pdf_path.name}: 扫描件无 OCR 缓存且 mock 模式无视觉模型（提交 data/ocr_cache 可离线测试）"
        )

    pages = render_pages(pdf_path)
    texts: list[str] = []
    for i, png in enumerate(pages[:max_pages]):
        b64 = base64.b64encode(png).decode()
        payload = {
            "model": settings.ocr_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        {"type": "text", "text": OCR_PROMPT},
                    ],
                }
            ],
            "temperature": 0.0,
        }
        try:
            data = await llm._post_with_retry("/chat/completions", payload)
            texts.append(data["choices"][0]["message"]["content"] or "")
        except (LLMError, KeyError, IndexError) as e:
            logger.warning("OCR 第 %d 页失败: %s", i + 1, e)
    text = "\n".join(t.strip() for t in texts if t.strip())
    if text:
        cache_file.write_text(text, encoding="utf-8")
    return text
