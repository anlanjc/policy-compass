"""文档加载：markdown / txt / pdf（含扫描件 OCR 兜底）。

真实政务 PDF 的现实：相当比例是红头扫描件，pypdf 抽出文本近于零（实测
某 4 页红头文件仅 3 字符）。因此 PDF 路径设计为两级：先 pypdf 文本抽取，
页均字符密度低于阈值判定为扫描件，自动降级视觉模型 OCR（见 ocr.py）。
"""

from __future__ import annotations

from pathlib import Path

from .ocr import is_scanned_pdf, ocr_pdf


async def load_file(path: Path, llm=None) -> str:
    """加载文档文本。llm 为 LLMClient 实例，仅在扫描件 OCR 时使用。"""
    suffix = path.suffix.lower()
    if suffix in (".md", ".markdown", ".txt"):
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if is_scanned_pdf(text, len(reader.pages)):
            if llm is None:
                raise ValueError(
                    f"{path.name}: 疑似扫描件（文本密度过低）且未提供视觉模型，无法解析"
                )
            from .ocr import ocr_pdf

            text = await ocr_pdf(llm, path)
            if not text.strip():
                raise ValueError(f"{path.name}: 扫描件 OCR 失败（视觉模型无返回）")
        return text
    raise ValueError(f"不支持的文件类型: {suffix}")


def list_corpus(corpus_dir: str | Path) -> list[Path]:
    root = Path(corpus_dir)
    if not root.exists():
        return []
    return sorted(
        p
        for p in root.rglob("*")
        if p.suffix.lower() in (".md", ".markdown", ".txt", ".pdf")
    )
