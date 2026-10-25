"""文档加载：markdown / txt / pdf。

PDF 仅做基础文本抽取（pypdf）；真实政务 PDF 常含表格与印章，复杂版式解析
不是本项目重点，接口留作扩展点（详见 docs/DESIGN.md 的取舍说明）。
"""

from __future__ import annotations

from pathlib import Path


def load_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".md", ".markdown", ".txt"):
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
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
