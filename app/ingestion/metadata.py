"""政策文档元数据抽取。

业务背景：同名政策各地版本不同、且会修订废止。检索若不带城市/领域/时效信号，
"杭州租房提取额度"很可能召回上海的文件或已废止的旧版——这是本系统区别于
通用 RAG demo 的核心业务约束之一。

元数据在摄取期一次性抽取（确定性规则，不用 LLM——规则能锁死的绝不交给模型），
存入 SQLite，检索期用于过滤与加权。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import date


@dataclass
class PolicyMeta:
    doc_id: str
    file_name: str
    title: str
    doc_no: str = ""  # 发文字号
    effective_date: str = ""  # 生效日期 YYYY-MM-DD
    status: str = "现行有效"  # 现行有效 / 已废止
    city: str = "全国"  # 适用城市
    domain: str = "通用"  # 公积金 / 社保 / 医保 / 通用
    file_hash: str = ""

    @property
    def is_valid(self) -> bool:
        return "废止" not in self.status


_META_PATTERNS = {
    "doc_no": re.compile(r"【发文字号】\s*(.+)"),
    "effective_date": re.compile(r"【生效日期】\s*(\d{4}-\d{2}-\d{2})"),
    "status": re.compile(r"【状态】\s*(.+)"),
    "city": re.compile(r"【适用城市】\s*(.+)"),
    "domain": re.compile(r"【领域】\s*(.+)"),
}

# ---- 脏文档兜底模式：真实公文（含 OCR 转录文本）没有元数据头，从正文抽取 ----
_DIRTY_DOC_NO = re.compile(r"[\u4e00-\u9fff]{2,12}〔\d{4}〕\d+号")

_HTML_DANGEROUS = re.compile(r'[<>&"\']')


def _strip_html_chars(s: str) -> str:
    """剥离 HTML 危险字符（标题/文号入库前的纵深防御，S3 评审 PC-6）。"""
    return _HTML_DANGEROUS.sub("", s)
# 施行/实施两种公文表述都接受（实测真实文件两种都在用）
_DIRTY_EFFECTIVE = re.compile(r"[自](\d{4})年(\d{1,2})月(\d{1,2})日起?(?:实施|施行)")
_KNOWN_CITIES = ("杭州", "上海", "北京", "深圳", "广州", "成都")
_DOMAIN_KEYWORDS = (("公积金", "公积金"), ("社保", "社保"), ("养老", "社保"),
                    ("医保", "医保"), ("医疗", "医保"))


def _fallback_extract(meta: PolicyMeta, content: str) -> None:
    """无元数据头时的正文抽取（扫描件 OCR 文本的典型形态）。"""
    head = content[:600]
    if not meta.doc_no:
        m = _DIRTY_DOC_NO.search(content)
        if m:
            meta.doc_no = m.group(0)
    if not meta.effective_date:
        m = _DIRTY_EFFECTIVE.search(content)
        if m:
            meta.effective_date = "%04d-%02d-%02d" % tuple(
                int(g) for g in m.groups()
            )
    if not meta.title:
        # 红头标题通常在前几行：取首个含"通知/办法/细则/规定/规则/条件"的行
        for line in head.splitlines()[:6]:
            line = line.strip()
            if len(line) >= 8 and re.search(r"通知|办法|细则|规定|规则|条件|指引|规程|说明", line):
                meta.title = line
                break
    if meta.city == "全国" and not content.startswith("【"):
        for c in _KNOWN_CITIES:
            if c in head:
                meta.city = c
                break
    if meta.domain == "通用" and not content.startswith("【"):
        for kw, dom in _DOMAIN_KEYWORDS:
            if kw in head:
                meta.domain = dom
                break


def extract_metadata(file_name: str, content: str) -> PolicyMeta:
    meta = PolicyMeta(
        doc_id=file_name,
        file_name=file_name,
        title="",
        file_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )
    for line in content.splitlines():
        line = line.strip()
        if not meta.title and line.startswith("# "):
            meta.title = line[2:].strip()
        for key, pat in _META_PATTERNS.items():
            m = pat.match(line)
            if m:
                setattr(meta, key, m.group(1).strip())
    if not content.startswith("【"):  # 脏文档（无元数据头，如 OCR 转录的公文）
        _fallback_extract(meta, content)
    if not meta.title:
        meta.title = file_name
    # 纵深防御（S3 评审 PC-6）：title/doc_no 来自上传内容首行，会被前端引用卡渲染——
    # 剥离 HTML 危险字符，`# <img onerror=...>` 之类的投毒行不得原样入库
    meta.title = _strip_html_chars(meta.title)
    if meta.doc_no:
        meta.doc_no = _strip_html_chars(meta.doc_no)
    # 规范化
    if meta.city not in ("杭州", "上海", "北京", "深圳", "广州", "成都"):
        meta.city = "全国" if meta.city in ("全国", "") else meta.city
    if meta.domain not in ("公积金", "社保", "医保", "通用"):
        meta.domain = "通用"
    return meta


def date_rank(d: str) -> float:
    """生效日期转可比数值（用于新旧政策排序），非法日期返回 0。"""
    try:
        return date.fromisoformat(d).toordinal() * 1.0
    except (ValueError, TypeError):
        return 0.0
