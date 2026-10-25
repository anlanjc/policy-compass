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
    if not meta.title:
        meta.title = file_name
    # 规范化
    if meta.city not in ("杭州", "上海", "北京", "深圳", "广州"):
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
