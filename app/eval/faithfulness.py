"""引用 faithfulness 自动校验器（PC-HR3）。

与 verify 节点（句子级 bigram 重叠）互补的**实体级**校验：对答案中每个
文号/条款号/数量原子，回查其引用 chunk 原文 + PC-3 chunk 级条款元数据（sections）
+ 引用文号元数据 + 工具计算结果——不一致打标并归因：
- 生成幻觉：全库任何 chunk 都找不到该原子；
- 引用错块：原子在库内其他 chunk 存在、但不在本次引用集；
- 校验器误报：20 条子集人工校准识别后由排除规则吸收（如年份/热线/日期）。

纯离线评测装置：只读答案与索引，不改变任何运行时行为（冻结口径零风险）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# 文号：与 ingestion/metadata 的 _DIRTY_DOC_NO 同形（〔2026〕1号）
RE_DOC_NO = re.compile(r"[\u4e00-\u9fff]{2,12}〔(\d{4})〕(\d+)号")
# 条款号：第X条（中文或阿拉伯数字）
RE_ARTICLE = re.compile(r"第([一二两三四五六七八九十百零\d]+)条")
# 数量原子：数字+政策单位（元/万/倍/%/期限）
RE_AMOUNT = re.compile(r"(\d+(?:\.\d+)?)(万元|个月|个工作日|万|元|%|倍|年)")

_CN_DIGITS = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def cn_to_int(s: str) -> int | None:
    """中文数字 → int（支持 一~九百九十九 与纯数字）；解析失败返回 None。"""
    if s.isdigit():
        return int(s)
    total, num = 0, 0
    for ch in s:
        if ch == "十":
            num = (num or 1) * 10
            total += num
            num = 0
        elif ch == "百":
            num = (num or 1) * 100
            total += num
            num = 0
        elif ch in _CN_DIGITS:
            num = num * 10 + _CN_DIGITS[ch]
        elif ch == "零":
            continue
        else:
            return None
    return total + num


@dataclass
class Atom:
    kind: str  # doc_no / article / amount
    text: str  # 答案中的原文形态
    article_no: int | None = None  # article 原子的归一化条款号


@dataclass
class FaithfulnessResult:
    total: int = 0
    supported: int = 0
    unsupported: list[dict] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.supported / self.total if self.total else 1.0


def extract_atoms(answer: str) -> list[Atom]:
    """提取实体原子。文号先抽取并掩码（防〔2026〕里的年份被当数量）。"""
    atoms: list[Atom] = []
    masked = answer
    for m in RE_DOC_NO.finditer(answer):
        atoms.append(Atom("doc_no", m.group(0)))
    masked = RE_DOC_NO.sub("□", answer)
    for m in RE_ARTICLE.finditer(masked):
        n = cn_to_int(m.group(1))
        if n:
            atoms.append(Atom("article", m.group(0), article_no=n))
    for m in RE_AMOUNT.finditer(masked):
        val, unit = m.group(1), m.group(2)
        if unit == "年" and 1900 <= float(val) <= 2100:
            continue  # 公历年份（生效日期类）不作为数量原子校验
        atoms.append(Atom("amount", m.group(0)))
    return atoms


def _amount_forms(atom_text: str) -> list[str]:
    """同一数量的合法书写形态：90万元 ↔ 90万；其余原样。"""
    forms = {atom_text}
    if atom_text.endswith("万元"):
        forms.add(atom_text[:-2] + "万")
    elif atom_text.endswith("万"):
        forms.add(atom_text[:-1] + "万元")
    return sorted(forms)


def _evidence_has_amount(evidence: str, atom_text: str) -> bool:
    for form in _amount_forms(atom_text):
        num, unit = form[:-len(_unit_of(form))], _unit_of(form)
        if re.search(rf"(?<!\d){re.escape(num)}{re.escape(unit)}", evidence):
            return True
    return False


def _unit_of(form: str) -> str:
    for u in ("万元", "个月", "个工作日", "万", "元", "%", "倍", "年"):
        if form.endswith(u):
            return u
    return ""


def check_answer(
    answer: str,
    sources: list[dict],
    texts_by_chunk: dict[str, str],
    tool_evidence: str = "",
    all_texts: dict[str, str] | None = None,
) -> FaithfulnessResult:
    """校验单条答案。

    - 证据集 = 引用 chunk 原文 + 引用 source 的 doc_no/sections 元数据 + 工具结果文本；
    - all_texts（全库文本）仅用于归因：原子在库内其他块存在 → "引用错块"，
      全库未见 → "生成幻觉或校验器误报"（由人工校准裁决）。
    """
    cited_texts = [
        texts_by_chunk[s["chunk_id"]]
        for s in sources
        if s.get("chunk_id") and s["chunk_id"] in texts_by_chunk
    ]
    evidence = "\n".join(cited_texts) + "\n" + tool_evidence
    doc_nos = [s.get("doc_no") or "" for s in sources]
    # sections 元数据存整串标签（"第七条"），剥"第/条"取数字归一（PC-3 联动）
    cited_sections = {
        n for s in sources for a in (s.get("sections") or [])
        if (n := cn_to_int(re.sub(r"^第|条$", "", a))) is not None
    }

    atoms = extract_atoms(answer)
    result = FaithfulnessResult(total=len(atoms))
    corpus_evidence = "\n".join(all_texts.values()) if all_texts else None
    for a in atoms:
        ok = False
        if a.kind == "doc_no":
            # 答案语流会带前缀（"依据杭房公委〔2025〕4号"）——支持判定用
            # "末两汉字+〔年〕号"后缀针（机关名末二字+年号+序号同一即同文号）
            m = re.search(r"([\u4e00-\u9fff]{2}〔\d{4}〕\d+号)$", a.text)
            needle = a.text if m is None else m.group(1)
            ok = needle in evidence or any(needle in d for d in doc_nos)
        elif a.kind == "article":
            forms = {a.text, f"第{a.article_no}条"}
            ok = a.article_no in cited_sections or any(f in evidence for f in forms)
        else:  # amount
            ok = _evidence_has_amount(evidence, a.text)
        if ok:
            result.supported += 1
        else:
            attribution = "生成幻觉或校验器误报（全库未见）"
            if corpus_evidence is not None:
                found = (
                    a.text in corpus_evidence
                    or (a.kind == "amount" and _evidence_has_amount(corpus_evidence, a.text))
                    or (a.kind == "article" and a.text in corpus_evidence)
                )
                if found:
                    attribution = "引用错块（事实在库、未在引用集）"
            result.unsupported.append(
                {"kind": a.kind, "atom": a.text, "attribution": attribution}
            )
    return result
