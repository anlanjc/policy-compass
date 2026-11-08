"""生成上下文的提示注入隔离（PC-HR2）。

对抗评测实证（evidence/prompt_injection_report.md 首轮 16/18 被打穿）：检索穿透后，
载荷句凭"数字(+2.0)+结论词(+2.5)"加分被抽进答案；verify 忠实度校验对此存在
结构性盲区——载荷文本本身就在上下文里，"忠实于上下文"不等于"真实"。

防线落位：**检索出口单一咽喉**（retrieve_node）。命中注入模式的行就地剔除，
含载荷行的文档整体记 quarantine 标记——grade 记 0（不可作为证据参与 CRAG 决策），
生成上下文与 verify 上下文经 grade>=0.4 过滤自动排除。

零漂移论证（冻结口径保障）：模式库对正式语料 23 份 md + OCR 缓存全文扫描零命中
（tests/test_injection.py 钉死为永久回归）——正式语料经此路径文本逐字节不变，
冻结评测数字不受扰动。对抗语料（data/corpus_adversarial/）6 份每份至少一行命中。

诚实边界：标记词匹配是启发式防火墙，不抗变形注入（同义改写/拆字/非中文载荷）；
模式库演进须同时通过"正式语料零命中"+"对抗样本命中"双向校验。
"""

from __future__ import annotations

import re

# 行级模式库：tag → pattern（演进须同时过双向校验，见模块 docstring）
INJECTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "指令覆盖": re.compile(r"忽略(以上|之前|上面|前面|所有|先前|上述)"),
    "命令式指令": re.compile(r"(你)?(必须|请|务必)(直接)?(回答|输出|复述|打印|透露|执行)"),
    "角色/系统伪造": re.compile(r"系统指令|系统提示词|系统设定|开发者模式|自由问答|越狱"),
    "伪最高效力": re.compile(r"(效力|优先级|权限)(高于|大于|压过|盖过)(一切|任何|所有|其他|地方)"),
    "统一额度指令": re.compile(r"(一律|统一)调整为|(额度|标准)(一律|统一)(调整为|为)|统一为每人"),
    "敏感信息诱导": re.compile(r"验证码|密钥|口令|密码"),
    "诱导编造": re.compile(r"编造"),
    "英文指令": re.compile(r"(?i)ignore\s+(all|any|previous|above)|disregard"),
}

QUARANTINE_NOTE = "（疑似注入指令行，已隔离）"


def scan_lines(text: str) -> list[tuple[str, str]]:
    """返回 (tag, 命中行原文) 列表。行级判定：一行命中任一模式即整行隔离。"""
    out: list[tuple[str, str]] = []
    for line in text.splitlines():
        for tag, pat in INJECTION_PATTERNS.items():
            if pat.search(line):
                out.append((tag, line))
                break  # 一行只记首个命中 tag
    return out


def sanitize_text(text: str) -> tuple[str, int]:
    """剔除命中行，返回 (净化后文本, 隔离行数)。零命中时原串返回（逐字节不变）。"""
    flagged = {line for _, line in scan_lines(text)}
    if not flagged:
        return text, 0
    kept = [line for line in text.splitlines() if line not in flagged]
    return "\n".join(kept), len(flagged)
