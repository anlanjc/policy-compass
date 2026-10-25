"""意图路由：LLM 结构化输出 + 规则兜底。

为什么路由必须存在（业务约束）：同名政策各地标准不同、且政策语言与用户口语
差异大——先识别"城市 + 领域 + 是否需要计算"，检索才能限定正确子空间。

可靠性：LLM 路由失败（结构化输出不可恢复）时降级为正则规则路由（确定性），
路由错误率因此有界，这是"LLM 推理 vs 硬编码锁死"边界的典型取舍。
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from ..config import settings
from ..llm.client import LLMError, StructuredOutputError

logger = logging.getLogger(__name__)


class RouteResult(BaseModel):
    city: str = Field(default="全国", description="问题适用的城市")
    domain: str = Field(default="通用", description="公积金/社保/医保/通用")
    need_calc: bool = Field(default=False, description="是否需要额度类数字计算")
    sub_questions: list[str] = Field(default_factory=list, description="对比型问题拆出的子问题")

    def validated(self) -> "RouteResult":
        if self.city not in settings.cities:
            self.city = "全国"
        if self.domain not in settings.domains:
            self.domain = "通用"
        return self


CITY_ALIASES = {"沪": "上海", "京": "北京", "杭": "杭州", "穗": "广州", "深": "深圳"}
DOMAIN_KEYWORDS = {
    "公积金": ["公积金", "房贷", "贷款额度", "租房提取", "缴存"],
    "社保": ["社保", "五险", "养老", "断缴", "补缴", "落户", "居转户", "缴费基数"],
    "医保": ["医保", "医疗", "报销", "大病"],
}
CALC_PAT = ("余额", "月薪", "缴了", "缴存", "能贷", "贷多少", "额度", "多少",
            "几万", "月供")


def rule_based_route(question: str) -> RouteResult:
    """确定性规则路由：LLM 不可用时的兜底，也是路由正确性的对照组。"""
    hits = [name for name in settings.cities if name != "全国" and name in question]
    city = hits[0] if len(hits) == 1 else "全国"  # 多城对比/零城 → 全国（不限定）
    if not hits:
        for alias, full in CITY_ALIASES.items():
            if alias in question:
                city = full
                break
    domain = "通用"
    for dom, words in DOMAIN_KEYWORDS.items():
        if any(w in question for w in words):
            domain = dom
            break
    need_calc = any(w in question for w in CALC_PAT) and any(
        ch.isdigit() for ch in question
    )
    sub_questions: list[str] = []
    if any(w in question for w in ("和", "对比", "区别", "哪个")):
        sub_questions = [question]
    return RouteResult(city=city, domain=domain, need_calc=need_calc,
                       sub_questions=sub_questions).validated()


async def route_query(llm, question: str, history: list[dict]) -> RouteResult:
    """优先 LLM 路由，失败降级规则路由。"""
    history_hint = ""
    if history:
        last_user = next(
            (m["content"] for m in reversed(history) if m["role"] == "user"), ""
        )
        if last_user:
            history_hint = f"\n（对话上下文提示：上一问是「{last_user[:50]}」，若本轮问题中的代词指代它，请结合理解）"
    try:
        result = await llm.chat_json(
            [
                {
                    "role": "system",
                    "content": f"#task=route\n你是政务政策问答系统的意图路由器。根据用户问题判断：\n"
                    f"city：问题适用于哪个城市（{'/'.join(settings.cities)}，未提及则\"全国\"；"
                    f"若问题对比多个城市，取\"全国\"并在 sub_questions 里拆出各城市子问题）；\n"
                    f"domain：属于公积金/社保/医保/通用 哪个领域；\n"
                    f"need_calc：回答是否需要基于数字的额度计算（如贷款额度测算）；\n"
                    f"sub_questions：对比型问题拆出的子问题列表，无则空数组。{history_hint}",
                },
                {"role": "user", "content": question},
            ],
            RouteResult,
        )
        return result.validated()
    except (StructuredOutputError, LLMError) as e:
        logger.warning("LLM 路由失败，降级规则路由: %s", e)
        return rule_based_route(question)
