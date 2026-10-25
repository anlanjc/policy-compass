"""Function Calling 计算工具：LLM 提参数，公式硬编码。

设计原则（面试金句"确定性环节必须锁死"的落地）：
- 政策额度公式与限额表是确定性规则，绝不让 LLM"心算"——LLM 只负责从自然语言
  对话中抽取结构化参数（这正是它擅长的语义理解）；
- 参数抽取走 chat_json 自校验链路；工具执行是纯函数，可单测、可复算；
- 同一套工具通过 MCP server（mcp/server.py）暴露给外部 Agent 生态。

限额表/倍数与演示语料严格一致（data/corpus），答案可交叉验证。
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, Field

from ..llm.stats import stats

# ---------- 工具实现（纯函数，全部可单测） ----------

RENT_LIMITS: dict[str, dict[str, Any]] = {
    "北京": {"monthly": 2000, "quarterly": 6000, "yearly": 24000, "note": "按季度提取，租住公租房按实际租金不限额"},
    "上海": {"monthly": 3000, "quarterly": 9000, "yearly": 36000, "note": "支持按月或按季度划转，家庭双人合计可翻倍"},
    "杭州": {"monthly": 1500, "quarterly": 4500, "yearly": 18000, "note": "二孩家庭上浮至2250/月，三孩3000/月"},
    "深圳": {"monthly": 2600, "quarterly": 7800, "yearly": 31200, "note": "演示数字"},
    "广州": {"monthly": 1800, "quarterly": 5400, "yearly": 21600, "note": "演示数字"},
}

LOAN_RULES: dict[str, dict[str, Any]] = {
    "杭州": {"method": "余额倍数法", "multiplier": 15, "cap_single": 600000, "cap_family": 1200000, "income_ratio": 0.6},
    "上海": {"method": "余额倍数法", "multiplier": 10, "cap_single": 500000, "cap_family": 1000000, "income_ratio": 0.5},
    "北京": {"method": "缴存年限法", "per_year": 100000, "multiplier": 15, "cap_single": 1200000, "cap_family": 1200000, "income_ratio": 0.6},
}


def rent_extract_limit(city: str, is_family: bool = False) -> dict[str, Any]:
    table = RENT_LIMITS.get(city)
    if not table:
        return {"ok": False, "error": f"知识库暂未收录 {city} 的租赁提取限额，请以当地公积金中心公告为准"}
    limit = table["monthly"] * (2 if is_family else 1)
    yearly = table["yearly"] * (2 if is_family else 1)
    return {
        "ok": True,
        "city": city,
        "monthly_limit": limit,
        "yearly_limit": yearly,
        "note": table["note"] + ("；夫妻双方均缴存可分别提取" if is_family else ""),
    }


def _monthly_payment(principal: float, annual_rate: float, years: int) -> float:
    """等额本息月供（与 docs 公式一致），利率 2.85% 首套。"""
    r = annual_rate / 12
    n = years * 12
    if r == 0:
        return principal / n
    return principal * r * (1 + r) ** n / ((1 + r) ** n - 1)


def housing_loan_quota(
    city: str,
    balance: float | None = None,
    monthly_income: float | None = None,
    years: int | None = None,
    is_family: bool = False,
) -> dict[str, Any]:
    rule = LOAN_RULES.get(city)
    if not rule:
        return {"ok": False, "error": f"知识库暂未收录 {city} 的公积金贷款额度规则"}
    if balance is None or balance <= 0:
        return {"ok": False, "error": "缺少公积金账户余额参数，请补充（如：余额4万）后重试"}

    cap = rule["cap_family"] if is_family else rule["cap_single"]
    calc_detail: dict[str, Any] = {}

    if rule["method"] == "余额倍数法":
        by_balance = balance * rule["multiplier"]
        calc_detail["by_balance"] = f"{balance:.0f} × {rule['multiplier']} = {by_balance:.0f}"
    else:  # 北京：缴存年限法 + 余额倍数校核，取低
        effective_years = max(years or 1, 1)
        by_years = effective_years * rule["per_year"]
        by_balance = balance * rule["multiplier"]
        by_balance = min(by_balance, by_years, cap)
        calc_detail["by_years"] = f"缴存{effective_years}年 × {rule['per_year']:.0f} = {by_years:.0f}"
        calc_detail["by_balance"] = f"{balance:.0f} × {rule['multiplier']} = {balance * rule['multiplier']:.0f}"
        calc_detail["rule"] = "年限额度与余额倍数校核取低"

    quota = min(by_balance, cap) if rule["method"] == "余额倍数法" else by_balance
    calc_detail["cap"] = f"{'家庭' if is_family else '个人'}上限 {cap:.0f}"

    warn = ""
    if monthly_income and monthly_income > 0:
        pay = _monthly_payment(quota, 0.0285, 30)
        ratio = rule["income_ratio"]
        if pay > monthly_income * ratio:
            warn = (f"按30年等额本息月供约{pay:.0f}元，超过月收入{ratio:.0%}的约束"
                    f"（{monthly_income * ratio:.0f}元），实际可贷额度将下调")

    return {
        "ok": True,
        "city": city,
        "method": rule["method"],
        "quota": round(quota),
        "quota_wan": round(quota / 10000, 1),
        "detail": calc_detail,
        "income_warning": warn,
    }


TOOL_IMPLS = {
    "rent_extract_limit": rent_extract_limit,
    "housing_loan_quota": housing_loan_quota,
}

# ---------- OpenAI Function Calling 兼容 schema（展示/原生调用可复用） ----------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "rent_extract_limit",
            "description": "查询各城市住房公积金租赁提取的月度/年度限额",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "enum": list(RENT_LIMITS), "description": "缴存城市"},
                    "is_family": {"type": "boolean", "description": "夫妻双方均缴存，默认 false"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "housing_loan_quota",
            "description": "按城市规则测算住房公积金贷款额度（余额倍数法/缴存年限法）",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "enum": list(LOAN_RULES), "description": "贷款城市"},
                    "balance": {"type": "number", "description": "公积金账户余额（元）"},
                    "monthly_income": {"type": "number", "description": "月收入（元），用于还款能力校验"},
                    "years": {"type": "integer", "description": "已缴存年数"},
                    "is_family": {"type": "boolean", "description": "是否家庭（双人缴存）申请"},
                },
                "required": ["city", "balance"],
            },
        },
    },
]

TOOL_NAMES = list(TOOL_IMPLS)


class ToolCall(BaseModel):
    tool: str = Field(description="要调用的工具名")
    args: dict[str, Any] = Field(default_factory=dict, description="工具参数")


def execute_tool(call: ToolCall) -> dict[str, Any]:
    """执行工具并记录统计；未知工具/执行异常返回结构化错误（不抛出，链路不中断）。"""
    stats.incr("tool_calls")
    impl = TOOL_IMPLS.get(call.tool)
    if impl is None:
        return {"ok": False, "error": f"未知工具: {call.tool}"}
    try:
        return impl(**call.args)
    except TypeError as e:
        return {"ok": False, "error": f"参数不匹配: {e}"}


TOOL_PROMPT = (
    "#task=tool\n你是政策计算工具的选择器。根据用户问题选择一个工具并抽取参数：\n"
    f"可用工具：{TOOL_NAMES}\n"
    "- rent_extract_limit：查询租房提取限额（问题涉及 租房/房租/提取 标准时）\n"
    "- housing_loan_quota：测算贷款额度（问题涉及 贷款/能贷/额度 且含余额等数字时）\n"
    "数字换算：'4万'→40000，'两万'→20000，'5年'→years=5。\n"
    "问题中未出现的可选参数不要编造，余额等必填参数缺失时仍输出工具名并把 balance 置 null。"
    "若问题不需要任何计算工具，输出 {\"tool\": \"none\", \"args\": {}}。"
)


async def select_tool(llm, question: str) -> ToolCall:
    """LLM 语义抽取工具调用参数；失败或选择 none 时返回 tool='none'。"""
    from ..llm.client import LLMError, StructuredOutputError

    class MaybeCall(ToolCall):
        pass

    try:
        call = await llm.chat_json(
            [
                {"role": "system", "content": TOOL_PROMPT},
                {"role": "user", "content": question},
            ],
            MaybeCall,
        )
        if call.tool == "none" or call.tool not in TOOL_IMPLS:
            return ToolCall(tool="none", args={})
        return call
    except (StructuredOutputError, LLMError):
        return ToolCall(tool="none", args={})
