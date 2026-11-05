"""Function Calling 计算工具：LLM 提参数，公式硬编码。

设计原则（面试金句"确定性环节必须锁死"的落地）：
- 政策额度公式与限额表是确定性规则，绝不让 LLM"心算"——LLM 只负责从自然语言
  对话中抽取结构化参数（这正是它擅长的语义理解）；
- 参数抽取走 chat_json 自校验链路；工具执行是纯函数，可单测、可复算；
- 同一套工具通过 MCP server（mcp/server.py）暴露给外部 Agent 生态。

限额表/倍数与真实语料一致（data/corpus 同名文件，来源见各条注释），答案可交叉验证。
"""

from __future__ import annotations

import json
import math
from typing import Any

from pydantic import BaseModel, Field

from ..llm.stats import stats

# ---------- 工具实现（纯函数，全部可单测） ----------

# 租赁提取限额：数字全部取自 data/corpus 真实政策原文（来源 URL 见各语料文件头部）。
# 深圳 2026-10 起无固定月限额——现行《深圳市住房公积金提取管理规定》（深公积金规〔2025〕3号）
# 第十一条按"申请当月应缴存额的 65%"比例计算，无法折算为固定额度，故不收录（宁缺毋滥，
# 工具对深圳返回"未收录"并提示咨询公积金中心，语料中的比例规则仍可经检索回答）。
RENT_LIMITS: dict[str, dict[str, Any]] = {
    "北京": {"monthly": 2000, "quarterly": 6000, "yearly": 24000,
             "note": "京房公积金发〔2023〕6号：无发票租房提取每人每月2000元；提供发票并备案的按实际月租金提取"},
    # 上海为"户"口径（含单身家庭），夫妻不翻倍 → family_multiplier=1
    "上海": {"monthly": 4000, "quarterly": 12000, "yearly": 48000, "family_multiplier": 1,
             "note": "沪公积金管委会〔2024〕6号：市场租赁每户家庭（含单身）月提取限额4000元；新市民/青年人网签备案可按实际房租提取"},
    "杭州": {"monthly": 2000, "quarterly": 6000, "yearly": 24000,
             "note": "杭房公委〔2023〕8号：市区2000元/月，桐庐淳安建德1200元/月；新市民/青年人可全额提取月缴存额；多孩上浮50%（杭房公委〔2025〕4号）"},
    "广州": {"monthly": 2000, "quarterly": 6000, "yearly": 24000,
             "note": "穗公积金中心规字〔2024〕2号：无租赁合同或未登记备案的租房提取限额提高至每人每月2000元"},
    "成都": {"monthly": 1800, "quarterly": 5400, "yearly": 21600,
             "note": "成都公积金委2026年调整通知（据人民网/成都日报转载）：租住商品住房每月不超过1800元、年累计2.16万元；多子女家庭上浮50%（每月不超过2700元）"},
}

# 贷款额度规则：数字全部取自真实政策原文/权威转载（2026-10-29 抓取，出处见注释与 data/corpus 同名文件）。
# 原文没有的字段宁缺毋滥：沪杭现行文件均未载收入比条款，不再沿用演示口径的 0.6/0.5。
LOAN_RULES: dict[str, dict[str, Any]] = {
    # 杭房公委〔2026〕1号（2026-04-01 实施，杭州市政府公报刊载）：倍数 15→20，
    # 贷款最高额度 130→180 万、职工个人可贷额度 ≤90 万。官方口径为"申请贷款时近12个月
    # （不含申请当月）账户月均余额×倍数"（不含一次性补缴、不足12个月按实际月数、
    # 结果四舍五入精确到千位）——本工具按传入的当前余额近似并在 note 披露；新市民/青年+20%、
    # 多子女/高层次人才+50%、绿建/以旧换新+20%，择高叠加最高+70%（见语料）。
    "杭州": {"method": "余额倍数法", "multiplier": 20, "round_to": 1000,
             "cap_single": 900000, "cap_family": 1800000,
             "note": "杭州按申请贷款时近12个月账户月均余额×20倍测算（本工具按当前账户余额近似），"
                     "家庭最高180万元、个人不超过90万元；新市民/青年人+20%、多子女/高层次人才+50%等可上浮"},
    # 沪公积金管委会〔2026〕1号（2026-02-26 实施，两源交叉转载）：首套基本额度个人最高100万/
    # 家庭200万，缴补充公积金再+20/+40万（含补充合计120/240万）；二套80/160（+20/+40）。
    # 通知未载余额倍数/收入比条款（官网 shgjj.com 抓取不可达），按最高限额口径计算，不再沿用演示口径 ×10；
    # 多子女/绿色建筑最高上浮35%（首套含补充最高324万）。
    "上海": {"method": "固定额度法", "cap_single": 1000000, "cap_family": 2000000,
             "note": "上海按首套最高限额口径：个人最高100万元/家庭最高200万元（缴交补充公积金另+20/+40万元；"
                     "沪公积金管委会〔2026〕1号），实际额度还受余额倍数、房价成数、还款能力等约束"},
    # 京房公积金〔2018〕51号：每缴存一年可贷10万（不满一年按一年），最高120万，月供≤月收入60%
    "北京": {"method": "缴存年限法", "per_year": 100000, "cap_single": 1200000, "cap_family": 1200000, "income_ratio": 0.6},
    # 穗公积金中心规字〔2024〕2号：一人申请最高80万，两人及以上共同申请160万（原文无余额倍数公式）
    "广州": {"method": "固定额度法", "cap_single": 800000, "cap_family": 1600000},
}


def rent_extract_limit(city: str, is_family: bool = False) -> dict[str, Any]:
    table = RENT_LIMITS.get(city)
    if not table:
        return {"ok": False, "error": f"知识库暂未收录 {city} 的租赁提取限额，请以当地公积金中心公告为准"}
    fam = table.get("family_multiplier", 2)
    limit = table["monthly"] * (fam if is_family else 1)
    yearly = table["yearly"] * (fam if is_family else 1)
    return {
        "ok": True,
        "city": city,
        "monthly_limit": limit,
        "yearly_limit": yearly,
        "note": table["note"] + ("；夫妻双方均缴存可分别提取" if is_family and fam == 2 else ""),
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
        if rule.get("round_to"):
            # 杭州"计算结果四舍五入，精确到千位"（杭房公委〔2026〕1号配套细则）
            by_balance = round(by_balance / rule["round_to"]) * rule["round_to"]
        calc_detail["by_balance"] = f"{balance:.0f} × {rule['multiplier']} = {by_balance:.0f}"
        quota = min(by_balance, cap)
    elif rule["method"] == "缴存年限法":
        # 京房公积金〔2018〕51号：额度=缴存年限×10万（不满一年按一年），上限120万，
        # 已婚按夫妻缴存年限较长一方核算——原文无余额倍数校核，不再使用演示口径的 ×15
        effective_years = max(years or 1, 1)
        by_years = effective_years * rule["per_year"]
        quota = min(by_years, cap)
        calc_detail["by_years"] = f"缴存{effective_years}年 × {rule['per_year']:.0f} = {by_years:.0f}"
    else:  # 固定额度法（广州/上海）：按申请情形的固定上限，原文无额度公式
        quota = cap
    calc_detail["cap"] = f"{'家庭' if is_family else '个人'}上限 {cap:.0f}"

    warn = ""
    ratio = rule.get("income_ratio", 0)
    if monthly_income and monthly_income > 0 and ratio > 0:
        pay = _monthly_payment(quota, 0.0285, 30)
        if pay > monthly_income * ratio:
            warn = (f"按30年等额本息月供约{pay:.0f}元，超过月收入{ratio:.0%}的约束"
                    f"（{monthly_income * ratio:.0f}元），实际可贷额度将下调")

    result = {
        "ok": True,
        "city": city,
        "method": rule["method"],
        "quota": round(quota),
        "quota_wan": round(quota / 10000, 1),
        "detail": calc_detail,
        "income_warning": warn,
    }
    if rule.get("note"):
        result["note"] = rule["note"]
    return result


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
            "description": "按城市规则测算住房公积金贷款额度（余额倍数法/缴存年限法/固定额度法）",
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


def _annotation_base(ann: Any) -> type | None:
    """解出注解的基底真实类型：兼容 `from __future__ import annotations` 下的
    `'float | None'` 字符串注解与 `X | None` Union 形态（S3 评审 PC-4）。

    此前 `ann is int`/`ann is bool` 对字符串注解恒为 False，years/is_family 的
    防御转换是死代码——is_family 传 'false' 被当真值，家庭口径静默翻倍额度。
    """
    import inspect
    import typing
    import types as _types

    if ann is None or ann is inspect.Parameter.empty:
        return None
    origin = typing.get_origin(ann)
    if origin is typing.Union or origin is _types.UnionType:
        non_none = [a for a in typing.get_args(ann) if a is not type(None)]
        if len(non_none) == 1:
            ann = non_none[0]
    return ann if isinstance(ann, type) else None


def _coerce_args(impl, args: dict[str, Any]) -> dict[str, Any]:
    """按工具签名防御性转换模型传参。

    真实模型（实测 deepseek-v4）常见三类偏差：数字传字符串（"40000"）、
    自创参数别名（salary）、多传不存在的键。此处依次处理：
    别名归一 → 按类型注解转换 → 丢弃未知键；转换失败保留原值让工具自身校验兜底。
    """
    import inspect
    import typing

    args = {ARG_ALIASES.get(k, k): v for k, v in args.items()}
    sig = inspect.signature(impl)
    # 运行时注解因 __future__ annotations 全为字符串，必须经 get_type_hints 解析
    try:
        hints = typing.get_type_hints(impl)
    except Exception:
        hints = {}
    coerced: dict[str, Any] = {}
    for k, v in args.items():
        param = sig.parameters.get(k)
        if param is None:
            continue  # 模型幻觉出的参数直接丢弃
        ann = _annotation_base(hints.get(k, param.annotation))
        try:
            if ann is float and not isinstance(v, (int, float)):
                v = float(str(v).replace("万", "e4").replace("，", "").strip() or 0)
            elif ann is int and not isinstance(v, int):
                v = int(float(v))
            elif ann is bool and isinstance(v, str):
                v = v.strip().lower() in ("true", "1", "yes", "是")
        except (TypeError, ValueError):
            pass  # 转换失败原样传，由工具内的参数校验返回结构化错误
        coerced[k] = v
    return coerced


def execute_tool(call: ToolCall) -> dict[str, Any]:
    """执行工具并记录统计；未知工具/执行异常返回结构化错误（不抛出，链路不中断）。"""
    stats.incr("tool_calls")
    impl = TOOL_IMPLS.get(call.tool)
    if impl is None:
        return {"ok": False, "error": f"未知工具: {call.tool}"}
    try:
        return impl(**_coerce_args(impl, call.args))
    except TypeError as e:
        return {"ok": False, "error": f"参数不匹配: {e}"}


TOOL_PROMPT = (
    "#task=tool\n你是政策计算工具的选择器。根据用户问题选择一个工具并抽取参数。\n"
    "可用工具及参数 schema（参数名必须完全一致，不得自创别名如 salary）：\n"
    f"{json.dumps(TOOL_SCHEMAS, ensure_ascii=False, indent=1)}\n"
    "规则：\n"
    "- city 必填（用户未明说时根据语境推断，如'我杭州…'→杭州）；\n"
    "- 用户问题中未出现任何支持城市且语境无城市线索时，输出 {\"tool\": \"none\", \"args\": {}}，"
    "绝不允许猜测城市——把 A 城额度当作 B 城答案等于编造；\n"
    "- balance/monthly_income 单位为元：'4万'→40000，'2万'→20000；\n"
    "- 未出现的可选参数不要编造；不需要计算工具时输出 {\"tool\": \"none\", \"args\": {}}。"
)

# 常见参数别名归一（模型自创同义词的防御）
ARG_ALIASES = {
    "salary": "monthly_income",
    "income": "monthly_income",
    "month_income": "monthly_income",
    "account_balance": "balance",
    "money": "balance",
    "is_married": "is_family",
    "family": "is_family",
}


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
