"""工具层（Function Calling）与 Agentic 编排（LangGraph）测试。"""

from app.agent.tools import ToolCall, execute_tool, housing_loan_quota, rent_extract_limit


def test_hangzhou_loan_balance_multiplier():
    """杭州余额倍数法：4万×15=60万，未超个人上限60万。"""
    r = housing_loan_quota(city="杭州", balance=40000, monthly_income=20000, years=5)
    assert r["ok"] and r["quota"] == 600000


def test_shanghai_family_cap():
    """上海家庭：12万×10=120万 > 家庭上限100万 → 取上限。"""
    r = housing_loan_quota(city="上海", balance=120000, is_family=True)
    assert r["ok"] and r["quota"] == 1000000


def test_beijing_years_method():
    """北京年限法：8年×10万=80万；余额9万×15=135万校核；取低=80万。"""
    r = housing_loan_quota(city="北京", balance=90000, years=8)
    assert r["ok"] and r["quota"] == 800000


def test_income_constraint_warning():
    r = housing_loan_quota(city="杭州", balance=200000, monthly_income=3000)
    assert r["ok"] and r["income_warning"]  # 月供超收入60%触发提示


def test_missing_balance_returns_structured_error():
    r = housing_loan_quota(city="杭州", balance=None)
    assert not r["ok"] and "余额" in r["error"]


def test_rent_limit_table():
    assert rent_extract_limit("北京")["monthly_limit"] == 2000
    assert rent_extract_limit("上海", is_family=True)["monthly_limit"] == 6000
    assert not rent_extract_limit("火星")["ok"]


def test_execute_tool_unknown():
    r = execute_tool(ToolCall(tool="not_exist", args={}))
    assert not r["ok"]


async def test_agent_calc_path(components):
    """计算题：路由→检索→评分→工具→生成→校验，工具结果数字正确。"""
    agent = components["agent"]
    state = await agent.run("我杭州公积金余额4万缴了5年月薪2万，最多能贷多少？")
    events = [e["event"] for e in state["events"]]
    assert events[0] == "route"
    assert "retrieve" in events and "grade" in events
    assert "tool" in events
    tool = state["tool_result"]
    assert tool["tool"] == "housing_loan_quota"
    assert tool["result"]["quota_wan"] == 60.0
    assert not state["refusal"]


async def test_agent_refuse_path(components):
    """库外题：改写循环耗尽后诚实拒答，绝不编造。"""
    agent = components["agent"]
    state = await agent.run("护照到期了怎么换证")
    assert state["refusal"]
    assert "fallback" in [e["event"] for e in state["events"]]
    assert "12329" in state["answer"] or "12345" in state["answer"]


async def test_agent_rewrite_loop_bounded(components):
    """改写循环有界：不超过 max_query_rewrites 轮，无死循环。"""
    from app.config import settings

    agent = components["agent"]
    state = await agent.run("拍车牌摇号政策是什么")
    rewrites = [e for e in state["events"] if e["event"] == "rewrite"]
    assert len(rewrites) <= settings.max_query_rewrites
    assert "retrieve" in [e["event"] for e in state["events"]]


async def test_agent_rule_router_fallback(components):
    """规则路由：多城对比 → 全国；单城 → 该城。"""
    from app.agent.router import rule_based_route

    assert rule_based_route("北京和上海租房提取额度").city == "全国"
    assert rule_based_route("杭州公积金怎么提取").city == "杭州"
    assert rule_based_route("公积金贷款利率").city == "全国"
