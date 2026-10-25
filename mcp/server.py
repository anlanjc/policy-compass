"""MCP Server：把政策计算工具暴露给任意 MCP 客户端（Claude Desktop 等）。

为什么值得做：MCP 是工具生态的"USB-C"。同一个计算能力在本项目内服务
LangGraph Agent（进程内调用），同时经 MCP 服务外部生态——工具一次实现，
两处分发，这正是 MCP 的设计初衷。

运行（需可选安装官方 SDK：pip install "mcp[cli]"）：
    python mcp/server.py          # stdio 传输，供 Claude Desktop 等接入
配置示例（claude_desktop_config.json）:
    {"mcpServers": {"policy-tools": {"command": "python", "args": ["<绝对路径>/mcp/server.py"]}}}
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP

from app.agent.tools import (
    LOAN_RULES,
    RENT_LIMITS,
    housing_loan_quota,
    rent_extract_limit,
)

mcp = FastMCP("policy-compass-tools")


@mcp.tool()
def rent_extract_limit_tool(city: str, is_family: bool = False) -> dict:
    """查询各城市住房公积金租赁提取的月度/年度限额。

    Args:
        city: 缴存城市（北京/上海/杭州/深圳/广州）
        is_family: 夫妻双方均缴存，默认 False
    """
    return rent_extract_limit(city, is_family)


@mcp.tool()
def housing_loan_quota_tool(
    city: str,
    balance: float,
    monthly_income: float | None = None,
    years: int | None = None,
    is_family: bool = False,
) -> dict:
    """按城市规则测算住房公积金贷款额度（杭州/上海余额倍数法，北京缴存年限法）。

    Args:
        city: 贷款城市（杭州/上海/北京）
        balance: 公积金账户余额（元）
        monthly_income: 月收入（元），用于还款能力校验
        years: 已缴存年数（北京必需）
        is_family: 是否家庭（双人缴存）申请
    """
    return housing_loan_quota(city, balance, monthly_income, years, is_family)


@mcp.tool()
def supported_cities() -> dict:
    """列出当前支持额度计算的城市及其规则概要。"""
    return {
        "rent_extract": {c: f"{v['monthly']}元/月（{v['note']}）" for c, v in RENT_LIMITS.items()},
        "loan_quota": {
            c: f"{v['method']}，上限{v['cap_single'] // 10000}万(个人)/{v['cap_family'] // 10000}万(家庭)"
            for c, v in LOAN_RULES.items()
        },
    }


if __name__ == "__main__":
    mcp.run()  # stdio
