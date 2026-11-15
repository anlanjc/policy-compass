

async def test_settings_env_aliases_and_force_mock(monkeypatch):
    """配置别名双收 + force_mock 真生效（演示车道 2026-11-09 实测发现的死变量三连）。"""
    from app.config import Settings

    monkeypatch.setenv("POLICY_COMPASS_ADMIN_KEY", "doc-name-key")
    s = Settings(llm_mode="mock")
    assert s.admin_key == "doc-name-key"  # DEPLOY.md 文档名复活

    monkeypatch.setenv("ADMIN_KEY", "bare-name-key")
    s = Settings(llm_mode="mock")
    assert s.admin_key == "bare-name-key"  # 裸名仍可用

    monkeypatch.setenv("POLICY_COMPASS_CORS_ORIGINS", '["https://a.example"]')
    s = Settings(llm_mode="mock")
    assert s.cors_origins == ["https://a.example"]

    monkeypatch.setenv("POLICY_COMPASS_FORCE_MOCK", "1")
    monkeypatch.setenv("LLM_API_KEY", "sk-should-be-ignored")
    s = Settings(llm_mode="auto")
    assert s.force_mock is True
    assert s.effective_llm_mode == "mock"  # 有 key 也强制 mock（CI 语义落地）


def test_doc_test_count_matches_suite():
    """文档测试数==实际测试函数数（POLICY-8 防再漂移守门，2026-11-16）。

    前轮数字回填不彻底：同一时点文档漂移出 23/29/57 三个口径且均与实际不符、
    EVIDENCE 自相矛盾——架空「所有数字可复现」总声明。本测试把 README 徽章数
    钉死为 tests/ 下测试函数实际计数（与评审复核同法：^def test_ 逐文件统计）。
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    n_tests = 0
    for p in sorted((root / "tests").glob("test_*.py")):
        n_tests += len(
            re.findall(r"^\s*(?:async )?def test_\w+", p.read_text(encoding="utf-8"), re.M)
        )
    assert n_tests >= 60, f"测试数意外减少: {n_tests}，检查 tests/"
    readme = (root / "README.md").read_text(encoding="utf-8")
    m = re.search(r"badge/pytest-(\d+)%20项-brightgreen", readme)
    assert m, "README pytest 徽章缺失或格式变更"
    assert int(m.group(1)) == n_tests, (
        f"README 徽章 {m.group(1)} ≠ 实际测试函数数 {n_tests}——文档数字再漂移"
    )
