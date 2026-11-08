

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
