"""全局配置：pydantic-settings + .env。

设计原则：
- 检索/编排参数集中管理，调参实验只改一处，配合 eval 一键对比。
- 无 API key 自动降级 mock 模式，clone 后零配置可跑通全链路（CI 同样可跑）。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------- LLM（OpenAI 兼容协议：DeepSeek / GLM / 硅基流动均可） ----------
    llm_mode: str = "auto"  # auto：有 key 用 real；可强制 mock / real
    llm_api_base: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    llm_timeout: float = 60.0
    llm_max_retries: int = 3  # 网络/限流错误的指数退避重试
    llm_trust_env: bool = False  # 不读系统/环境代理：Windows 注册表代理会让 dashscope TLS 握手失败（实测踩坑）；需要代理部署时显式置 true
    llm_mock_delay: float = 0.0  # mock 模式每次 LLM 调用的模拟延迟秒数（压测用，模拟真实 LLM 延迟形态）
    json_repair_rounds: int = 2  # 结构化输出解析失败后的纠错重试轮数

    # ---------- Embedding（留空复用 LLM 配置） ----------
    embed_enabled: bool = True  # 无 embedding 额度时置 false：向量路休眠，检索退化为 BM25+业务加权
    embed_api_base: str = ""
    embed_api_key: str = ""
    embed_model: str = "BAAI/bge-m3"
    embed_dim: int = 1024
    embed_batch_size: int = 32

    # ---------- Rerank（可选，失败自动降级为不重排） ----------
    rerank_enabled: bool = True
    rerank_api_base: str = "https://api.siliconflow.cn/v1"
    rerank_api_key: str = ""
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_timeout: float = 10.0

    # ---------- 扫描件 OCR（视觉模型兜底） ----------
    ocr_model: str = "qwen-vl-max-latest"  # 与 LLM 同端点；需该模型有额度
    ocr_cache_dir: str = str(BASE_DIR / "data" / "ocr_cache")

    # ---------- 摄取 ----------
    chunk_size: int = 450  # 按中文政策条文实测 300/450/800 对比后取 450，见 eval 报告
    chunk_overlap: int = 60
    corpus_dir: str = str(BASE_DIR / "data" / "corpus")
    lexicon_path: str = str(BASE_DIR / "data" / "lexicon.json")

    # ---------- 检索 ----------
    top_k_recall: int = 20  # BM25 / 向量各自召回量
    top_k_final: int = 5  # 融合重排后进入生成的 chunk 数
    rrf_k: int = 60  # Reciprocal Rank Fusion 常数
    vector_weight: float = 0.8  # RRF 中向量路权重（检索路置信度加权：
    # 实测 embedding 质量不稳定时向量路噪声会污染融合排序，降权可护住 BM25 主信号；
    # 真实 bge-m3 下可回调至 1.0）
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    city_match_boost: float = 1.8  # 命中城市元数据的融合分加成
    domain_match_boost: float = 1.3  # 命中领域元数据的加成
    title_match_boost: float = 1.6  # 政策标题(事项名)与查询核心词重叠≥40% 的加成
    invalid_doc_penalty: float = 0.3  # 已废止政策得分乘数（保留可检索但强降权）

    # ---------- Agentic 编排 ----------
    grade_threshold: float = 0.5  # LLM 文档相关性打分阈值（0~1）
    max_query_rewrites: int = 2  # 黑话改写 + 补全重检索的最大轮数（防死循环）
    max_generate_retries: int = 1  # 忠实度校验不通过后的重生成次数
    graph_recursion_limit: int = 20

    # ---------- 存储 / 服务 ----------
    db_path: str = str(BASE_DIR / "data" / "index.db")
    trace_dir: str = str(BASE_DIR / "data" / "traces")
    history_window: int = 6  # 多轮对话携带历史轮数（token 成本控制）
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    # 知识库写接口（上传/删除/重摄取）管理密钥：未配置 = 写接口整体禁用（403），
    # 公网部署防投毒/清空（S3 评审 PC-3）；配置后写请求须带 X-Admin-Key 头
    admin_key: str = ""
    # 跨域白名单（浏览器跨域演示才需要；默认空 = 仅同源，CORS 不放 *）
    cors_origins: list[str] = []

    # 支持的城市与领域（路由与元数据过滤的词表）
    cities: list[str] = ["杭州", "上海", "北京", "深圳", "广州", "成都", "全国"]
    domains: list[str] = ["公积金", "社保", "医保", "通用"]

    @property
    def effective_llm_mode(self) -> str:
        if self.llm_mode in ("mock", "real"):
            return self.llm_mode
        return "real" if self.llm_api_key else "mock"

    @property
    def effective_embed_base(self) -> str:
        return self.embed_api_base or self.llm_api_base

    @property
    def effective_embed_key(self) -> str:
        return self.embed_api_key or self.llm_api_key


@lru_cache
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """测试用：修改环境变量后重建配置缓存。"""
    get_settings.cache_clear()


settings = get_settings()

# CI / 测试环境强制 mock，避免外部依赖导致不稳定
if os.environ.get("POLICY_COMPASS_FORCE_MOCK") == "1":
    settings.llm_mode = "mock"
