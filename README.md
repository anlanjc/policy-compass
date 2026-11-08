# 🧭 政策罗盘 PolicyCompass

> 跨城市社保/公积金政策的 **Agentic RAG** 问答服务 —— 把「Corrective RAG 自纠错检索 + Function Calling 精确计算 + 忠实度校验拒答」落成可运行、可评测、可部署的工程系统。
>
> 站在 [Datawhale《All in RAG》](https://github.com/datawhalechina/all-in-rag) 教程"四步构建 RAG"之上的业务化演进：教程教 RAG 基础，本项目补上教程尚未覆盖的 **Agentic RAG** 专题，并以政策问答这一强约束场景做深。

[![CI](https://github.com/anlanjc/policy-compass/actions/workflows/ci.yml/badge.svg)](https://github.com/anlanjc/policy-compass/actions/workflows/ci.yml) [![pytest](https://img.shields.io/badge/pytest-29%20项-brightgreen)]() [![eval_v1_real](https://img.shields.io/badge/v1_real-100%25%20(33%2F33)-brightgreen)]() [![eval_v2_real](https://img.shields.io/badge/v2_real-93.3%25-brightgreen)]() [![recall@5](https://img.shields.io/badge/recall@5-97.9%2F97.2-blue)]() [![消融](https://img.shields.io/badge/向量腿已真模型复验-生产维持BM25·结论留档-blue)]()

## 架构总览

```mermaid
flowchart LR
    Q[用户提问<br/>跨城/黑话口语] --> R{{混合检索<br/>BM25+向量·城市加权}}
    R --> CRAG{{CRAG 自纠错}}
    CRAG -->|正确| FC[Function Calling<br/>精确计算工具]
    CRAG -->|存疑| RW[查询改写<br/>黑话词典 token 化扩展] --> R
    CRAG -->|语料外| REF[诚实拒答<br/>编造=0]
    FC --> FV{{忠实度校验<br/>数字 vs 检索原文}}
    FV --> CITE[引用定位<br/>chunk 级条款/页码/文号徽章]
    FV -->|不忠实| REF
    CITE --> A[SSE 流式答案<br/>卷宗时间线面板]
```

## 界面速览

| Function Calling 计算 + 工具调用 Trace | CRAG 自纠错循环 → 诚实拒答 |
|---|---|
| ![FC 计算](docs/screenshots/02_calc_fc.png) | ![CRAG 拒答](docs/screenshots/04_refuse.png) |

| 跨城转移接续 · 文号引用 | 首页 · Agent 决策路径实时面板 |
|---|---|
| ![转移接续](docs/screenshots/03_transfer.png) | ![首页](docs/screenshots/01_home.png) |

> 前端采用**「政务文书」设计语言**（纸面浅色 / 藏蓝 + 朱红 / 衬线标题 / 印章标识 / 文号徽章引用卡）——与目标用户（查政策的市民）和语料形态（红头文件）一致，也与 askbi 的暗色数据大屏形成两个真实产品的差异化。左侧「决策路径」面板由 SSE 全链路实时推送逐节点留痕（路由→检索→评分→黑话改写→工具调用→生成→忠实度校验）；每条引用自带文号徽章 + 生效日期 + 条款号 + 现行/废止标注，可溯源到政策原文。部署手册见 [DEPLOY.md](DEPLOY.md)。

## 为什么是"政策问答"而不是又一个知识库 Demo

通用 RAG demo 的通病是**技术悬空**——没有任何业务约束强迫你做对什么。政策问答的每个技术设计都被业务事实逼出来：

| 业务事实（人人可验证） | 推导出的技术设计 |
|---|---|
| 同名政策各地不同：北京租房提取 2000/月、杭州 2000/月（杭房公委〔2023〕8号；北京另有 1500→2000 的调整历史） | **意图路由 + 元数据过滤**：先识别城市/领域再检索，不带城市过滤必然答错城市 |
| 政策会修订废止（当前语料 23 份均现行有效；废止降权机制由合成样本单测钉住，历史语料曾实测泄漏 2→0） | 摄取期抽取**文号/生效日期/状态元数据**，检索期废止文件强降权、排序期现行优先 |
| 额度公式是确定的，LLM 心算不可靠，政策数字答错即事故 | **Function Calling 计算工具**：LLM 只做语义抽参，公式硬编码可复算（"确定性与模糊推理的边界"） |
| 用户说"断缴/跳槽/把公积金带走"，条文写"中断缴费/工作变动/转移接续" | **黑话词典 + 查询改写**，CRAG 循环里做业务化改写 |
| 问"拍车牌/退税"等库外问题，宁可拒答不能编 | **拒答分支 + 忠实度自检**，两次不通过强制降级为"给原文+建议热线" |

## 架构

```mermaid
flowchart TB
    U[用户问题] --> R[意图路由<br/>LLM 结构化输出<br/>城市/领域/需计算<br/>失败降级规则路由]
    R --> RET[混合检索<br/>BM25 手写实现 + 向量<br/>RRF 融合 + 城市/领域/标题加权<br/>废止文件降权 + rerank]
    RET --> G{文档相关性评分<br/>LLM 并行打分 0~1}
    G -- 相关 --> T{需计算?}
    G -- 不相关 --> W[黑话改写重检索<br/>≤2 轮防死循环]
    W --> RET
    W -- 改写耗尽 --> F[诚实拒答<br/>不编造]
    T -- 是 --> TC[Function Calling<br/>LLM 抽参 → 公式硬编码计算]
    T -- 否 --> GEN
    TC --> GEN[生成<br/>带文号引用 标注现行/废止]
    GEN --> V{忠实度校验<br/>逐句溯源}
    V -- 通过 --> OUT[回答 + 引用来源]
    V -- 不通过 --> GEN2[强化提示重生成 1 次]
    GEN2 -- 仍失败 --> F

    subgraph 支撑层
      ING[摄取管线<br/>条文感知切分<br/>元数据抽取<br/>SQLite 增量索引]
      EVAL[评估闭环<br/>33 题六类评测集<br/>recall/MRR/消融对比]
      TRACE[全链路 Trace<br/>SSE 实时推送<br/>JSONL 留档]
    end
```

## 核心特性（附面试考点）

| # | 特性 | 实现要点 | 面试常问 |
|---|---|---|---|
| 1 | **CRAG 自纠错编排** | LangGraph StateGraph：检索→评分→条件边改写循环→生成→忠实度校验→降级，量化信号驱动条件边 | 为什么需要 agent？循环怎么防死循环？ |
| 2 | **手写混合检索** | BM25（Okapi 公式逐行实现）+ 向量双路 → RRF 融合 → 城市/领域/标题事项名三级加权 → bge-reranker 重排（失败自动降级） | 为什么不用 LangChain Retriever？加权信号怎么来的？ |
| 3 | **中文分词粒度治理** | 实测踩坑："杭州市住房公积金"（文档）vs "杭州公积金"（查询）词面完全失配致 BM25 零命中 → 词 + CJK 字符 bigram 双通道索引（Lucene CJK 思路） | BM25 原理、idf 饱和、b 参数作用 |
| 4 | **Function Calling 工具层** | LLM 语义抽参（chat_json 自校验）→ 公式纯函数计算（三城规则）→ 同一工具经 MCP server 二次分发 | 工具调用完整流程？为什么不用原生 tools 参数？ |
| 5 | **政策时效管理** | 元数据（文号/生效日期/状态）摄取期抽取；检索期废止降权保留可查（能答"政策变了什么"）；文件 hash 增量热更新 | RAG 知识库热更新怎么做？ |
| 6 | **可靠性工程** | LLM 指数退避重试 / 结构化输出解析失败自校验纠错 / 评分失败保守记分 / rerank 失效降级 / 全局异常兜底 SSE error 帧 | 一环挂了系统会怎样？ |

## 快速开始

```bash
# 方式一：零配置体验（mock 模式，全链路可跑，无需任何 API key）
pip install -r requirements.txt
python scripts/serve.py
# 打开 http://127.0.0.1:8000 —— 打字机流式 + 实时 agent 决策路径面板

# 方式二：接入真实模型（推荐，出正式评测数字）
cp .env.example .env   # 填 LLM_API_KEY（DeepSeek/GLM/硅基流动）与 EMBED_API_KEY
python scripts/serve.py

# 方式三：Docker
docker compose up --build
```

CLI 演示（适合录屏）：
```bash
python scripts/chat.py "我杭州公积金余额4万缴了5年月薪2万，最多能贷多少？"
```

## 评测（一键复现）

```bash
python scripts/eval.py            # 检索指标 + 消融 + 端到端，产出 docs/eval_report.md
python -m pytest tests/ -q        # 29 项测试（mock 模式，零 API key 可跑 CI）
```

**评测分层口径**：v1 33 条 = smoke 回归集（定位声明：回归集不是 benchmark）；v2 90 条 = full 分层集（事实 25/对比 15/计算 15/拒答 12/时效 8/黑话 8/多跳 7，其中 15 条 holdout 永不参与调参、报告单列）。所有百分比来自实际 runner 输出，mock 与 real 分开标注，完整证据链（消融/故障注入/压测/失败归因/复现命令）见 [EVIDENCE.md](EVIDENCE.md)。

**正式口径**：`deepseek-v4-flash-0731`（阿里云百炼）+ BM25 混合检索（embedding 额度不可用，`EMBED_ENABLED=false` 向量路休眠，架构支持一键恢复；**向量腿价值已消融实测**——MRR/端到端正增益、recall@5 弱增益，保留开关生产默认关闭，见 [docs/ablation.md](docs/ablation.md)）+ 分层评测（v1 33 题 smoke + v2 90 题分层）。语料 16 份真实文件（7 城+全国，另 7 份标注演示）。完整报告见 [docs/eval_report.md](docs/eval_report.md) 与 [EVIDENCE.md](EVIDENCE.md)。

**检索消融**（v2 78 题含目标文档集；BM25 行=PC-3 检索修复后 2026-11-05 复跑，与冻结一致；向量行=2026-11-05 真实 embedding 实测，[完整消融报告](docs/ablation.md)）：

| 配置 | recall@5 | MRR | 已废止文件泄漏进 Top5 |
|---|---|---|---|
| BM25 纯净 | 97.9% | 0.949 | 0 |
| **BM25 + 业务加权（生产 BM25-only）** | 97.2% | 0.949 | **0** |
| 混合+业务加权（真实向量 text-embedding-v4） | **98.1%** | **0.984** | **0** |

> 数字解读：BM25 两行差距在 ±1~2 题翻转量级（n=78 小样本噪声级）——**加权的保留价值是废止零泄漏与跨城召回结构，非 recall 增益**（PC-2 诚实翻转结论不变；hybrid_no_boost 98.7%>hybrid_full 98.1% 亦提示向量开启时加权有轻微抢占，不在噪声量级上调参）。**向量腿实测（含真模型复验）**：recall@5 +0.9pp 弱增益、MRR 0.949→0.984 排序真增益、mock e2e +2.2pp——但**真模型同金标复验 v2 92.2%/v1 93.9%（vs 冻结 93.3%/100%），差异全为评分抖动误拒、方向反向，mock 增益不可外推 real**——**最终结论=保留架构与开关，生产默认维持 BM25-only**（详见 ablation.md §0/§4）。废止泄漏由单元测试覆盖。

**端到端行为指标**（real 口径，deepseek-v4-flash-0731，2026-10-29）：

| 评测集 | n | 通过率 | 说明 |
|---|---|---|---|
| v1 smoke | 33 | **100.0%** | 33/33 全过（PC-3 2026-11-02 复测：检索扩展机制修复后，历史 2 例评分抖动误拒消解；冻结口径 93.9% 见 EVIDENCE §4） |
| v2 full | 90 | **93.3%** | calc 14/15 · refuse 12/12 零编造；PC-3 复测与冻结口径持平（84/90，失败集更新见 EVIDENCE） |
| v2 full（mock） | 90 | **91.1%** | mock 链路验证口径（PC-3 检索扩展机制修复后复测：扩展词 df 命中率≈0→100%；差 3 条为 mock 抽取脆性、真模型 5/5 验证通过，见 EVIDENCE PC-3 节） |

> real 两行为 PC-3（2026-11-01/02）检索机制修复后全量重跑口径：黑话扩展 token 化（修复前扩展词 df 命中率≈0，feature 近乎空转）+ 多城问法城市加权——v1 100%（历史 2 例评分抖动误拒消解）、v2 93.3% 与冻结口径持平，检索消融 recall@5 95.9/95.3 → 97.9/97.2。冻结历史口径与逐轮明细见 [EVIDENCE.md §4](EVIDENCE.md)。完整失败归因（0 条编造）见 [evidence/fail_analysis.md](evidence/fail_analysis.md)。完整证据链（消融/故障注入 61 次编造=0/压测/复现命令）见 [EVIDENCE.md](EVIDENCE.md)。


## API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/chat` | 问答（`stream: true` 返回 SSE：meta→trace→token→done） |
| GET | `/api/documents` | 文档列表（含状态/文号/城市元数据） |
| POST | `/api/documents` | 上传摄取（增量，自动重建索引） |
| DELETE | `/api/documents/{id}` | 删除并级联清索引 |
| POST | `/api/ingest/sync` | 重扫语料目录（政策热更新，内联等待完成） |
| POST | `/api/ingest/reindex` | **后台**增量重建：立即 202 返回，singleflight 防叠跑（进行中 409） |
| GET | `/api/ingest/reindex/status` | 重建进度轮询：running / 报告 / 当前快照版本 |
| GET | `/api/health` · `/api/stats` | 健康 / 运行统计（token、重试、JSON 修复率、拒答数） |

> 热更新机制（ADR-6）：文件 sha256 指纹三态增量（未变零重嵌/变更重嵌/删除级联），
> 构建完成后一次性原子发布新快照——在途请求用旧快照跑完，新请求自动切新版本。
> CLI 入口：`python scripts/ingest.py --reindex-incremental`；写接口均需 `X-Admin-Key`。

## 设计取舍 FAQ

**为什么 SQLite + numpy 而不是 Milvus/Chroma？**
目标规模数千 chunk，numpy 暴力余弦 <5ms，独立向量库反而增加部署复杂度。检索层只依赖 `chunks()`/`matrix()` 两个方法，规模上万后平滑替换 Milvus 适配层。

**为什么 LangGraph 之外还有一个"手动流式"路径？**
LangGraph 原生流式只到节点粒度。`POST /api/chat?stream=true` 复用**同一组节点函数**驱动前置阶段，生成节点交由 `llm.chat_stream` 做 token 级推送，流毕执行 verify（不通过推 correction 帧替换答案）。节点函数是唯一真相源，图与非流式行为一致——框架服务需求，而非反过来。

**为什么工具调用用 chat_json 而不是原生 tools 参数？**
不是所有 OpenAI 兼容端点都稳定支持 tools；且 chat_json 链路已有"解析失败→错误回灌→重试"的自校验，参数错误率可控可观测（`/api/stats` 的 json_first_pass_failure_rate）。

**rerank 挂了会怎样？**
捕获异常 → 跳过重排用 RRF 融合序（`stats.rerank_fallbacks` 计数），主链路无感。

## 与 All-in-RAG 教程的关系

- 教程[第四章](https://datawhalechina.github.io/all-in-rag/)讲混合检索原理 → 本项目给出**带业务加权信号与消融数字的工程实现**；
- 教程第六章讲评估方法论 → 本项目给出**六类 33 题评测集 + 四组消融 + 一键报告**的闭环；
- 教程 Extra-chapter 欢迎社区投稿 **Agentic RAG** 专题 → 本项目即是该专题的一个完整参考实现（CRAG 编排 + 拒答 + 工具调用）。

## 目录结构

```
policy-compass/
├── app/
│   ├── llm/          # OpenAI 兼容客户端（重试/SSE/自校验）+ 确定性 mock
│   ├── ingestion/    # 条文感知切分 / 元数据抽取 / SQLite 增量索引
│   ├── retrieval/    # 手写 BM25 / RRF 混合检索 / rerank 降级
│   ├── agent/        # 意图路由 / Function Calling 工具 / LangGraph CRAG
│   ├── api/          # FastAPI SSE / 文档管理 / 统计
│   ├── eval/         # 指标 + 消融 + 报告
│   └── obs/          # trace JSONL
├── data/corpus/      # 23 份政策语料：16 份真实文件（7 城+全国，含贷款/社保/居转户/利率，
│                     #   每份头部注明 source_url 与抓取时间）+ 7 份演示文档（头部有【演示语料】声明）
├── data/eval_set.jsonl     # v1 33 题评测集（smoke 回归集）
├── data/eval_set_v2.jsonl  # v2 90 题分层评测集（含 15 条 holdout）
├── mcp/server.py     # 计算工具的 MCP 暴露（可选 pip install "mcp[cli]"）
├── static/index.html # 演示前端（打字机 + trace 面板）
├── scripts/          # serve / ingest / eval / chat / fault_inject / load_test
├── evidence/         # 证据链归档（评测报告/故障注入/压测/失败归因/git log）
└── tests/            # 29 项 pytest
```

> 语料声明：`data/corpus` 中 **16 份为真实政策文件（覆盖 7 城市+全国）**——北京租赁提取〔2023〕6号 / 北京直付房租〔2025〕1号 / 北京贷款〔2018〕51号 / 上海租赁提取沪公积金管委会〔2024〕6号 / 上海贷款沪公积金管委会〔2026〕1号（官方站不可达，双源权威转载交叉核对）/ 广州穗公积金中心规字〔2024〕2号 / 深圳提取管理规定深公积金规〔2025〕3号 / 成都 2026 租房提取调整（人民网转载）/ 杭州贷款杭房公委〔2026〕1号（杭州市政府公报）/ 《住房公积金管理条例》2026 第三次修订（国家行政法规库）/ 《社会保险法》条文节选（主席令第35号，国家信访局全文页）/ 国办发〔2019〕13号《降低社会保险费率综合方案》（人社法规库转载）/ 沪府规〔2025〕8号居转户办法要点 / 央行《关于下调个人住房公积金贷款利率的通知》（2025-05-08 起，法规库转载）/ 杭州租赁提取 2 份红头扫描件 PDF（视觉 OCR 摄取）。**每份文件头部注明来源 URL 与抓取时间**；成都官方站反爬无法直抓，采用人民网/成都日报权威转载并标注。其余 **7 份**仍为按真实结构编写的演示数据（头部有【演示语料】声明，仅用于教学演示），相关官方文件发布后按同流程替换。工具限额表（`app/agent/tools.py` RENT_LIMITS/LOAN_RULES）与评测金标已对齐真实数字，来源以注释形式标注到文号。
>
> **扫描件解析**：pypdf 对红头扫描件抽取文本近于零（实测 4 页 3 字符），摄取管线自动降级视觉模型 OCR（`app/ingestion/ocr.py`），结果按文件哈希缓存于 `data/ocr_cache/`。

## Roadmap

- [ ] rerank 后的查询子问题并行检索（对比型问题拆分检索再融合）
- [ ] Langfuse 接入（当前自研 trace 的生产化替代）
- [ ] 对比型问题的多路城市子查询（当前靠加权解决）
- [ ] 政策"新旧差异"专门回答模式（已废止文件单独成块）
