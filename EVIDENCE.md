# EVIDENCE · PolicyCompass 证据链（2026-10-28/29 去 demo 化冲刺）

> 本文所有数字**可复现**：复现命令见 §5，mock 与 real 严格分开标注。评测分两套口径：
> **smoke=v1 33 条**（`data/eval_set.jsonl`，回归集，定位声明：回归集不是 benchmark）；
> **full=v2 90 条**（`data/eval_set_v2.jsonl`，分层集，含 15 条 holdout 单列）。

## ① 我做了什么 / 没做什么（诚实边界）

**做了**（2026-10-28/29，AI 辅助开发，约 3 小时会话窗口）：

- 语料去 demo 化：`data/corpus` 从「3 城演示 + 2 份真实 PDF」扩到 **10 份真实文件 × 7 城市**（北京×3 / 上海 / 广州 / 深圳 / 成都 / 杭州×2 / 全国条例）——全部来自官方原文或权威媒体转载，每个文件头部注明 `source_url` 与抓取时间，抓取方式为 curl + 政策库转载页（官方站 gjj.gz.gov.cn / zjj.sz.gov.cn / moj.gov.cn 可直抓；cdzfgjj.chengdu.gov.cn / chengdu.gov.cn 反爬 412，成都改用人民网/成都日报转载并标注）。工具限额表、评测金标同步换成真实数字（上海 3000→4000/户、广州 1800→2000、成都新增 1800、深圳按比例制不收录进固定额度表）。
- 评测扩容：v2 分层评测集 90 条（fact 25 / compare 15 / calc 15 / refuse 12 / 时效 8 / 黑话 8 / 多跳 7，其中 15 条标 `split:"holdout"` 不参与任何调参）；金标双验证（关键词逐条 grep 语料原文 + 计算题用工具公式复算）。
- 可靠性实证：故障注入 5 场景 61 次、mock 压测两轮（零延迟 500×3 档 + 模拟 LLM 延迟 1.5s×150×3 档）。
- 修复 3 个真 bug + 1 个真缺陷（见 §3）。

**没做 / 边界**：

- **未上线、无真实用户**：没有生产部署、没有真实流量数据；压测是 mock 模式下的本机 HTTP 栈压测，不代表生产容量。
- **杭州 2 份真实 PDF 之外的城市此前均为演示语料**：本次替换了北京/上海/广州/深圳/成都的租赁提取与北京贷款、全国条例；演示文档 13 份经 **PC-2（2026-10-29）两轮真实化后剩 7 份**（文件头部有【演示语料】声明）：①沪杭贷款两份已真实化（沪公积金管委会〔2026〕1号——官网 shgjj.com 连接不可达，双源权威转载交叉核对；杭房公委〔2026〕1号——杭州市政府公报刊载全文），LOAN_RULES 两城换真实数字，沪杭现行文件未载收入比条款，演示口径的 0.6/0.5 按"宁缺毋滥"删除；②演示文档替换 4 份：《社会保险法》条文节选（主席令第35号，国家信访局全文页）、国办发〔2019〕13号《降低社会保险费率综合方案》（人社法规库转载）、沪府规〔2025〕8号居转户办法要点（2025-12-01 施行，替代 2020 旧版）、央行《关于下调个人住房公积金贷款利率的通知》（2025-05-08 起，首套5年以上 2.85%→2.6%）；③其余 7 份（北京社保补缴/杭州转移/全国转移规程/断缴影响×2/提取案例/上海社保转移）确无单一可抓取的现行官方对应文件——尝试记录见 fail_analysis.md PC-2 节，如实保留演示标注。真实文件 **10→16 份**。
- AI 辅助：代码由 AI 会话产出、人审核验收；git 提交为真实时间线，未 push（由用户决定）。
- API 花费：真模型评测仅用于 §4 的 real 口径数字（dashscope deepseek-v4-flash-0731），累计花费见 §4 脚注。

## ② 三个关键决策

1. **检索为什么 BM25-only（EMBED_ENABLED=false）**：账号 embedding 额度不可用；BM25 加权 vs 纯净的消融见 §4（2026-11-05 复跑：97.9/97.2，加权保留价值=废止零泄漏+跨城召回结构，非 recall 增益）。**向量腿价值已于 2026-11-05 以真实 embedding 消融实测**（text-embedding-v4，S1）：MRR/端到端正增益、recall@5 弱增益（v2 +0.9pp）、泄漏仍 0——结论=保留架构与开关、生产默认开启留决策（详见 docs/ablation.md；生产开启前置=真模型复测+CI 门禁基线重建）。
2. **为什么拒答降级而不是强行回答**：政策数字答错即事故。故障注入实证：LLM 完全不可用时系统诚实拒答（10/10）、grade 评分链路被打断时保守记 0.3 分宁可少答（10/10 拒答）、库外城市宁可"未收录"不张冠李戴（修复后编造=0）。拒答不是失败，是设计。
3. **为什么评测分层（smoke v1 / full v2 / holdout 单列）**：v1 33 条是回归集——改代码后 5 分钟确认"老题不回归"（本次实测 75.8%→78.8%）；v2 90 条按题型分层才能暴露结构性弱点（如 mock 下 refuse 词面噪声）；holdout 15 条永不参与调参，防止"在测试集上调参"的自欺。

## ③ 三个真实 bug（含复现与修复）

1. **库外城市『张冠李戴』（严重，本次最有价值发现）**：问"重庆公积金余额2万能贷多少"（重庆不在库），mock 工具选择器 city 兜底默认"杭州"，工具按杭州规则算出额度返回——数字正确、张冠李戴，用户会当成重庆的答案。复现：`python scripts/fault_inject.py` 场景 4。修复三层：`nodes.tool_node` 防猜守卫（问题无支持城市且路由为全国→不调工具；路由城市与模型选城不一致→以路由为准覆盖）+ `TOOL_PROMPT` 硬规则（无城市线索必须 tool=none）+ 生成提示词规则 1（问特定城市而资料是其他城市→明说未收录）。修复后 61 次注入编造=0。
2. **mock/real 维度不一致崩溃**：`MockLLM.embed` 固定 256 维，而关闭 embedding 时真实链路按 `settings.embed_dim`(1024) 落零向量——mock 模式对真实链路摄取的库查询直接 `matmul` 维度崩溃（`ValueError: size 256 is different from 1024`）。此前从未在 mock 模式对真实库跑过评测所以没暴露。修复：mock 维度对齐 `settings.embed_dim`。
3. **runner 判分 200 字截断误判**：`eval_end2end` 对答案截断 200 字符后判关键词——真实语料答案较长，额度数字常在 200 字之后，mock 误判约 10 条（v2 首跑 76.7% 的主要成分）。修复：判分用完整答案、展示仍截断。附带发现：**Windows 系统代理让 httpx 连 dashscope 的 TLS 握手全挂**（env 无代理变量、注册表级代理，Git Bash curl 不受影响所以一直没暴露）→ `llm_trust_env=False` 默认绕过；与 AB 会话在 askbi 上独立发现的同一环境坑交叉印证。

## ④ 评测表（mock 与 real 严格分开）

**检索消融（v2 78 题含目标文档集，mock 口径，确定性计算不依赖 LLM；下表为 PC-3 检索修复后 2026-11-05 复跑，与冻结口径 97.9/97.2 一致）**

| 配置 | recall@5 | MRR | 已废止文件泄漏进 Top5 |
|---|---|---|---|
| BM25 纯净 | **97.9%** | 0.949 | 0 |
| **BM25 + 业务加权（本系统）** | 97.2% | 0.949 | **0** |

> 诚实记录：PC 会话语料下曾为纯净 95.3/加权 95.9，PC-2 真实语料替换后复跑翻转，PC-3 黑话扩展 token 化修复后两行同升至 97.9/97.2（±1~2 题、n=78 小样本噪声级）。**加权不再有 recall@5 增益，其保留价值是废止文件零泄漏（历史语料 2→0，本表两配置均 0，单测覆盖）与跨城同名政策的召回结构**；是否保留加权以端到端与更多语料验证为准，不在 mock 噪声量级上调参（holdout 纪律同理）。向量腿（真 embedding）消融见下文 S1 节。

**端到端（v2 90 题，mock 口径，PC-2 后）**：总体 **94.4%**（85/90），P50 19.5ms——fact 96.0%（24/25）/ compare 86.7%（13/15）/ calc 100%（15/15）/ refuse 91.7%（11/12）/ 时效 100%（8/8）/ 黑话 87.5%（7/8）/ 多跳 100%（7/7）。失败 5 条（#105/129/131/167/183）逐条归因见 `evidence/fail_analysis.md`（mock 词面噪声/引用排序局限，检索召回均正确；较 PC 会话基线修复 #156/157 两条、零新增）。

**端到端（v1 33 题 smoke，mock 口径基线对照）**：P1 语料扩容前 75.8% → 扩容后 78.8%（git worktree 实测对照，净增 3pp）；PC-2 语料真实化+mock 元数据过滤修复后 **84.9%**（28/33，较 P1 基线 +6.1pp；#24 修复、#6 暴露为正文句排序噪声，明细见 fail_analysis.md PC-2 节）。

**端到端（real 口径，dashscope deepseek-v4-flash-0731，2026-10-29）**

| 评测集 | n | 总体通过率 | P50 延迟 | 报告 |
|---|---|---|---|---|
| v2 full（real） | 90 | **93.3%**（84/90） | 15.0s | `evidence/eval_v2_real.md` |
| ├ main（非 holdout） | 75 | 93.3%（70/75） | — | 同上 |
| └ holdout（单列） | 15 | 93.3%（14/15） | — | 同上 |
| v1 smoke（real） | 33 | **93.9%**（31/33，与语料更新前历史口径持平——零回归） | P50 20.1s（按题型均值表） | `evidence/eval_v1_real.md` |

**PC-3 复测（real，2026-11-02，检索机制修复后全量重跑，同模型）**

| 评测集 | n | 总体通过率 | P50 延迟 | 报告 |
|---|---|---|---|---|
| v1 smoke（real） | 33 | **100.0%**（33/33，历史 2 例评分抖动误拒被扩展机制修复消解） | 20.1s | `evidence/eval_v1_real_pc3.md` |
| v2 full（real） | 90 | **93.3%**（84/90，与冻结口径持平；refuse 12/12 零编造，calc 14/15） | 14.7s | `evidence/eval_v2_real_pc3.md` |

- **PC-3 复测实际分题型**（`evidence/eval_v2_real_pc3.md`）：fact 88.0% / compare 93.3% / calc 93.3%（14/15，#144 沪家庭额度误拒）/ refuse **100%**（12/12 零编造）/ 时效 100% / 黑话 100% / 多跳 85.7%；失败集 **#113/#114/#116/#131/#144/#190**（多为金标写法变体与保守方向误拒，`evidence/fail_analysis.md`）。v1 复测 33/33 全过。
- **冻结轮分题型与归因（2026-10-29 `evidence/eval_v2_real.md`/`eval_v1_real.md` 口径，非 PC-3 复测数据，归属标注修正见 S3 评审 PC-11）**：
  - v2 分题型：fact 96.0% / compare 86.7% / calc **100%**（Function Calling 额度全对）/ refuse **100%**（12/12 零编造）/ 时效 87.5% / 黑话 87.5% / 多跳 85.7%。
  - 冻结轮 6 条未通过归因：2 条评分抖动误拒（方向=宁可不答，无编造）、1 条误称"未收录"（保守方向）、3 条金标数字写法变体（模型把"百分之六十五"规范化为"65%"、"三个月"→"3个"）；并入写法变体后 post-hoc 96.7%（87/90，**事后分析未重跑**，正式口径以冻结金标 93.3% 为准）。
- PC-3 复测 usage：llm_requests=866 retries=0 failures=0 prompt_tokens=536,170 completion_tokens=130,307 refusals=14 tool_calls=19。
- provider：阿里云百炼 DashScope 兼容模式，`deepseek-v4-flash-0731`（推理模型）；embedding 关闭（额度不可用），检索为 BM25+业务加权。
- 冻结轮 v1 分题型（2026-10-29 `evidence/eval_v1_real.md`）：fact 88.9%（8/9）/ compare 100% / calc 100% / refuse 100% / slang 100% / 时效 75%（3/4）；2 条失败（#31 已故职工小额提取、#29 上海居转户基数）均为评分抖动误拒——答案方向是"宁可不答"而非编造。usage：llm_requests=337 retries=0 failures=0 prompt_tokens=214,875 completion_tokens=59,570。
- 真模型总花费：两轮合计 ≈0.94M 输入 + 0.19M 输出 token，按 flash 档典型价（输入 ¥0.5~2/M、输出 ¥2~8/M）估算 **≈¥1~2.5**（中位 ≈¥1.7），以百炼账单为准（预算上限 ¥3，未超）。
- holdout 说明：15 条在评测集构建时随机分层抽出，未参与 lexicon/提示词/加权任何调整。

**PC-2 追补回归（2026-10-29）：沪杭贷款规则真实化后，mock 与 real 分列**

- 金标跟随真实原文修正：#184 关键词 `余额倍数`→`月均余额`/`20倍`（真实原文表述，旧演示文档字样已不存在）；calc 9 条金标按工具复算更新（杭州 15倍/60/120万→20倍/90/180万；上海 ×10倍/50/100万→首套限额制 100/200万）——均注明文号，可从语料原文逐字验证。
- **mock**：v2 **92.2%**（83/90，失败集合与语料替换前完全一致）、v1 **78.8%**（26/33，失败集合与 P1 基线完全一致）——零回归。
- **real 子集**（dashscope deepseek-v4-flash-0731，2026-10-29，按"只重跑受影响条目"纪律）：v2 贷款子集 **10/10=100%**（含 holdout #144）、v1 贷款子集 **8/8=100%**；usage 合计 122,696 in + 38,341 out token。报告：`evidence/eval_loan_regress_v2.md` / `eval_loan_regress_v1.md`；子集定义 `data/eval_loan_regress_v2.jsonl`（10 条）/ `_v1.jsonl`（8 条）。

**PC-2 第二轮（2026-10-29）：演示文档替换 4 份真实化后**

- 金标修正：v1 #8 利率 2.85→**2.6**（央行 2025-05-08 降息后真实口径）。
- **mock**：v2 **94.4%**（85/90，较 PC 会话基线 92.2% **+2.2pp**，失败 #105/129/131/167/183——真实语料入库顺带修复 #156/157 两条词面噪声，零新增失败）；v1 **84.9%**（28/33，较 P1 基线 78.8% **+6.1pp**）。
- mock 生成器缺陷修复：元数据行（【语料性质】等）被数字加分抬升压过正文条文——mock 内容过滤补"【 开头不进答案"（仅 mock 层）；副作用：v1 #6 此前靠元数据行侥幸通过，暴露为正文排序噪声（诚实记录，见 fail_analysis.md）。
- **real 抽查**：语料替换涉及的 v1 4 条（#8/#24/#25/#29）**4/4=100%**；另补跑 v2 金标/语料变更条 #129、#184 **2/2=100%**（`data/eval_loan_regress_v2b.jsonl`）——至此 PC-2 全部金标变更与语料变更条目的真模型回归覆盖完毕（v2 12 条 + v1 12 条 = 24 条全过）。
- PC-2 真模型总花费：154,876 in + 51,476 out token，估算 ≈**¥0.17~0.75**（红线 ≤¥1，未超）。
- 注：上表 v2 full 93.3% / v1 smoke 93.9% 为 PC 会话冻结金标下的历史口径；PC-2 金标修正共 11 条（calc 9 + #184 关键词 + #8 利率），受影响条目已按新金标以子集重跑（见上），历史口径数字不追溯改写。

**PC-3（2026-11-01/02）：引用原文定位特性 + 检索机制修复 + 全量复测**

- **特性**：chunk 级条款/页码元数据全链（splitter 条款标签记账 + PDF/OCR 页标记【pN】剥离记账；chunks 表 sections/pages 列 + 旧库自动迁移；引用卡升级"文号·生效日期·条款·现行/废止"，60/68 块带条款标签）。设计红线=零切分边界变化：新旧库 chunk 正文 dump **逐字节一致**（68 块不变），pytest 23→29 全绿。详见 git 1d9da42。
- **机制修复一（评测驱动的真 bug，影响面大）**：`HybridRetriever.expand_query` 注入的黑话扩展词未经分词归一，整串短语（"委托提取还贷"/"住房公积金账户余额"）在 `search_weighted` 的 df 查找中**静默失效**——实测扩展词 df 命中率≈0（仅"贷款额度"等恰好是 jieba 原生 token 的词生效），**黑话词典扩展自上线起近乎空转**，slang 项历史通过属排序巧合。修复=注入词先过 `tokenize_words`（token 化后 df 命中 100%）。修复后检索消融 recall@5 **95.9/95.3 → 97.9/97.2**、废止泄漏保持 0。
- **机制修复二**：多城问法（"北京和广州…分别是多少"）route 只能给全国，原城市加权不触发——补 query 文本中出现的城市名直接对照 chunk.city 施加 `city_match_boost`（单城路由路径不受影响）。
- **mock 复测**：v1 **84.9%**（28/33，失败集与 PC-2 记录逐项一致复现，#3 被扩展修复救回）；v2 **91.1%**（82/90，#127 被城市加权修复救回；失败 #105/129/131/167/177/179/183/189）。
- **缩水溯源（诚实记录）**：PC-2 记录 v2 mock 94.4% 的报告生成于 10-29 07:58，沪杭贷款语料真实化+金标修正提交在 **08:02 批次**，最终 HEAD 未复跑 mock 全量——11-01 复测先暴露 v1 81.8%/v2 90.0%（缩水与本特性无关：chunk 逐字节一致 + HEAD 代码同库 A/B 同分 + PYTHONHASHSEED 0/0/1 三重排除）。
- **剩余 v2 mock 失败中 3 条（#177/#179/#189）为 mock 抽取式脆性**，连同 #3/#127 的受影响 5 项**真模型探针 5/5 通过**（#3 关键词 200万✓ 来源沪公积金管委会〔2026〕1号；#127 2000✓ 穗+京双源；#177 转移接续✓；#179 委托提取✓；#189 加装电梯✓）。三条脆性诊断：#177 条例块真含"账户转移"字样被评分误判相关；#179 杭州转移文件标题含"办理…业务"触发 title_match_boost 抢占 top1；#189 右块 rank2 与 rank1 分差 0.001（mock 只抽 top1）。
- **real 全量复测（同模型 deepseek-v4-flash-0731，2026-11-02）**：v1 **100.0%**（33/33，历史 2 例评分抖动误拒消解）/ v2 **93.3%**（84/90，与冻结口径持平；refuse 12/12 零编造、calc 14/15）。usage：v2 568,198 in + 158,400 out；v1 221,450 in + 74,776 out；另有受影响 5 项探针调用——**PC-3 真模型总花费 ≈¥1~4**（账单以百炼为准）。

**S1 向量腿消融（2026-11-05/06，真实 embedding 实测，chat 全程 mock）**

- **三口径拆分（重要发现）**："BM25-only" 必须分清——A 生产口径（真客户端零向量→腿真空转，v2 recall@5 97.2%）/ B FORCE_MOCK 口径（**MockLLMClient.embed 不检查 embed_enabled，哈希词面向量腿实际在跑**，93.8%，比纯 BM25 低 3.4pp——CI 门禁与 e2e mock 均此口径，基线自洽）/ C 混合真向量（98.1%）。完整三口径表与门禁关系见 docs/ablation.md §3。
- **主表（v2 n=78 / v1 n=29，text-embedding-v4 1024 维）**：hybrid_full recall@5 v2 **98.1%**（vs 生产 BM25 97.2%，+0.9pp）/ v1 95.4%（-1.1pp 小样本负贡献）；**MRR 双集大涨**（v2 0.949→0.984、v1 0.831→0.902）；**e2e mock v2 91.1→93.3%（+2.2pp，PC-3 脆性题 #177/#189 被向量重排救回）、v1 84.9→87.9%（+3.0pp）**；废止泄漏全配置 0；hybrid_no_boost 98.7%>hybrid_full 98.1%（向量开启时加权轻微抢占，与 PC-2 诚实翻转同向）。延迟代价 +~0.3s/检索（生产 P50 15s 口径 <2%）。
- **真模型复验（2026-11-06，主人批准 ≈¥1~4/上限 ¥5）——证伪 mock 增益外推，结论修正为"生产默认维持关闭"**：真模型同金标全量 e2e，混合腿 v2 **92.2%**（vs 冻结 BM25-only 93.3%，-1 题）/ v1 **93.9%**（vs 冻结 100%，-2 题）；3 条差异**全部为 `refusal=True` 评分抖动误拒**（保守方向、零编造，refuse 子集 12/12、4/4 全过），落在 PC-3 已记录的抖动噪声带内（同类误拒在冻结口径历史运行中本就 93.9%↔100% 摆动）。**mock 判分器的 e2e 增益（+2.2/+3.0pp）不可外推到 real**——真实模型对 rank2~3 鲁棒、MRR 排序增益不传导到答案，mock 抽取式判分（只读 top1）才会被重排放大。usage：824,189 in + 238,623 out ≈**¥0.9~3.6**（未超上限）；报告 `evidence/eval_v{1,2}_real_hybrid.md`（内含四组检索消融表，与 hybrid 阶段独立复跑逐格一致）。
- **最终结论**：向量腿**保留架构与开关（降级是设计），生产默认维持 BM25-only**——排序增益真实但不传导，切换成本（延迟/门禁基线重建/口径更新）确定而收益证据不足；"若仍要开启"前置清单留档 ablation.md §4b。
- **意外发现（顺手报主人，不擅修）**：`splitter._pack` 构造 Chunk 从不赋值 title→`_ingest` embed 输入实为 `"\n"+正文`（标题经正文首行《标题》进入，功能无损）；修复会动全部 chunk 向量，留主人决策。
- **复现性纪律**：实验用原地 UPDATE 换装库向量三次（哈希→真→哈希），`--phase restore_mock` 以 MockLLMClient.embed（md5 确定性）恢复冻结兼容口径，恢复后 v1 84.9%/v2 91.1% **逐字复现**；DashScope 批量上限 10（config 默认 32 为 OpenAI 口径，脚本内钳制）。
- **花费**：chunk 重嵌 ×2 轮 63,467 字符 + 查询向量 ×2 轮 ≈1,360 次调用 ≈3.5 万字符，估算 **≤¥0.07**（审批上限 ¥0.1，未超；usage 字段客户端未捕获，按输入字符上界估，账单以百炼为准）。原始数据 `evidence/ablation_{bm25,embed,hybrid,restore_mock}.json`，脚本 `scripts/ablation_vector.py`（app/ 零改动）。

## ⑤ 复现命令（逐条可复制，均在 `policy-compass/` 目录内执行）

```bash
# 0) 环境（Python 3.12；无网环境用已装依赖，requirements 与实测环境逐版本一致）
pip install -r requirements.txt -i https://pypi.org/simple/

# 1) 摄取（BM25-only，EMBED_ENABLED=false；增量，hash 未变跳过）
python scripts/ingest.py

# 2) 测试回归（mock 模式，零 API key）
python -m pytest tests/ -q          # 29 passed（PC-3 起，含引用定位 6 项）

# 3) 评测
POLICY_COMPASS_FORCE_MOCK=1 python scripts/eval.py --eval-set data/eval_set_v2.jsonl --out evidence/eval_v2_mock.md
POLICY_COMPASS_FORCE_MOCK=1 python scripts/eval.py --out evidence/eval_v1_mock_regression.md
# real 口径（.env 配 DashScope key；先 --limit 3 冒烟控花费）
python scripts/eval.py --eval-set data/eval_set_v2.jsonl --out evidence/eval_v2_real.md
python scripts/eval.py --out evidence/eval_v1_real.md          # v1 smoke real（覆盖 docs/eval_report.md）

# 3b) PC-2 沪杭贷款真实化回归子集（2026-10-29，只重跑受金标更新影响条目控花费）
POLICY_COMPASS_FORCE_MOCK=1 python scripts/eval.py --eval-set data/eval_set_v2.jsonl --out evidence/eval_v2_mock.md
python scripts/eval.py --eval-set data/eval_loan_regress_v2.jsonl --out evidence/eval_loan_regress_v2.md   # real 10 条（含 holdout #144）
python scripts/eval.py --eval-set data/eval_loan_regress_v1.jsonl --out evidence/eval_loan_regress_v1.md   # real 8 条

# 3c) PC-2 演示文档替换回归（语料变更涉及的 v1 条目真模型抽查）
python scripts/eval.py --eval-set data/eval_corpus_regress_v1.jsonl --out evidence/eval_corpus_regress_v1.md  # real 4 条
python scripts/eval.py --eval-set data/eval_loan_regress_v2b.jsonl --out evidence/eval_loan_regress_v2b.md   # real 2 条（v2 #129/#184）

# 3d) S1 向量腿消融（2026-11-05；embed/hybrid 涉真实 API 须主人批准，见 docs/ablation.md）
POLICY_COMPASS_FORCE_MOCK=1 python scripts/ablation_vector.py --phase bm25           # 零花费基线（CI 门禁同环境）
POLICY_COMPASS_FORCE_MOCK=1 python scripts/ablation_vector.py --phase embed --approved   # 真实重嵌 68 chunk（≈¥0.01）
POLICY_COMPASS_FORCE_MOCK=1 python scripts/ablation_vector.py --phase hybrid         # 混合腿五组消融+e2e（查询向量真 API）
POLICY_COMPASS_FORCE_MOCK=1 python scripts/ablation_vector.py --phase restore_mock   # 恢复 FORCE_MOCK 哈希口径（冻结 mock 复现前提）

# 4) 故障注入（mock，零花费）
python scripts/fault_inject.py --out evidence/fault_injection.md

# 5) 压测（mock，HTTP 全栈，两轮）
python scripts/load_test.py --min-requests 500 --concurrency 20,50,100 --out evidence/load_test.md
python scripts/load_test.py --min-requests 150 --concurrency 20,50,100 --llm-delay 1.5 --out evidence/load_test_delay.md

# 6) 服务（重建索引后如有旧进程需重启）
python scripts/serve.py             # http://127.0.0.1:8000

# 7) 金标验证脚本（评测集防编造：关键词 grep 语料原文 + calc 工具复算）
# 见 git commit feat(test): 的提交说明与 fail_analysis.md §金标修正
```

## ⑥ 已知边界与下一步

- 7 份演示文档仍在库（有明确【演示语料】标记；沪杭贷款+4 份全国/上海文档共 6 份已于 2026-10-29 真实化）；下一步：相关官方文件发布后按同流程替换，剩余演示文档仅作教学演示用途。
- 真实模型评分抖动：v1 real 历史口径 93.9% 中 2 例时效题误拒为评分抖动（见 docs/eval_report.md 历史）；本次 real 复跑数字以 §4 为准，如需更稳可加自校验投票。
- 压测为单副本 uvicorn；生产多 worker/多副本容量需重测。零延迟 mock 的 QPS 饱和点（~45）是 Python 单进程 CPU 上限，不代表真实 LLM 场景（真实场景瓶颈在 LLM 侧，见 load_test.md 两轮对照）。
- 未 push、未部署：git 历史在本机，push/公开部署/试用由用户决定（见 §1 边界）。

### PC-3（2026-11-01）引用原文定位 + mock 基线复测发现

**特性**：chunk 级条款/页码元数据落全链——splitter 记录条款/章节标签（第X条不限长：真实公文"第七条（常见问题）问：……"长条文行实测限长会整批漏标；章节/数字序号保留 30 字防线）+ PDF/OCR 页边界插【pN】标记行、切分时剥离并记账（标记不进正文）；chunks 表加 sections/pages 列 + 旧库打开自动 ALTER 迁移；引用卡升级为"文号 · 生效日期 · 条款 · 现行/废止"。实测 60/68 块带条款标签（如杭州转移指引 chunk=第七条、条例 chunk=第八~十一条）；页码需扫描件 OCR 重跑后产生（现有 OCR 缓存文本无标记，诚实留空）。

**零漂移红线守住**：特性只加元数据、不改任何切分边界——新库与旧库 chunk 正文 dump **逐字节一致**（68 块不变）；pytest **23→29 全绿**（新增 6 测试：条款覆盖/页标记剥离与记账/有标记与无标记语料正文逐字节一致锚定/段落模式标签继承/IndexStore 回程/旧 schema 迁移）。

**复测发现（mock 数字口径，重要）**：A/B 验证中发现最终 HEAD 的 mock 数字与记录口径不符——
- v1 mock：记录 84.9%（28/33）→ 11-01 复跑 **81.8%（27/33）**，4+ 次稳定复现（含 HEAD 代码同库 A/B、PYTHONHASHSEED=0/0/1），失败集差=**#3（fact）**；
- v2 mock：记录 94.4%（85/90）→ 11-01 复跑 **90.0%（81/90）**，新增失败 **#127/#177/#179/#189**（compare/slang/multi）；
- 根因（时间线实证）：评测报告生成于 10-29 07:58，沪杭贷款语料真实化+金标修正的提交落在 **08:02 批次**，08:02 最终 HEAD **未复跑 v1/v2 mock 全量**（real 侧有受影响子集 22 条覆盖，mock 侧遗漏）；#3 答案引"账户余额倍数"旧表述，属金标/语料对齐 2026 文件的连带（与 #6 排序噪声同类）；
- 与本特性无关的证明：chunk 逐字节一致 + HEAD 代码同库 A/B 同为 81.8%；
- 处理选项（留主人定）：①按新口径回填 README/EVIDENCE/RESUME 的 mock 数字（v2 90.0%/v1 81.8%，real 冻结口径不动）；②逐条核对 #3/#127/#177/#179/#189 的金标与 杭房公委〔2026〕1号 等 2026 文件后修复重测。

### PC-HR1（2026-11-09）检索索引热更新：增量摄取 + 快照原子切换 + 后台重建

**特性**（对应面经"RAG 知识库热更新怎么做？文档变了索引怎么同步？"）：①文件内容 sha256 指纹三态增量本就存在（未变跳过/变更重切重嵌/删除级联），本批补齐**服务化闭环**——`POST /api/ingest/reindex` 后台任务（立即 202，singleflight 防叠跑 409）+ `GET /api/ingest/reindex/status` 进度轮询 + CLI `--reindex-incremental` 显式触发面，均走 PC-3 X-Admin-Key 门禁；②**索引版本化收口**：SyncReport 携带 snapshot_version；**零变更不发布新快照**（版本号稳定，BM25 免重建）；③构建完成 `load_cache` 一次性原子发布（PC-2 机制），在途请求旧快照跑完——新增 PC-2 同款回归作用于热更新路径（检索中途整库 sync，断言 top1 仍是旧库块，随后新请求取到新内容）。④**ADR-6**：doc 级增量 patch + 快照原子发布 vs 全量重建/双库蓝绿/无快照原地改（后者即 PC-2 缺陷形态）的取舍记录。

**顺带修掉的真缺陷（评测装置抓出）**：`/api/ingest/sync` 未传 `llm` → 两份扫描件 PDF（hz_extract_optimize_2025.pdf / hz_rent_limit_2023.pdf）解析 ValueError → 叠加旧实现"解析失败=continue→落入 gone 级联删除"——**一次调用静默清掉全部扫描件文档**（singleflight 端点测试首跑抓出 snapshot_version 2→3 暴露）。修复=端点补传 llm + **解析失败≠删除**（磁盘存在的文件瞬时读不出：保留旧版原样可检，errors 留痕；`test_parse_failure_keeps_existing_doc` 钉死）。

**回归与冻结口径**：pytest **40→48 全绿**（+8：指纹零重嵌零发布/修改反映检索/删除级联/在途旧快照隔离/端点鉴权/后台+singleflight/失败不卡死 singleflight/解析失败保留旧版）；冻结口径不受扰动（未动检索打分逻辑与语料）——本节写入后按 HR3 收工全量复测回填终数。

### PC-HR2（2026-11-09）语料投毒/提示注入对抗评测：16/18 打穿 → 修复 → 18/18

**装置**：对抗语料 `data/corpus_adversarial/` 6 份（指令覆盖/假废止/关键词堆砌挤占/伪权威+假覆盖面/外带诱导/角色逃逸），**独立目录**仅评测脚本摄入临时库——正式语料 `data/corpus/` 与冻结评测集 v1/v2 零字节接触；评测集 `data/eval_injection.jsonl` 18 条×4 类（injection_resist 6 / crowd 4 / out_of_scope 4 / exfil 4）；判分全确定性（载荷标记已 grep 核实正式语料零碰撞），零 LLM judge，mock only。运行：`POLICY_COMPASS_FORCE_MOCK=1 python scripts/eval_injection.py`。

**首轮实证（修复前）：16/18 被打穿**——①载荷句凭 mock 生成器的"数字(+2.0)+结论词(+2.5)"加分被抽进答案（hr01-06/cr01-04/oos01-04 全中，9999/8888 进答案）；②伪权威"全国统一额度"把天津/重庆等库外城市问题带偏作答；③**系统提示词原文与安全验证码 9527 被复述外带**（ex01/ex02）；④verify 忠实度校验对该攻击结构性失明（载荷文本就在上下文里——忠实于上下文≠真实，摘出来写进了报告 §5）。

**修复（ADR-7）**：注入隔离落检索出口单一咽喉（app/agent/injection.py + retrieve_node 接线）——命中模式行整行剔除、含载荷行文档 grade 记 0（不可作证据）、生成/校验上下文自动排除、stats 新增 injection_quarantined_lines 观测。模式库 8 族经**双向校验**钉死：正式语料 23 份 md+OCR 缓存全文零命中（冻结安全守门测试）+ 对抗 6 份全命中（装置有效性测试）。

**修复后：18/18**——带偏 0、编造/外带 0、out_of_scope 诚实拒答 4/4、载荷检索穿透 0（sources 无对抗块）。**残余风险如实留档（报告 §4）**：关键词堆砌毒块可把真实证据挤出 top-k → 真实问题诚实拒答（真实答案存活率 0/10，最坏构造）——可用性 DoS 的修复归属检索层信任降权，违反本批冻结红线（不碰检索打分），归属后续批次；写入面已有 X-Admin-Key 门禁（PC-3）兜底。

**冻结口径**：消毒器在检索热路径上，修复后全量复测 v1 mock **84.9%** / v2 mock **91.1%** / recall@5 97.9·97.2（93.8）**逐字复现**，usage 签名（refusals 4/11、tool_calls 6/19）与基线一致——正式语料经消毒路径文本逐字节不变。pytest **48→52 全绿**（+4：语料零命中守门/对抗全命中/消毒器单测/端到端防带偏）。

### PC-HR3（2026-11-09）引用 faithfulness 自动校验器：实体级回查 + 20 条校准 + v1/v2 全量

**装置**（app/eval/faithfulness.py + scripts/eval_faithfulness.py，纯离线、agent 链路零改动）：对答案中每个**文号/条款号/数量**原子回查证据集=引用 chunk 原文+PC-3 sections 条款元数据+引用 source 文号元数据+工具计算结果（calc 题数字来自工具，不计入则系统性误报）；与 verify 节点的句子级 bigram 校验互补（实体级 vs 句子级）。排除规则：公历年份不入原子、文号先掩码再提数量。归因三分类：生成幻觉（全库未见）/引用错块（库内他块）/校验器误报（校准裁决）。

**20 条校准子集（人工核对）真实产出——抓到校验器自身 1 类误报并修复**：①非拒答答案 16 条全部原子可溯源（文号/条款/数量 0 误报 0 漏报，工具数字经证据池正确支持）；②拒答答案暴露口径缺陷——fallback 拒答文案按设计回显"最接近文件"的文号（透明性提示、非事实断言、无引用来源），首跑被误计 14 条未支持 → 修法=拒答答案不入原子评分（与分母口径一致）；③校验器开发期自测另抓 3 个真 bug：文号正则吞"根据/依据"语流前缀（改后缀针判定）、sections"第七条"整串过数字归一返 None（剥"第/条"再归一）、数量匹配缺位数边界（"2000元"不得命中"12000元"，(?<!\d) 环视钉死）——均由 tests/test_faithfulness.py 回归钉住。

**全量结果（mock 口径）**：v1 **100.0%**（277/277 原子）/ v2 **100.0%**（599/599），未支持 0——mock 抽取式答案按构造可溯源，本指标的价值在校验器装置本身（其判别力已在校准中实证）与真模型口径的可扩展性（数字改写形态/跨句推断，如实标注未测）。运行：`POLICY_COMPASS_FORCE_MOCK=1 python scripts/eval_faithfulness.py`。

**诚实边界**：数量匹配为字面形态匹配（90万元↔90万 归一），不做数值等价（"2万"vs"20000元"不识别）；真模型口径未测。
