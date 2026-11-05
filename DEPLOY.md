# PolicyCompass 部署手册（公开演示版）

> 目标：把 mock 版公开部署为一个可分享的链接（零 API 成本），用于简历/面试"可现场点开"的证据。
> 真实模型版部署方式相同，只需在平台环境变量里配 key（注意费用与限流）。
> 与 askbi/DEPLOY.md 同构；PC 特有项（语料入镜像、向量/重排开关）见下文。

## 方案 A：Render（推荐，免费档够用）

1. 把仓库推到 GitHub（见下方"前置"）。
2. Render 控制台 → New → Web Service → 连接仓库。
3. 配置：
   - Runtime: **Docker**（仓库根有 Dockerfile，自动识别）
   - Instance Type: Free
   - Environment Variables: `LLM_MODE=mock`（显式写上，防止镜像内残留配置意外调真模型）
4. 部署后拿到 `https://<app>.onrender.com`，打开即是 Web 工作台。
5. 验证：问"北京租房提取公积金每月最多能取多少"→ 出答案 + 文号引用（现行/废止徽章）+ SSE trace 面板逐节点推送。

**注意**：
- Free 档 15 分钟无流量会休眠，首次唤醒要 30-60 秒——演示前先自己点开一次。
- **端口**：Render 的 Docker 运行时默认期望应用监听 `PORT`（10000）；本项目默认 `APP_PORT=8000`，在平台环境变量里加一行 `APP_PORT=10000` 即可对齐（pydantic Settings 自动读取）。
- **语料已在镜像里**：Dockerfile `COPY data ./data` 把 16 份真实语料（含 OCR 缓存）打进镜像，首次启动自动增量摄取（BM25 索引，秒级完成），无需挂载外部存储。
- **公网部署必读（S3 评审 PC-3 后的新默认）**：知识库写接口（上传/删除/重摄取）**默认整体禁用**（403）；需要 API 热更新时配置 `POLICY_COMPASS_ADMIN_KEY=<随机长串>`，调用方带 `X-Admin-Key` 头。跨域前端另配 `POLICY_COMPASS_CORS_ORIGINS`（JSON 数组，默认仅同源）。问答/健康/统计等读接口无需密钥。

## 方案 B：Railway / Fly.io

- Railway：New Project → Deploy from Repo → 自动识别 Dockerfile；端口自动探测，无需改 `APP_PORT`。
- Fly.io：`fly launch`（检测 Dockerfile）→ `fly deploy`。免费额度需绑卡，介意可略过。

## 前置（一次性）

```bash
cd policy-compass
git remote add origin git@github.com:<user>/policy-compass.git
git push -u origin main        # 分支已统一 main，ci.yml 的 push 触发器即生效
```

## 环境变量清单（Docker/平台通用）

| 变量 | 默认 | 说明 |
|---|---|---|
| `LLM_MODE` | `auto` | `auto`=有 key 用 real、无 key 自动 mock；可强制 `mock`/`real` |
| `POLICY_COMPASS_FORCE_MOCK` | - | 置 `1` 强制 mock（CI 口径；与上一行二选一即可） |
| `LLM_API_BASE` / `LLM_API_KEY` / `LLM_MODEL` | deepseek 三件套 | 真模型配置（OpenAI 兼容协议，DashScope/DeepSeek/硅基流动均可） |
| `EMBED_ENABLED` | `true` | 无 embedding 额度置 `false`：向量路休眠，退化为 BM25+业务加权（本地 68 chunks 消融 95.9% 口径） |
| `RERANK_ENABLED` | `true` | 无 rerank key 置 `false`：失败本来也会自动降级，显式关掉更干净 |
| `APP_HOST` / `APP_PORT` | `0.0.0.0` / `8000` | 平台要求特定端口时改 `APP_PORT` |
| `LLM_TRUST_ENV` | `false` | 默认不读系统代理（Windows 注册表半死代理会让 TLS 握手失败，实测踩坑）；部署环境需要走代理时显式置 `true` |
| `LLM_MOCK_DELAY` | `0` | mock 模式下模拟 LLM 延迟秒数（演示 SSE trace 推送节奏可设 0.3-0.5，压测用 1.5） |

## 部署后的证据动作（很重要，部署本身不是目的）

1. 链接写进简历项目行与 README 顶部（`在线演示（mock）`徽章）。
2. 找 5-10 个同学每人问 10 个真问题，按试点模板记录成功率/失败案例——**政策问法的长尾（各地方言/黑话表述）只能从真人提问里来**，试用问题回流评测集 v3 的价值比 askbi 更高（见 PRODUCTION_GAP §2-1）。
3. 用真实 HTTP 链路复跑 `scripts/load_test.py`，把压测口径从"本地进程全栈 QPS 45-49"补一条"公网实测"——两个数字是两个问题（编排上限 vs 端到端），Render 免费档 CPU 弱、公网数字必然低，提前想好双口径叙事，不是退步。
4. 政策语料更新演示：`git pull` 换一版语料重新部署，配合 EVIDENCE 里"废止文件强降权"叙事讲热更新链路（hash 增量索引已有）。
