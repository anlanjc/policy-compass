"""确定性 Mock LLM：无 API key 环境下跑通全链路（CI、演示、开发自测）。

设计要点：
- 继承 LLMClient，只覆写 chat / chat_stream / embed —— chat_json 的
  "解析失败→纠错重试→pydantic 校验"链路在 mock 下照常执行，不是被绕过；
- 调用方通过 system 消息首行 `#task=route|grade|rewrite|tool|verify|generate|refuse`
  声明任务类型，mock 按 task 做确定性行为（词面重叠打分、词典改写、数字抽取等），
  这些启发式与真实模型的职责对齐，保证 mock 下的行为可预测、测试可断言；
- mock embedding 采用分词哈希映射（词袋 → 定维向量）：向量路在 mock 下退化为
  词面匹配，检索质量主要靠 BM25 支撑 —— README 已说明，真实质量评测请用真实 embedding。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import AsyncIterator, Sequence

import numpy as np

from ..config import settings
from .client import LLMClient


def _lexicon() -> dict[str, list[str]]:
    path = Path(settings.lexicon_path)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


class MockLLMClient(LLMClient):
    def __init__(self) -> None:
        self._lex = _lexicon()

    # ---------- 接口覆写 ----------

    async def chat(
        self,
        messages: Sequence[dict],
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> str:
        if settings.llm_mock_delay > 0:
            await asyncio.sleep(settings.llm_mock_delay)  # 压测：模拟真实 LLM 延迟形态
        # chat_json 会追加 schema system 消息，#task 标记可能在任意 system 中
        system_blob = "".join(
            m["content"] for m in messages if m["role"] == "system"
        )
        user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        m = re.search(r"#task=(\w+)", system_blob)
        task = m.group(1) if m else "generate"

        if task == "route":
            return self._route(user)
        if task == "grade":
            return self._grade(user)
        if task == "rewrite":
            return self._rewrite(user)
        if task == "tool":
            return self._tool(user)
        if task == "verify":
            return self._verify(user)
        if task == "refuse":
            return "很抱歉，知识库中没有收录与您问题相关的现行政策文件。为避免提供错误的政策信息，我无法回答该问题。您可以换一个与社保、公积金、医保相关的问题，或咨询当地 12329 / 12333 热线。"
        return self._generate(user)

    async def chat_stream(
        self, messages: Sequence[dict], temperature: float = 0.3
    ) -> AsyncIterator[str]:
        text = await self.chat(messages, temperature)
        for i in range(0, len(text), 4):
            yield text[i : i + 4]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import jieba

        # 维度必须与 settings.embed_dim 一致：索引库向量宽度按 embed_dim 建，
        # mock 曾固定 256 维，对真实链路摄取的库（关闭 embedding 时落 embed_dim 维零向量）
        # 做查询会因维度不匹配直接崩溃（v2 mock 评测实测发现并修复）
        dim = settings.embed_dim
        out: list[list[float]] = []
        for text in texts:
            vec = np.zeros(dim)
            tokens = [t for t in jieba.lcut(text) if t.strip()]
            grams = tokens + [a + b for a, b in zip(tokens, tokens[1:])]
            for g in grams:
                h = int(hashlib.md5(g.encode("utf-8")).hexdigest(), 16)
                vec[h % dim] += 1.0
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec /= norm
            out.append(vec.tolist())
        return out

    async def close(self) -> None:  # pragma: no cover
        pass

    # ---------- 确定性任务实现 ----------

    def _route(self, user: str) -> str:
        hits = [c for c in settings.cities if c in user and c != "全国"]
        city = hits[0] if len(hits) == 1 else "全国"
        domain = "通用"
        for d in ("公积金", "社保", "医保"):
            if d in user:
                domain = d
                break
        calc = bool(
            re.search(r"\d", user)
            and re.search(r"贷|额度|余额|月薪|提取|多少", user)
        )
        sub: list[str] = []
        if "和" in user and all(x in user for x in ("杭州", "上海")) or (
            "对比" in user or "区别" in user or "哪个" in user
        ):
            sub = [user]
        return json.dumps(
            {"city": city, "domain": domain, "need_calc": calc, "sub_questions": sub},
            ensure_ascii=False,
        )

    STOP_WORDS = {
        "怎么", "什么", "如何", "办理", "需要", "可以", "分别", "现在", "最新",
        "最多", "能够", "哪些", "以及", "具体", "包括", "还是", "还是", "政策",
    }

    def _expand(self, text: str) -> list[str]:
        """黑话词典展开（grade/generate 共用）。"""
        out = []
        for slang, formal in self._lex.items():
            if slang in text:
                out.extend(formal)
        return out

    def _grade(self, user: str) -> str:
        import jieba

        q_part = user.split("待评文档")[0]
        question = re.sub(r"问题[：:]", "", q_part).strip()
        doc = re.split(r"待评文档[：:]", user, 1)[-1] if "待评文档" in user else user
        stop = self.STOP_WORDS
        q_tokens = {
            t for t in jieba.lcut(question)
            if len(t) >= 2 and t not in stop
        }
        d_tokens = {t for t in jieba.lcut(doc) if len(t) >= 2}
        overlap = len(q_tokens & d_tokens) / max(len(q_tokens), 1)
        # 标题事项名锚定：政策标题=事项名，问题实体词若与标题零重叠，
        # 说明只是正文顺带共现（如"个人账户"之于转移指南），强降分。
        m_title = re.search(r"《[^》]+》", doc)
        title_line = m_title.group(0) if m_title else ""
        title_grams = {
            title_line[i : i + 2] for i in range(len(title_line) - 1)
            if re.match(r"[\u4e00-\u9fff]{2}", title_line[i : i + 2])
        }
        q_grams = {
            question[i : i + 2] for i in range(len(question) - 1)
            if re.match(r"[\u4e00-\u9fff]{2}", question[i : i + 2])
        } | {g for w in self._expand(question) for g in
             [w[i : i + 2] for i in range(len(w) - 1)]}
        city_words = set(getattr(settings, "cities", []))
        anchored = len((q_grams - city_words) & title_grams) >= 1
        if overlap > 0.30:
            score = 0.95
        elif overlap > 0.18:
            score = 0.75
        elif overlap > 0.10:
            score = 0.5
        elif overlap > 0.05:
            score = 0.25
        else:
            score = 0.05
        if not anchored:
            score = min(score, 0.25)
        return json.dumps(
            {"score": score,
             "reason": f"实体词重叠率 {overlap:.2f}，标题锚定 {'命中' if anchored else '未命中'}"},
            ensure_ascii=False,
        )

    def _rewrite(self, user: str) -> str:
        q = user.replace("用户原始问题：", "").strip().splitlines()[0]
        parts = [q]
        expanded = False
        for slang, formal in self._lex.items():
            if slang in q:
                parts.append("、".join(formal))
                expanded = True
        if expanded:
            parts.append("政策 标准 规定")
        return json.dumps(
            {
                "rewritten": " ".join(parts),
                "expanded_terms": [s for s in self._lex if s in q],
            },
            ensure_ascii=False,
        )

    def _tool(self, user: str) -> str:
        # 默认"全国"而非猜测某城：无城市线索时工具返回"未收录"才是诚实行为
        # （故障注入实测：默认杭州会把杭州额度当作库外城市的答案）
        city = next((c for c in settings.cities if c in user and c != "全国"), "全国")
        if re.search(r"租房|房租|提取", user) and not re.search(r"贷款|能贷", user):
            args: dict = {"city": city}
            m = re.search(r"(无房|没有房)", user)
            args["has_house"] = bool(m)
            marital = re.search(r"(已婚|配偶|夫妻|家庭)", user)
            args["is_family"] = bool(marital)
            return json.dumps(
                {"tool": "rent_extract_limit", "args": args}, ensure_ascii=False
            )
        balance = None
        m = re.search(r"余额(?:有)?\s*(\d+(?:\.\d+)?)\s*万", user)
        if m:
            balance = float(m.group(1)) * 10000
        else:
            # "账户合计12万" / "账户有6万" / "公积金有8万" 等邻近表述
            m = re.search(
                r"(?:账户|合计|公积金)[^0-9]{0,6}(\d+(?:\.\d+)?)\s*万", user
            )
            if m:
                balance = float(m.group(1)) * 10000
            else:
                nums = re.findall(r"(\d+(?:\.\d+)?)\s*万", user)
                if nums:
                    balance = float(nums[0]) * 10000
        m2 = re.search(r"月薪(?:有)?\s*(\d+(?:\.\d+)?)\s*万", user)
        income = float(m2.group(1)) * 10000 if m2 else 20000.0
        m3 = re.search(r"缴(?:存|纳)?(?:了)?\s*(\d+)\s*年", user)
        years = int(m3.group(1)) if m3 else None
        m4 = re.search(r"(?:夫妻|配偶|两人|双方|我和爱人|都缴)", user)
        args = {
            "city": city,
            "balance": balance,
            "monthly_income": income,
            "years": years,
            "is_family": bool(m4),
        }
        return json.dumps(
            {"tool": "housing_loan_quota", "args": args}, ensure_ascii=False
        )

    @staticmethod
    def _grams(text: str) -> set[str]:
        """字符 bigram 集合（去空白）。

        不用 jieba 词级匹配的原因（实测踩坑）：同一字符串在孤立短句与长文本中
        切分结果不同（"杭房公委〔2025〕4号"孤立切为 杭房/公委，长行中可能是
        杭房公委），词集交集为空导致原文引用被误判无出处；bigram 与上下文无关。
        """
        s = re.sub(r"\s", "", text)
        return {s[i : i + 2] for i in range(len(s) - 1)}

    def _verify(self, user: str) -> str:
        answer = re.split(r"候选回答[：:]", user, 1)[-1] if "候选回答" in user else ""
        context = user.split("候选回答")[0]
        # 政策场景的忠实度定义：实质性数字/额度结论必须有出处，
        # 过渡性表述（"根据文件""为您整理"）与程序性兜底话术不在校验范围。
        substantial = re.compile(r"[\d〇一二三四五六七八九十百]+|元|万|倍|条|个月|工作日")
        boilerplate = re.compile(r"热线|以.{0,12}为准|咨询|仅供参考")
        sentences = [
            s
            for s in re.split(r"[。；\n]", answer)
            if len(s) >= 8 and substantial.search(s) and not boilerplate.search(s)
        ]
        unsupported = []
        # 不设最短行过滤：OCR 公文中"ZJAC16-2025-0302"、文号行等关键出处
        # 恰恰是短行（曾被 >20 过滤丢弃导致原文引用被误判无出处）；
        # 0.6 的 bigram 重叠比率对短行同样是强判据，不会误放行
        ctx_lines = [c for c in context.split("\n") if len(c) >= 6]
        ctx_grams = [self._grams(c) for c in ctx_lines]
        for s in sentences:
            grams = self._grams(s)
            if not grams:
                continue
            # 与任一上下文行的 bigram 重叠率 ≥ 0.6 视为有出处
            supported = any(
                len(grams & cg) / len(grams) >= 0.6 for cg in ctx_grams if cg
            )
            if not supported:
                unsupported.append(s[:30])
        return json.dumps(
            {"grounded": len(unsupported) == 0, "unsupported": unsupported},
            ensure_ascii=False,
        )

    def _generate(self, user: str) -> str:
        """抽取式生成：按查询实体词相关性+数字密度选关键句，标注来源编号。

        与真实模型职责对齐（基于 context 回答），保证 verify 分支可被触发与测试。
        """
        import jieba

        question = ""
        m = re.search(r"【用户问题】(.+)", user)
        if m:
            question = m.group(1)
        # 黑话词典扩展：让"断了/带走"等口语词能匹配规范条文句
        expanded = []
        for slang, formal in self._lex.items():
            if slang in question:
                expanded.extend(formal)
        q_text = question + " " + " ".join(expanded)
        q_grams = {
            question[i : i + 2]
            for i in range(len(question) - 1)
            if re.match(r"[\u4e00-\u9fff]{2}", question[i : i + 2])
        } | {g for w in expanded for g in [w[i : i + 2] for i in range(len(w) - 1)]}
        ctx_block = user.split("参考资料", 1)[-1] if "参考资料" in user else user
        blocks: dict[int, list[str]] = {}
        current = 0
        for line in ctx_block.split("\n"):
            if line.strip().startswith("【用户问题】"):
                break  # prompt 尾部的用户问题回显不是资料内容
            mm = re.match(r"^\[(\d+)\]\s*(.*)", line.strip())
            if mm:
                current = int(mm.group(1))
                blocks.setdefault(current, []).append(mm.group(2))
            elif current and line.strip():
                blocks[current].append(line.strip())
        chunks = {i: ls for i, ls in blocks.items() if sum(len(x) for x in ls) > 30}
        if not chunks:
            return "根据现有政策文件，未找到与该问题直接相关的条文。"

        def sent_score(s: str) -> float:
            s_grams = {
                s[i : i + 2] for i in range(len(s) - 1)
                if re.match(r"[\u4e00-\u9fff]{2}", s[i : i + 2])
            }
            overlap = len(q_grams & s_grams)
            num_bonus = 2.0 if re.search(r"\d", s) else 0.0
            # 政策答案句的典型形态：含结论性标志词 + 数字
            concl_bonus = 2.5 if re.search(
                r"最高|不超过|合计|限额|额度为|上限|不低于|倍|满\d|起施行|起试行|可贷", s
            ) else 0.0
            return overlap + num_bonus + concl_bonus

        parts = ["根据现行政策文件，为您整理如下："]
        for i, lines in list(chunks.items())[:3]:
            content_lines = [
                l for l in lines
                if not l.startswith("《") and "现行有效" not in l and "已废止" not in l
                and not l.startswith("【")  # 语料元数据行（来源URL/抓取时间等）不得进答案
                and not re.match(r"^[-—－]\s*\d+\s*[-—－]$", l.strip())  # OCR 页脚
            ]
            sentences = [
                s for s in re.split(r"[。]", "\n".join(content_lines)) if len(s.strip()) >= 10
            ]
            ranked = sorted(sentences, key=sent_score, reverse=True)
            digest = "。".join(ranked[:2])
            parts.append(f"{digest}。[{i}]")
        parts.append(
            "以上内容摘自知识库现行有效的政策文件，具体办理以当地公积金/社保中心为准。"
        )
        return "\n".join(parts)
        parts.append(
            "以上内容摘自知识库现行有效的政策文件，具体办理以当地公积金/社保中心为准。"
        )
        return "\n".join(parts)
