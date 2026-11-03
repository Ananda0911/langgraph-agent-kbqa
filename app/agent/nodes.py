"""Agent 节点实现：意图识别 / 自主规划 / 工具执行 / 反思评估 / 回答生成。

每个节点为纯函数式：读 AgentState → 返回增量更新 dict。
LLM 不可用时全部节点可降级（关键词规则 / 抽取式回答），保证无 Key 演示链路。
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.agent.action_memory import ActionMemory
from app.agent.state import AgentState
from app.agent.tools.base import ToolRegistry

logger = logging.getLogger(__name__)

INTENT_SYSTEM = "你是一个精准的意图分类器，只输出 JSON，不输出任何其他文本。"


def _extract_json(text: str) -> dict | None:
    """从 LLM 输出中鲁棒地提取第一个 JSON 对象。"""
    if not text:
        return None
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    raw = m.group(0)
    for attempt in (raw, raw.replace("'", '"')):
        try:
            obj = json.loads(attempt)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


class AgentNodes:
    """LangGraph 各节点的实现（依赖注入：LLM / 工具注册表 / 行动记忆）。"""

    def __init__(self, llm, tools: ToolRegistry, action_memory: ActionMemory, agent_cfg: dict):
        self.llm = llm
        self.tools = tools
        self.am = action_memory
        self.max_iterations = int(agent_cfg.get("max_iterations", 3))
        self.reflection_enabled = bool(agent_cfg.get("reflection", True))
        self.max_plan_steps = int(agent_cfg.get("max_plan_steps", 6))
        self.model_override: str | None = None  # 多模型切换：当前使用的 model id

    # ---------- 公共 ----------

    def _llm_chat(self, system: str, user: str, temperature: float = 0.2) -> str:
        kwargs: dict[str, Any] = {"temperature": temperature}
        if self.model_override:
            kwargs["model"] = self.model_override
        return self.llm.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}], **kwargs
        )

    def _trace(self, state: dict, etype: str, content: Any, meta: dict | None = None) -> list:
        entry = {"ts_type": etype, "content": content, "meta": meta or {}}
        state["trace"] = state.get("trace", []) + [entry]
        self.am.record(state["session_id"], etype, content, meta)
        return state["trace"]

    # ---------- 1) 意图识别 ----------

    # 关键词规则（LLM 不可用 / 输出解析失败时降级）
    KEYWORD_RULES: list[tuple[str, str]] = [
        (r"(联系方式|电话|联系[\s\S]{0,6}谁|通讯录|邮箱是多少|找[\s\S]{0,6}人)", "mcp_contacts"),
        (r"(最新|新闻|今天|近期|现在|当前|202[5-6]年|时事|天气|股价|汇率|联网|网上)", "web_search"),
        (r"(订单|销售额|销售数据|营业额|成交|回款|多少单|数据库查询)", "db_query"),
        (r"(员工|部门|人数|同事|入职|花名册|在职)", "db_query"),
        (r"(计算|求和|平均|百分比|增长率|折算|=|\d+\s*[\+\-\*/]\s*\d+)", "calculator"),
    ]
    KB_HINT = r"(制度|条款|报销|保险|研报|文档|政策|规定|说明|流程|产品|权益|理赔)"

    def _keyword_intent(self, query: str) -> dict:
        matched: dict[str, None] = {}
        for pat, tool in self.KEYWORD_RULES:
            if re.search(pat, query):
                matched[tool] = None
        if not matched:
            # 默认走知识库检索（企业问答主路径）
            return {"type": "kb_search", "tools": ["kb_search"], "reason": "默认路由至知识库检索", "need_plan": False}
        tools = list(matched)
        if len(tools) == 1:
            t = tools[0]
            return {"type": t, "tools": [t], "reason": f"关键词命中 {t}", "need_plan": False}
        return {"type": "multi", "tools": tools, "reason": "命中多类关键词，需多步规划", "need_plan": True}

    def intent_node(self, state: dict) -> dict:
        query = state["query"]
        intent: dict | None = None
        if self.llm.available:
            try:
                prompt = (
                    f"可用工具清单：\n{self.tools.prompt()}\n\n"
                    f"用户问题：{query}\n\n"
                    "请判断回答该问题需要哪些工具，输出 JSON（禁止其他文本）：\n"
                    '{"type": "kb_search|db_query|web_search|calculator|mcp_contacts|multi|chat", '
                    '"tools": ["工具名"], "reason": "简短理由", "need_plan": true/false}\n'
                    "规则: 纯闲聊/问候/无需外部信息 → type=chat, tools=[]；"
                    "需要多个工具或多个步骤 → type=multi, need_plan=true；"
                    "单一工具即可 → tools 只含该工具, need_plan=false。\n"
                    "路由要点：员工人数/销售额/订单/客户/统计汇总等数据问题 → db_query；"
                    "查找具体某人的联系方式/电话/工位 → mcp_contacts；"
                    "企业制度/条款/研报/文档内容 → kb_search；"
                    "最新时事/实时资讯 → web_search；算术计算 → calculator。"
                )
                obj = _extract_json(self._llm_chat(INTENT_SYSTEM, prompt, temperature=0.1))
                if obj:
                    valid = {t for t in obj.get("tools", []) if self.tools.get(t)}
                    itype = obj.get("type", "")
                    if itype == "chat" or valid:
                        intent = {
                            "type": itype,
                            "tools": sorted(valid) if valid else [],
                            "reason": str(obj.get("reason", ""))[:100],
                            "need_plan": bool(obj.get("need_plan")) or itype == "multi",
                        }
            except Exception as e:
                logger.warning("意图识别 LLM 调用失败，降级关键词规则: %s", e)
        if intent is None:
            intent = self._keyword_intent(query)

        updates: dict[str, Any] = {"intent": intent}
        self._trace(state, "intent", {"type": intent["type"], "tools": intent["tools"], "reason": intent["reason"]})

        # 单工具：直接在意图节点生成一步计划（无需 planner）
        if not intent.get("need_plan") and intent["tools"]:
            tool = intent["tools"][0]
            updates["plan"] = [{"description": query, "tool": tool, "args": self._default_args(tool, query)}]
            updates["step_index"] = 0
        updates["trace"] = state["trace"]
        return updates

    def _default_args(self, tool: str, query: str) -> dict:
        """单工具路由时的默认参数。"""
        if tool == "calculator":
            m = re.search(r"[-+*/().\d\s%]{3,}", query)
            return {"expression": m.group(0).strip() if m else query}
        if tool == "db_query":
            sql = "SELECT * FROM orders" if re.search(r"订单|销售|客户", query) else "SELECT * FROM employees"
            return {"sql": sql}
        if tool == "mcp_contacts":
            kw = re.sub(r"(查询|查找|搜索|请问|一下|的|联系方式|电话|邮箱|联系)", "", query).strip()
            return {"tool": "search_contacts", "arguments": {"keyword": kw or query}}
        return {"query": query}

    # ---------- 2) 自主规划 ----------

    def planner_node(self, state: dict) -> dict:
        query = state["query"]
        intent = state.get("intent", {})
        plan: list[dict] = []
        if self.llm.available:
            try:
                prompt = (
                    f"可用工具清单：\n{self.tools.prompt()}\n\n"
                    f"用户问题：{query}\n意图：{json.dumps(intent, ensure_ascii=False)}\n"
                    f"历史行动记忆：\n{self.am.recent_context(state['session_id']) or '（无）'}\n\n"
                    "请将问题拆解为按顺序执行的工具调用计划，输出 JSON（禁止其他文本）：\n"
                    '{"steps": [{"description": "步骤说明", "tool": "工具名", "args": {工具参数}}]}\n'
                    f"最多 {self.max_plan_steps} 步，每步只用一个工具，参数必须符合工具要求。"
                )
                obj = _extract_json(self._llm_chat(INTENT_SYSTEM, prompt, temperature=0.2))
                if obj:
                    for s in obj.get("steps", [])[: self.max_plan_steps]:
                        tool = s.get("tool")
                        if self.tools.get(tool):
                            plan.append({"description": str(s.get("description", ""))[:100], "tool": tool, "args": s.get("args") or {}})
            except Exception as e:
                logger.warning("规划节点 LLM 调用失败，降级默认计划: %s", e)
        if not plan:
            # 降级：每个命中工具生成一步默认调用
            for t in intent.get("tools", []):
                if self.tools.get(t):
                    plan.append({"description": f"调用 {t}", "tool": t, "args": self._default_args(t, query)})
        self._trace(state, "plan", plan)
        return {"plan": plan, "step_index": 0, "trace": state["trace"]}

    # ---------- 3) 工具执行 ----------

    def executor_node(self, state: dict) -> dict:
        plan = state.get("plan", [])
        idx = state.get("step_index", 0)
        if idx >= len(plan):
            return {}
        step = plan[idx]
        tool = self.tools.get(step["tool"])
        label = f"{step['tool']}({json.dumps(step.get('args', {}), ensure_ascii=False)[:80]})"
        self._trace(state, "tool_call", {"step": idx + 1, "tool": step["tool"], "args": step.get("args", {})})

        obs: dict[str, Any]
        if tool is None:
            obs = {"step": idx + 1, "tool": step["tool"], "args": step.get("args", {}),
                   "output": f"工具 {step['tool']} 不存在", "sources": [], "ok": False}
        else:
            try:
                result = tool.run(step.get("args") or {})
                obs = {
                    "step": idx + 1, "tool": step["tool"], "args": step.get("args", {}),
                    "output": str(result.get("output", "")), "sources": result.get("sources", []),
                    "ok": not result.get("error"),
                }
            except Exception as e:
                logger.exception("工具 %s 执行异常", step["tool"])
                obs = {"step": idx + 1, "tool": step["tool"], "args": step.get("args", {}),
                       "output": f"工具执行异常: {e}", "sources": [], "ok": False}
        step["status"] = "done" if obs["ok"] else "failed"
        self._trace(state, "observation", obs["output"][:200], {"step": idx + 1, "tool": obs["tool"], "ok": obs["ok"]})
        return {
            "observations": state.get("observations", []) + [obs],
            "step_index": idx + 1,
            "plan": state.get("plan", []),
            "trace": state["trace"],
        }

    # ---------- 4) 反思评估 ----------

    def reflection_node(self, state: dict) -> dict:
        enabled = self.reflection_enabled and self.llm.available and state.get("observations")
        reflection = {"pass": True, "feedback": "", "extra_steps": []}
        if enabled:
            all_ok = all(o["ok"] for o in state["observations"])
            try:
                obs_text = "\n\n".join(
                    f"[步骤{o['step']} {o['tool']} {'成功' if o['ok'] else '失败'}]\n{o['output'][:600]}"
                    for o in state["observations"]
                )
                prompt = (
                    f"用户问题：{state['query']}\n\n已有工具观察结果：\n{obs_text}\n\n"
                    "请评估现有信息是否足以完整回答用户问题。输出 JSON（禁止其他文本）：\n"
                    '{"pass": true/false, "feedback": "评估意见", '
                    '"extra_steps": [{"description": "...", "tool": "工具名", "args": {...}}]}\n'
                    "若信息足够 → pass=true, extra_steps=[]；若不足 → pass=false 并给出补充步骤（最多2步）。\n"
                    "注意：观察结果显示「未找到」「无相关内容」或明显与问题不符时，必须 pass=false，"
                    "并在 extra_steps 中改用正确的工具（如数据统计问题应改用 db_query）。"
                )
                obj = _extract_json(self._llm_chat(INTENT_SYSTEM, prompt, temperature=0.1))
                if obj:
                    extra = []
                    for s in (obj.get("extra_steps") or [])[:2]:
                        if self.tools.get(s.get("tool")):
                            extra.append({"description": str(s.get("description", ""))[:100],
                                          "tool": s["tool"], "args": s.get("args") or {}})
                    reflection = {
                        "pass": bool(obj.get("pass", True)) and all_ok,
                        "feedback": str(obj.get("feedback", ""))[:150],
                        "extra_steps": extra,
                    }
            except Exception as e:
                logger.warning("反思节点 LLM 调用失败，默认通过: %s", e)
        elif state.get("observations"):
            reflection = {"pass": all(o["ok"] for o in state["observations"]), "feedback": "", "extra_steps": []}

        updates: dict[str, Any] = {"reflection": reflection}
        self._trace(state, "reflection", {"pass": reflection["pass"], "feedback": reflection["feedback"]})

        # 不通过且未达上限：把补充步骤追加进计划，回到执行节点
        if (not reflection["pass"]) and reflection["extra_steps"] and state.get("iterations", 0) < self.max_iterations:
            updates["iterations"] = state.get("iterations", 0) + 1
            updates["plan"] = state.get("plan", []) + reflection["extra_steps"]
            self._trace(state, "replan", [s["tool"] for s in reflection["extra_steps"]])
        updates["trace"] = state["trace"]
        return updates

    # ---------- 5) 回答生成 ----------

    ANSWER_SYSTEM = "你是企业智能助手，基于给定的工具观察结果回答问题，引用事实时注明来源（文档用《文档名》，其他标明工具名）。回答简洁、专业、结构化。不要编造观察结果中没有的信息。"

    def answer_node(self, state: dict) -> dict:
        query = state["query"]
        observations = state.get("observations", [])
        sources: list[dict] = []
        for o in observations:
            sources.extend(o.get("sources", []))

        if not observations:
            # 纯对话（chat 意图）：多轮对话管理，直接 LLM 应答
            answer = ""
            if self.llm.available:
                try:
                    ctx = state.get("memory_ctx", "")
                    user = f"（对话记忆）\n{ctx}\n\n用户：{query}" if ctx else query
                    answer = self._llm_chat(self.ANSWER_SYSTEM, user, temperature=0.5)
                except Exception as e:
                    logger.warning("对话回答生成失败: %s", e)
            answer = answer or f"你好！我是企业智能助手，可以帮你：\n1. 查询企业知识库文档（制度/研报/条款）\n2. 查询企业数据库（员工/订单数据）\n3. 联网搜索最新资讯\n4. 查企业通讯录（MCP 服务）\n5. 数学计算\n\n当前为 Mock 模式，配置 llm.api_key 后可自由对话。"
            self._trace(state, "answer", answer[:200])
            return {"final_answer": answer, "sources": sources, "trace": state["trace"]}

        obs_text = "\n\n".join(
            f"[步骤{o['step']} · 工具 {o['tool']}\n{o['output'][:1500]}]" for o in observations
        )
        answer = ""
        if self.llm.available:
            try:
                ctx = state.get("memory_ctx", "")
                user = (
                    f"（对话记忆，供理解指代）\n{ctx}\n\n" if ctx else ""
                ) + f"用户问题：{query}\n\n工具观察结果：\n{obs_text}\n\n请综合以上观察结果回答用户问题。"
                answer = self._llm_chat(self.ANSWER_SYSTEM, user, temperature=0.3)
            except Exception as e:
                logger.warning("回答生成失败，降级抽取式: %s", e)
        if not answer:
            # Mock 降级：抽取观察结果要点
            answer = "根据工具执行结果，要点如下：\n"
            for o in observations:
                head = " ".join(o["output"].split())[:150]
                answer += f"- [{o['tool']}] {head}\n"
            answer += "\n（Mock 模式：配置 llm.api_key 后可获得完整生成回答。）"
        self._trace(state, "answer", answer[:200])
        return {"final_answer": answer, "sources": sources, "trace": state["trace"]}
