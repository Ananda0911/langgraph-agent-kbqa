"""LangGraph 组装：意图 → 规划 → 执行循环 → 反思 → 回答。

优先使用官方 langgraph.StateGraph；若环境中未安装则使用内置的
MiniGraph 等价实现（相同 API 子集），保证零依赖降级可运行。
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from app.agent.nodes import AgentNodes
from app.agent.state import AgentState

logger = logging.getLogger(__name__)


# ---------------- 路由函数 ----------------

def _route_intent(state: dict) -> str:
    intent = state.get("intent", {})
    if intent.get("need_plan") and intent.get("tools"):
        return "planner"
    if state.get("plan"):
        return "executor"
    return "answer"


def _route_executor(state: dict) -> str:
    plan = state.get("plan", [])
    if not plan or state.get("step_index", 0) >= len(plan):
        return "reflection"
    return "executor"


def _route_reflection(state: dict) -> str:
    r = state.get("reflection", {})
    if r.get("pass", True):
        return "answer"
    # 有待执行的补充步骤 → 继续执行；否则放弃重试直接回答
    if state.get("step_index", 0) < len(state.get("plan", [])):
        return "executor"
    return "answer"


# ---------------- MiniGraph 降级实现 ----------------

class MiniGraph:
    """langgraph.StateGraph 的最小等价物（add_node/add_edge/conditional_edges/compile）。

    仅用于 langgraph 未安装时的兜底，语义：从 START 出发沿边执行节点，
    条件边由路由函数返回下一节点名，直到 END。
    """

    START = "__start__"
    END = "__end__"

    def __init__(self, schema=None):
        self._nodes: dict[str, Callable] = {}
        self._edges: dict[str, str] = {}
        self._cond_edges: dict[str, tuple[Callable, dict | None]] = {}
        self._entry = self.START

    def add_node(self, name: str, fn: Callable) -> "MiniGraph":
        self._nodes[name] = fn
        return self

    def add_edge(self, a: str, b: str) -> "MiniGraph":
        if a == self.START:
            self._entry = b
        else:
            self._edges[a] = b
        return self

    def add_conditional_edges(self, a: str, router: Callable, mapping: dict | None = None) -> "MiniGraph":
        self._cond_edges[a] = (router, mapping)
        return self

    def compile(self):
        outer = self

        class _Compiled:
            backend = "minigraph"

            def invoke(self, state: dict, config: dict | None = None) -> dict:
                state = dict(state)
                current = outer._entry
                for _ in range(64):  # 防御性步数上限
                    if current == outer.END:
                        break
                    fn = outer._nodes.get(current)
                    if fn is None:
                        break
                    updates = fn(state) or {}
                    state.update(updates)
                    if current in outer._cond_edges:
                        router, _ = outer._cond_edges[current]
                        nxt = router(state)
                        current = nxt
                    else:
                        current = outer._edges.get(current, outer.END)
                return state

        return _Compiled()


# ---------------- 构建 Agent Graph ----------------

def build_agent_graph(nodes: AgentNodes):
    """组装并编译 Agent 图，返回可 invoke 的 compiled graph。"""
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError:
        logger.warning("未安装 langgraph，使用内置 MiniGraph 降级")
        START, END = MiniGraph.START, MiniGraph.END
        StateGraph = MiniGraph  # type: ignore[assignment]

    g = StateGraph(AgentState)
    g.add_node("intent", nodes.intent_node)
    g.add_node("planner", nodes.planner_node)
    g.add_node("executor", nodes.executor_node)
    g.add_node("reflection", nodes.reflection_node)
    g.add_node("answer", nodes.answer_node)

    g.add_edge(START, "intent")
    g.add_conditional_edges("intent", _route_intent)
    g.add_edge("planner", "executor")
    g.add_conditional_edges("executor", _route_executor)
    g.add_conditional_edges("reflection", _route_reflection)
    g.add_edge("answer", END)

    compiled = g.compile()
    backend = getattr(compiled, "backend", "langgraph")
    logger.info("Agent 图已编译（%s）：intent → planner → executor ⇄ reflection → answer", backend)
    return compiled
