"""Agent 状态定义：LangGraph StateGraph 各节点共享的状态。"""
from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """一次问答的完整状态（LangGraph 节点读写此结构）。

    说明：未使用 Annotated reducer，各节点返回的键整体覆盖；
    列表字段由节点自行读取-追加后返回。
    """

    query: str                    # 用户问题
    session_id: str                # 会话 ID
    memory_ctx: str                # 分级记忆 + 行动记忆注入的上下文
    model: str                    # 本次使用的模型 profile

    # 意图识别结果
    intent: dict[str, Any]        # {type, tools, reason}
    need_plan: bool               # 复杂任务 → planner 节点生成多步计划

    # 计划与执行
    plan: list[dict[str, Any]]    # [{description, tool, args, status}]
    step_index: int               # 当前执行到第几步
    observations: list[dict[str, Any]]  # [{step, tool, args, output, sources, ok}]

    # 反思评估
    iterations: int              # 反思轮数（防死循环）
    reflection: dict[str, Any]     # {pass, feedback, extra_steps}

    # 最终输出
    final_answer: str
    sources: list[dict[str, Any]]
    trace: list[dict[str, Any]]   # 全过程轨迹（前端展示）


def initial_state(query: str, session_id: str, memory_ctx: str = "", model: str = "") -> AgentState:
    return {
        "query": query,
        "session_id": session_id,
        "memory_ctx": memory_ctx,
        "model": model,
        "intent": {},
        "need_plan": False,
        "plan": [],
        "step_index": 0,
        "observations": [],
        "iterations": 0,
        "reflection": {},
        "final_answer": "",
        "sources": [],
        "trace": [],
    }
