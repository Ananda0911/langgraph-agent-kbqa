"""工具抽象：所有 Agent 工具实现 Tool 接口，经 ToolRegistry 注册。"""
from __future__ import annotations

import abc
from typing import Any


class Tool(abc.ABC):
    """工具基类。子类需提供 name / description，并实现 run。"""

    name: str = "tool"
    description: str = ""

    @abc.abstractmethod
    def run(self, args: dict[str, Any]) -> dict[str, Any]:
        """执行工具。

        返回:
            {"output": <给 LLM 看的文本>, "sources": [...], "error": <可选错误信息>}
        """

    def args_hint(self) -> str:
        """参数格式提示（注入 Planner Prompt）。"""
        return '{"query": "..."}'

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "description": self.description, "args_hint": self.args_hint()}


class ToolRegistry:
    """工具注册表。"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def prompt(self) -> str:
        """生成注入 LLM 的工具清单文本。"""
        lines = []
        for t in self._tools.values():
            lines.append(f"- {t.name}: {t.description} 参数: {t.args_hint()}")
        return "\n".join(lines)
