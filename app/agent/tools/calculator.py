"""安全计算器工具：基于 ast 白名单的算术表达式求值。"""
from __future__ import annotations

import ast
import operator
from typing import Any

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval(node: ast.AST):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError(f"不支持的表达式节点: {type(node).__name__}")


class CalculatorTool:
    """精确数值计算（安全沙箱，仅算术运算）。"""

    name = "calculator"
    description = "精确数学计算（加减乘除、幂、取模等），适合金额汇总、比例计算等。"

    def args_hint(self) -> str:
        return '{"expression": "例如 (450000+380000)*0.13"}'

    def run(self, args: dict[str, Any]) -> dict[str, Any]:
        expr = (args.get("expression") or args.get("expr") or "").strip()
        if not expr:
            return {"output": "错误：缺少 expression 参数", "sources": [], "error": "missing expression"}
        try:
            result = _eval(ast.parse(expr, mode="eval"))
        except Exception as e:
            return {"output": f"计算失败：{e}", "sources": [], "error": str(e)}
        if isinstance(result, float):
            result = round(result, 6)
        return {"output": f"{expr} = {result}", "sources": []}

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "description": self.description, "args_hint": self.args_hint()}
