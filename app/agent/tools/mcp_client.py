"""轻量 MCP 客户端：JSON-RPC over stdio（MCP 2024-11-05 协议子集）。

支持 initialize 握手 → notifications/initialized → tools/list → tools/call。
每次调用独立拉起子进程（演示场景足够，生产可换常驻连接池）。
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from app.config import BASE_DIR, get_config

logger = logging.getLogger(__name__)


class MCPStdioClient:
    """与单个 MCP Server 的 stdio JSON-RPC 通信。"""

    def __init__(self, name: str, command: str, args: list[str], cwd: str | None = None, timeout: float = 30):
        self.name = name
        self.command = command
        self.args = [a if Path(str(a)).is_absolute() else str(BASE_DIR / a) for a in args]
        self.cwd = str(BASE_DIR / cwd) if cwd else str(BASE_DIR)
        self.timeout = timeout
        self._proc: subprocess.Popen | None = None
        self._id = 0
        self._lock = threading.Lock()

    # ---------- 生命周期 ----------

    def start(self) -> None:
        if self._proc and self._proc.poll() is None:
            return
        self._proc = subprocess.Popen(
            [self.command, *self.args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.cwd,
            text=True,
            encoding="utf-8",
        )

    def close(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.stdin.close()
                self._proc.terminate()
            except Exception:
                pass
        self._proc = None

    # ---------- JSON-RPC ----------

    def _send(self, obj: dict) -> None:
        assert self._proc and self._proc.stdin
        self._proc.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()

    def _read_response(self, req_id: int) -> dict:
        assert self._proc and self._proc.stdout
        import time

        deadline = time.time() + self.timeout
        while time.time() < deadline:
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError(f"MCP server {self.name} 输出流关闭")
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("id") == req_id:
                if "error" in msg:
                    raise RuntimeError(f"MCP server {self.name} 返回错误: {msg['error']}")
                return msg.get("result", {})
        raise TimeoutError(f"MCP server {self.name} 响应超时")

    def _call(self, method: str, params: dict | None = None) -> dict:
        with self._lock:
            self.start()
            assert self._proc
            self._id += 1
            req_id = self._id
            self._send({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}})
            return self._read_response(req_id)

    def _notify(self, method: str, params: dict | None = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    # ---------- MCP 协议 ----------

    def initialize(self) -> dict:
        result = self._call(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "kbqa-agent", "version": "1.0.0"},
            },
        )
        self._notify("notifications/initialized")
        return result

    def list_tools(self) -> list[dict]:
        result = self._call("tools/list")
        return result.get("tools", [])

    def call_tool(self, tool_name: str, arguments: dict) -> Any:
        result = self._call("tools/call", {"name": tool_name, "arguments": arguments or {}})
        # MCP 返回 content: [{type: "text", text: "..."}]
        texts = [c.get("text", "") for c in result.get("content", []) if c.get("type") == "text"]
        return "\n".join(t for t in texts if t) or json.dumps(result, ensure_ascii=False)


class MCPTool:
    """将一个 MCP Server 的工具集合暴露为单个 Agent 工具。"""

    def __init__(self, server_name: str, command: str, args: list[str], description: str = "", cwd: str | None = None):
        self.name = f"mcp_{server_name}"
        self._client = MCPStdioClient(server_name, command, args, cwd=cwd)
        self._server_desc = description
        self._tools: list[dict] = []
        self._desc = self._server_desc
        self._refresh_tools()

    def _refresh_tools(self) -> None:
        """启动时连接 server 获取工具清单（失败则降级为占位描述）。"""
        try:
            self._client.initialize()
            self._tools = self._client.list_tools()
            self._client.close()
            tool_desc = "; ".join(
                f"{t['name']}({', '.join(t.get('inputSchema', {}).get('properties', {}).keys())}): {t.get('description', '')}"
                for t in self._tools
            )
            self._desc = f"MCP 服务「{self._server_desc}」工具: {tool_desc}"
        except Exception as e:
            logger.warning("MCP server %s 工具发现失败: %s", self._client.name, e)
            self._desc = f"MCP 服务「{self._server_desc}」（当前不可用: {e}）"

    @property
    def description(self) -> str:
        return self._desc

    @property
    def tools(self) -> list[dict]:
        return self._tools

    def args_hint(self) -> str:
        if not self._tools:
            return '{"tool": "工具名", "arguments": {...}}'
        t0 = self._tools[0]
        props = t0.get("inputSchema", {}).get("properties", {})
        sample = ", ".join(f'"{k}": "..."' for k in list(props)[:2])
        if len(self._tools) == 1:
            return "{" + sample + "}" if sample else "{}"
        return f'{{"tool": "{t0["name"]}", "arguments": {{{sample}}}}}'

    def run(self, args: dict[str, Any]) -> dict[str, Any]:
        tool_name = args.get("tool") or (self._tools[0]["name"] if self._tools else "")
        arguments = args.get("arguments") or {
            k: v for k, v in args.items() if k not in ("tool", "arguments")
        }
        if not tool_name:
            return {"output": f"MCP 服务无可用工具", "sources": [], "error": "no tools"}
        try:
            self._client.initialize()
            output = self._client.call_tool(tool_name, arguments)
        except Exception as e:
            return {"output": f"MCP 调用失败：{e}", "sources": [], "error": str(e)}
        finally:
            self._client.close()
        return {"output": f"[MCP 工具 {tool_name} 返回]\n{output}", "sources": []}

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "description": self.description, "args_hint": self.args_hint()}


def build_mcp_tools_from_config() -> list[MCPTool]:
    """从 config.yaml 的 agent.mcp_servers 段构建 MCP 工具列表。"""
    cfg = get_config().sub("agent.mcp_servers") or {}
    tools = []
    for name, scfg in cfg.items():
        try:
            tools.append(
                MCPTool(
                    server_name=name,
                    command=scfg.get("command", sys.executable),
                    args=scfg.get("args", []),
                    description=scfg.get("description", name),
                )
            )
        except Exception as e:
            logger.warning("MCP 服务 %s 初始化失败: %s", name, e)
    return tools
