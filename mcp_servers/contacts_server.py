"""演示用 MCP Server：企业通讯录（stdio JSON-RPC，MCP 2024-11-05 子集）。

启动方式（由 agent 配置自动拉起）:
    python mcp_servers/contacts_server.py
协议: 每行一个 JSON-RPC 消息（NDJSON），支持 initialize / tools/list / tools/call。
"""
from __future__ import annotations

import json
import sys

CONTACTS = [
    {"name": "张伟", "department": "市场部", "title": "市场总监", "phone": "13801010101", "email": "zhangwei@corp.com", "location": "总部 8F-801"},
    {"name": "李娜", "department": "财务部", "title": "财务经理", "phone": "13801010102", "email": "lina@corp.com", "location": "总部 6F-603"},
    {"name": "王强", "department": "技术部", "title": "架构师", "phone": "13801010103", "email": "wangqiang@corp.com", "location": "研发中心 3F-312"},
    {"name": "赵敏", "department": "人事部", "title": "HRBP", "phone": "13801010104", "email": "zhaomin@corp.com", "location": "总部 2F-215"},
    {"name": "陈涛", "department": "市场部", "title": "渠道经理", "phone": "13801010105", "email": "chentao@corp.com", "location": "总部 8F-808"},
    {"name": "孙丽", "department": "技术部", "title": "前端工程师", "phone": "13801010106", "email": "sunli@corp.com", "location": "研发中心 3F-306"},
    {"name": "周杰", "department": "技术部", "title": "算法工程师", "phone": "13801010107", "email": "zhoujie@corp.com", "location": "研发中心 3F-310"},
    {"name": "吴静", "department": "财务部", "title": "会计", "phone": "13801010108", "email": "wujing@corp.com", "location": "总部 6F-606"},
]

TOOLS = [
    {
        "name": "search_contacts",
        "description": "按姓名或部门关键词搜索企业通讯录，返回员工姓名、部门、职位、电话、邮箱、办公地点。",
        "inputSchema": {"type": "object", "properties": {"keyword": {"type": "string", "description": "姓名或部门关键词"}}, "required": ["keyword"]},
    },
    {
        "name": "list_departments",
        "description": "列出企业所有部门及各部门人数。",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def _match_contacts(kw: str) -> list[dict]:
    """通讯录匹配：支持中文连写（如「技术部周杰」）、姓名、部门、职位关键词。"""
    if not kw:
        return CONTACTS
    # 1) 优先精确：关键词包含某员工姓名 → 只返回姓名匹配的记录
    by_name = [c for c in CONTACTS if c["name"].lower() in kw or kw in c["name"].lower()]
    if by_name:
        return by_name
    # 2) 部门匹配（含连写：关键词包含部门名）
    by_dept = [c for c in CONTACTS if c["department"] in kw or kw in c["department"]]
    if by_dept:
        return by_dept
    # 3) 通用字段包含匹配
    return [
        c for c in CONTACTS
        if kw in " ".join(str(v) for v in c.values()).lower()
        or kw in c["title"]
    ]


def handle_tool_call(name: str, arguments: dict) -> dict:
    if name == "search_contacts":
        kw = (arguments.get("keyword") or "").strip().lower()
        matched = _match_contacts(kw)
        lines = [f"- {c['name']} | {c['department']} {c['title']} | 电话 {c['phone']} | {c['email']} | {c['location']}" for c in matched]
        text = "\n".join(lines) if lines else f"通讯录中未找到与「{arguments.get('keyword')}」相关的员工"
        return {"content": [{"type": "text", "text": text}]}
    if name == "list_departments":
        dept: dict[str, int] = {}
        for c in CONTACTS:
            dept[c["department"]] = dept.get(c["department"], 0) + 1
        text = "\n".join(f"- {d}: {n} 人" for d, n in dept.items())
        return {"content": [{"type": "text", "text": text}]}
    raise ValueError(f"未知工具: {name}")


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        method = msg.get("method")
        msg_id = msg.get("id")

        if method == "initialize":
            resp = {
                "jsonrpc": "2.0", "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "contacts-server", "version": "1.0.0"},
                },
            }
        elif method == "notifications/initialized":
            continue
        elif method == "tools/list":
            resp = {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}}
        elif method == "tools/call":
            try:
                result = handle_tool_call(msg["params"]["name"], msg["params"].get("arguments") or {})
                resp = {"jsonrpc": "2.0", "id": msg_id, "result": result}
            except Exception as e:
                resp = {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32000, "message": str(e)}}
        else:
            resp = {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": f"method not found: {method}"}}

        sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
