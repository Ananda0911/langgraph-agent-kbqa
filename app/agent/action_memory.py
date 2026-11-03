"""行动记忆（Action Memory）：记录并复用 Agent 每次执行的行动轨迹。

对应图片「行动记忆」能力：意图 → 计划 → 工具调用 → 观察 → 反思，
按会话持久，既用于前端轨迹展示，也作为上下文注入后续规划。
"""
from __future__ import annotations

import threading
import time
from typing import Any


class ActionMemory:
    """会话级行动轨迹存储（线程安全内存实现）。"""

    def __init__(self, max_entries: int = 200):
        self._lock = threading.RLock()
        self._sessions: dict[str, list[dict[str, Any]]] = {}
        self.max_entries = max_entries

    def record(self, session_id: str, entry_type: str, content: Any, meta: dict | None = None) -> None:
        with self._lock:
            entries = self._sessions.setdefault(session_id, [])
            entries.append({"ts": round(time.time(), 3), "type": entry_type, "content": content, "meta": meta or {}})
            if len(entries) > self.max_entries:
                del entries[: len(entries) - self.max_entries]

    def trace(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def recent_context(self, session_id: str, last_n: int = 2) -> str:
        """将最近 N 次问答的行动轨迹压缩为上下文文本，注入意图识别/规划节点。"""
        entries = self.trace(session_id)
        if not entries:
            return ""
        # 取最近几轮（以 intent 条目为分界）
        intents = [e for e in entries if e["type"] == "intent"]
        recent = entries
        if len(intents) > last_n:
            cut = entries.index(intents[-last_n])
            recent = entries[cut:]
        lines = []
        for e in recent:
            t, c = e["type"], e["content"]
            if t == "intent":
                lines.append(f"上一轮意图: {c}")
            elif t == "plan":
                steps = "; ".join(f"{s.get('tool')}({s.get('description', '')})" for s in c)
                lines.append(f"上一轮计划: {steps}")
            elif t == "tool_call":
                lines.append(f"调用 {c}")
            elif t == "observation":
                out = str(c)[:120]
                lines.append(f"观察: {out}")
            elif t == "reflection":
                lines.append(f"反思: {c}")
        return "\n".join(lines)

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
