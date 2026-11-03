"""分级记忆系统（对应图片核心特性 3）。

短期记忆:  最近 N 轮对话原文（List / deque）
长期记忆:  超过阈值的历史轮次压缩为摘要（LLM 摘要），按会话持久保留
解决多轮对话上下文丢失与 token 成本之间的平衡。

MemoryStore 为存储抽象（当前 InMemory 实现），提供 RedisStore 适配器
接口以便生产环境无缝切换（对应技术栈中的 Redis）。
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from app.config import get_config
from app.llm.client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class Turn:
    role: str          # user / assistant
    content: str
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content, "ts": self.ts}


class MemoryStore:
    """记忆存储抽象。生产环境可替换为 RedisStore（SETEX / LIST / ZSET）。"""

    def get_turns(self, session_id: str) -> list[Turn]: ...
    def save_turns(self, session_id: str, turns: list[Turn]) -> None: ...
    def get_summaries(self, session_id: str) -> list[str]: ...
    def save_summaries(self, session_id: str, summaries: list[str]) -> None: ...


class InMemoryStore(MemoryStore):
    def __init__(self):
        self._lock = threading.RLock()
        self._turns: dict[str, deque[Turn]] = {}
        self._summaries: dict[str, list[str]] = {}

    def get_turns(self, session_id) -> list[Turn]:
        with self._lock:
            return list(self._turns.get(session_id, []))

    def save_turns(self, session_id, turns) -> None:
        with self._lock:
            self._turns[session_id] = deque(turns)

    def get_summaries(self, session_id) -> list[str]:
        with self._lock:
            return list(self._summaries.get(session_id, []))

    def save_summaries(self, session_id, summaries) -> None:
        with self._lock:
            self._summaries[session_id] = summaries


class RedisStore(MemoryStore):
    """生产环境 Redis 适配器（可选启用）。

    需要安装 redis-py 并配置 data.redis_url，例如:
        redis://localhost:6379/0
    短期记忆用 LIST，长期摘要用 LIST（LPUSH/LRANGE）。
    """

    def __init__(self, redis_url: str):
        import redis  # type: ignore

        self._r = redis.Redis.from_url(redis_url, decode_responses=True)

    @staticmethod
    def _k_turns(sid: str) -> str:
        return f"kbqa:mem:turns:{sid}"

    @staticmethod
    def _k_summ(sid: str) -> str:
        return f"kbqa:mem:summ:{sid}"

    def get_turns(self, session_id) -> list[Turn]:
        raw = self._r.lrange(self._k_turns(session_id), 0, -1)
        import json

        return [Turn(**json.loads(x)) for x in reversed(raw)]

    def save_turns(self, session_id, turns) -> None:
        import json

        key = self._k_turns(session_id)
        self._r.delete(key)
        if turns:
            self._r.rpush(key, *[json.dumps(t.to_dict()) for t in turns])
            self._r.expire(key, 86400)

    def get_summaries(self, session_id) -> list[str]:
        return self._r.lrange(self._k_summ(session_id), 0, -1)

    def save_summaries(self, session_id, summaries) -> None:
        key = self._k_summ(session_id)
        self._r.delete(key)
        if summaries:
            self._r.rpush(key, *summaries)
            self._r.expire(key, 604800)


def create_memory_store() -> MemoryStore:
    cfg = get_config()
    redis_url = cfg.get("data.redis_url")
    if redis_url:
        try:
            return RedisStore(redis_url)
        except Exception as e:
            logger.warning("Redis 连接失败，回退内存存储: %s", e)
    return InMemoryStore()


class MemoryManager:
    """会话级分级记忆编排。"""

    def __init__(self, llm: LLMClient, store: MemoryStore | None = None):
        mcfg = get_config().sub("memory")
        self.short_term_rounds = int(mcfg.get("short_term_rounds", 6))
        self.long_term_max = int(mcfg.get("long_term_max_summaries", 20))
        self.compress_threshold = int(mcfg.get("compress_threshold", 8))
        self.llm = llm
        self.store = store or create_memory_store()

    # ---------- 对话写入 ----------

    def add_turn(self, session_id: str, role: str, content: str) -> None:
        turns = self.store.get_turns(session_id)
        turns.append(Turn(role=role, content=content))
        turns = self._maybe_compress(session_id, turns)
        self.store.save_turns(session_id, turns)

    def _maybe_compress(self, session_id: str, turns: list[Turn]) -> list[Turn]:
        """超过阈值时，把最老的一批轮次压缩进长期摘要。"""
        if len(turns) <= self.compress_threshold:
            return turns
        overflow = len(turns) - self.short_term_rounds
        to_compress = turns[:overflow]
        kept = turns[overflow:]
        if to_compress:
            transcript = "\n".join(
                f"{'用户' if t.role == 'user' else '助手'}: {t.content[:300]}" for t in to_compress
            )
            summary = self.llm.summarize(transcript)
            summaries = self.store.get_summaries(session_id)
            summaries.append(summary)
            summaries = summaries[-self.long_term_max :]
            self.store.save_summaries(session_id, summaries)
            logger.info("会话 %s 压缩 %d 轮入长期记忆", session_id, len(to_compress))
        return kept

    # ---------- 上下文构建 ----------

    def build_context(self, session_id: str) -> str:
        """拼装注入 Prompt 的记忆上下文：长期摘要 + 近期轮次。"""
        summaries = self.store.get_summaries(session_id)
        turns = self.store.get_turns(session_id)
        parts: list[str] = []
        if summaries:
            parts.append("历史对话要点：\n- " + "\n- ".join(summaries))
        recent = turns[-self.short_term_rounds :]
        if recent:
            parts.append(
                "近期对话：\n"
                + "\n".join(
                    f"{'用户' if t.role == 'user' else '助手'}: {t.content[:200]}" for t in recent
                )
            )
        return "\n\n".join(parts)

    def snapshot(self, session_id: str) -> dict:
        """供调试面板查看当前会话记忆状态。"""
        turns = self.store.get_turns(session_id)
        return {
            "short_term": [t.to_dict() for t in turns[-self.short_term_rounds :]],
            "short_term_count": len(turns),
            "long_term_summaries": self.store.get_summaries(session_id),
            "short_term_rounds": self.short_term_rounds,
            "compress_threshold": self.compress_threshold,
        }

    def clear(self, session_id: str) -> None:
        self.store.save_turns(session_id, [])
        self.store.save_summaries(session_id, [])
