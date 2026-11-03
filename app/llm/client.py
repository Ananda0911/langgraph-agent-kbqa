"""LLM 客户端：OpenAI 兼容 Chat API（Qwen-3 等）+ Mock 降级。

- 有 api_key: 通过 httpx 调用 {base_url}/chat/completions
- 无 api_key:  MockLLM 生成抽取式回答（拼接 top 片段关键句），保证全链路可演示
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class MockLLM:
    """无 Key 时的降级实现：从上下文中抽取关键句生成回答。"""

    available = False

    def chat(self, messages: list[dict], **kwargs) -> str:
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        # 从 prompt 中抽取【知识片段】内容拼装回答
        fragments = re.findall(r"【知识片段\d+】(.*?)(?=【知识片段\d+】|【|$)", last_user, re.S)
        if fragments:
            answer = "根据知识库检索到的内容，要点如下：\n"
            for i, frag in enumerate(fragments[:4], 1):
                frag = " ".join(frag.split())
                sents = re.split(r"(?<=[。！？])", frag)
                key = "".join(sents[:2])[:120]
                if key.strip():
                    answer += f"{i}. {key.strip()}\n"
            answer += "\n（当前为 Mock 模式回答：未配置 LLM api_key，以上为检索片段抽取结果。配置 config.yaml 中 llm.api_key 后可获得完整生成能力。）"
            return answer
        if "摘要" in last_user or "总结" in last_user:
            return last_user[:200] + " …（Mock 摘要）"
        return "（Mock 模式：请配置 config.yaml 中的 llm.api_key 以启用 Qwen-3 生成能力。）"


class OpenAICompatibleLLM:
    available = True

    def __init__(self, cfg: dict):
        import httpx

        self.base_url = cfg.get("base_url", "").rstrip("/")
        self.api_key = cfg.get("api_key", "")
        self.model = cfg.get("model", "qwen-plus")
        self.temperature = float(cfg.get("temperature", 0.3))
        self.max_tokens = int(cfg.get("max_tokens", 1024))
        self.timeout = float(cfg.get("timeout", 60))
        self._client = httpx.Client(timeout=self.timeout)

    def chat(self, messages: list[dict], **kwargs) -> str:
        body = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        try:
            resp = self._client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=body,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise LLMError(f"LLM 调用失败: {e}") from e


class LLMClient:
    """统一入口：自动选择真实 API 或 Mock 降级。"""

    def __init__(self, cfg: dict):
        self._impl: MockLLM | OpenAICompatibleLLM
        if cfg.get("api_key"):
            self._impl = OpenAICompatibleLLM(cfg)
            self.mode = "openai_compatible"
            logger.info("LLM 已启用真实 API: %s (%s)", cfg.get("model"), cfg.get("base_url"))
        else:
            self._impl = MockLLM()
            self.mode = "mock"
            logger.warning("未配置 llm.api_key，LLM 降级为 Mock 模式")

    @property
    def available(self) -> bool:
        return self._impl.available

    @property
    def model(self) -> str:
        return getattr(self._impl, "model", "")

    def set_model(self, model: str) -> None:
        """多模型支持：运行时切换当前使用的模型 id。"""
        if hasattr(self._impl, "model"):
            self._impl.model = model

    def chat(self, messages: list[dict], **kwargs) -> str:
        return self._impl.chat(messages, **kwargs)

    def summarize(self, text: str) -> str:
        """用于长期记忆压缩的摘要。"""
        if self.available:
            try:
                return self.chat(
                    [
                        {"role": "system", "content": "你是对话摘要助手，用一两句话提炼对话要点，保留关键实体和数字。"},
                        {"role": "user", "content": text},
                    ],
                    max_tokens=256,
                    temperature=0.1,
                )
            except LLMError as e:
                logger.warning("摘要失败，降级截断: %s", e)
                return text[:150] + "…"
        # Mock 摘要：取对话前 N 字
        compact = " ".join(text.split())
        return compact[:150]
