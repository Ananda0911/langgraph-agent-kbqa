"""Rerank 层：对混合召回的候选用交叉注意力式的精排模型重排。

- LLMReranker: 用 LLM 对 query-chunk 对做相关性打分（对应 BGE-Reranker 的轻量替代，
  需 LLM api_key；不可用时自动回退融合分数）
- FusionReranker: 直接使用融合分数（无向量库轻量模式）
"""
from __future__ import annotations

import logging
import re
from collections.abc import Callable

logger = logging.getLogger(__name__)

_RERANK_PROMPT = """你是一个搜索相关性评估专家。给定用户问题和候选文档片段，判断片段对回答该问题的相关性。
只输出一个 0 到 10 的整数分数，10 表示完全相关，0 表示无关。不要输出任何其他内容。

问题：{query}

候选片段：
{doc}

相关性分数："""


class BaseReranker:
    name = "base"

    def rerank(
        self, query: str, candidates: list[tuple[str, float]], get_text: Callable[[str], str]
    ) -> list[tuple[str, float]]:
        raise NotImplementedError


class FusionReranker(BaseReranker):
    """直接沿用混合融合分数排序（轻量模式 / LLM 不可用时的回退）。"""

    name = "fusion"

    def rerank(self, query, candidates, get_text):
        return sorted(candidates, key=lambda x: -x[1])


class LLMReranker(BaseReranker):
    """LLM 打分精排。candidates: [(chunk_id, fused_score)]，返回同结构。"""

    name = "llm"

    def __init__(self, llm_client):
        self._llm = llm_client

    def rerank(self, query, candidates, get_text):
        if not candidates or not self._llm.available:
            return sorted(candidates, key=lambda x: -x[1])
        reranked: list[tuple[str, float]] = []
        for cid, fused in candidates:
            text = get_text(cid)
            prompt = _RERANK_PROMPT.format(query=query, doc=text[:500])
            try:
                resp = self._llm.chat([{"role": "user", "content": prompt}])
                m = re.search(r"\d+", resp)
                score = float(m.group()) / 10.0 if m else 0.0
            except Exception as e:
                logger.warning("rerank 打分失败，回退融合分: %s", e)
                score = fused
            reranked.append((cid, score))
        return sorted(reranked, key=lambda x: -x[1])


class BGECrossEncoderReranker(BaseReranker):
    """BGE-Reranker 本地 cross-encoder 适配器（可选）。

    安装 sentence-transformers 并下载模型后启用：
        pip install sentence-transformers
    配置 retrieval.rerank.provider: bge
    """

    name = "bge"

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        from sentence_transformers import CrossEncoder  # type: ignore

        self._model = CrossEncoder(model_name, max_length=512)

    def rerank(self, query, candidates, get_text):
        if not candidates:
            return []
        pairs = [(query, get_text(cid)[:1000]) for cid, _ in candidates]
        scores = self._model.predict(pairs)
        out = [(cid, float(s)) for (cid, _), s in zip(candidates, scores)]
        return sorted(out, key=lambda x: -x[1])
