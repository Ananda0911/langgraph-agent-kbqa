"""混合检索融合层：BM25 + 向量双路结果融合。

- weighted: 每路分数 min-max 归一化后按权重线性融合（图片中的「加权融合」）
- rrf:      Reciprocal Rank Fusion，按排名倒数融合，对分数分布不敏感
"""
from __future__ import annotations

from collections.abc import Iterable


def _minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def fuse(
    bm25_hits: list[tuple[str, float]],
    vector_hits: list[tuple[str, float]],
    method: str = "weighted",
    bm25_weight: float = 0.5,
    vector_weight: float = 0.5,
    rrf_k: int = 60,
) -> list[tuple[str, float]]:
    """融合两路检索结果，返回 (chunk_id, fused_score) 降序列表。"""
    if not bm25_hits and not vector_hits:
        return []
    if not bm25_hits:
        return list(vector_hits)
    if not vector_hits:
        return list(bm25_hits)

    if method == "rrf":
        fused: dict[str, float] = {}
        for hits, weight in ((bm25_hits, bm25_weight), (vector_hits, vector_weight)):
            for rank, (cid, _) in enumerate(hits, start=1):
                fused[cid] = fused.get(cid, 0.0) + weight / (rrf_k + rank)
        return sorted(fused.items(), key=lambda x: -x[1])

    # 默认: weighted（min-max 归一化 + 线性加权）
    b_norm = _minmax(dict(bm25_hits))
    v_norm = _minmax(dict(vector_hits))
    all_ids = set(b_norm) | set(v_norm)
    scores = {
        cid: bm25_weight * b_norm.get(cid, 0.0) + vector_weight * v_norm.get(cid, 0.0)
        for cid in all_ids
    }
    return sorted(scores.items(), key=lambda x: -x[1])
