"""混合检索编排器：BM25 + Vector 双路召回 → 融合 → Rerank → Top-K。

对应图片核心特性 1「混合检索策略」：
BM25 + 向量加权融合，并引入 Rerank 精排，提升 Top-5 召回质量。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.retrieval.bm25 import BM25Index
from app.retrieval.hybrid import fuse
from app.retrieval.reranker import FusionReranker, LLMReranker
from app.retrieval.vector import VectorIndex


@dataclass
class SearchHit:
    chunk_id: str
    score: float
    text: str
    doc_name: str
    doc_id: str
    seq: int
    bm25_score: float = 0.0
    vector_score: float = 0.0
    fused_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "seq": self.seq,
            "score": round(self.score, 4),
            "bm25": round(self.bm25_score, 4),
            "vector": round(self.vector_score, 4),
            "fused": round(self.fused_score, 4),
            "text": self.text,
        }


@dataclass
class SearchResult:
    hits: list[SearchHit] = field(default_factory=list)
    latency_ms: float = 0.0
    debug: dict = field(default_factory=dict)


class HybridRetriever:
    def __init__(
        self,
        bm25_index: BM25Index,
        vector_index: VectorIndex,
        store,  # ChunkStore
        retrieval_cfg: dict,
        llm_client=None,
    ):
        self.bm25 = bm25_index
        self.vector = vector_index
        self.store = store
        self.cfg = retrieval_cfg
        self._llm = llm_client
        self._reranker = self._build_reranker(retrieval_cfg.get("rerank", {}))

    def _build_reranker(self, rcfg: dict):
        if not rcfg.get("enabled", True):
            return FusionReranker()
        provider = rcfg.get("provider", "fusion")
        if provider == "llm" and self._llm is not None and self._llm.available:
            return LLMReranker(self._llm)
        if provider == "bge":
            try:
                from app.retrieval.reranker import BGECrossEncoderReranker

                return BGECrossEncoderReranker()
            except Exception:
                pass
        return FusionReranker()

    def search(self, query: str, top_k: int | None = None) -> SearchResult:
        t0 = time.perf_counter()
        top_k = top_k or int(self.cfg.get("top_k", 5))
        cand_k = int(self.cfg.get("candidate_k", 10))
        fcfg = self.cfg.get("fusion", {})

        bm25_hits = self.bm25.search(query, top_k=cand_k)
        vector_hits = self.vector.search(query, top_k=cand_k)
        fused = fuse(
            bm25_hits,
            vector_hits,
            method=fcfg.get("method", "weighted"),
            bm25_weight=float(fcfg.get("bm25_weight", 0.5)),
            vector_weight=float(fcfg.get("vector_weight", 0.5)),
            rrf_k=int(fcfg.get("rrf_k", 60)),
        )
        bm25_map = dict(bm25_hits)
        vec_map = dict(vector_hits)
        candidates = fused[: max(cand_k, top_k)]

        reranked = self._reranker.rerank(query, candidates, lambda cid: self.store.text(cid))

        hits: list[SearchHit] = []
        seen_docs: dict[str, int] = {}
        for cid, score in reranked:
            chunk = self.store.get(cid)
            if chunk is None:
                continue
            # 同文档最多保留 2 块，保证来源多样性
            seen_docs[chunk.doc_id] = seen_docs.get(chunk.doc_id, 0) + 1
            if seen_docs[chunk.doc_id] > 2:
                continue
            hits.append(
                SearchHit(
                    chunk_id=cid,
                    score=score,
                    text=chunk.text,
                    doc_name=chunk.doc_name,
                    doc_id=chunk.doc_id,
                    seq=chunk.seq,
                    bm25_score=bm25_map.get(cid, 0.0),
                    vector_score=vec_map.get(cid, 0.0),
                    fused_score=dict(fused).get(cid, 0.0),
                )
            )
            if len(hits) >= top_k:
                break

        latency = (time.perf_counter() - t0) * 1000
        return SearchResult(
            hits=hits,
            latency_ms=round(latency, 2),
            debug={
                "bm25_candidates": [dict(chunk_id=c, score=round(s, 4)) for c, s in bm25_hits[:5]],
                "vector_candidates": [dict(chunk_id=c, score=round(s, 4)) for c, s in vector_hits[:5]],
                "reranker": self._reranker.name,
                "corpus_chunks": len(self.bm25),
            },
        )
