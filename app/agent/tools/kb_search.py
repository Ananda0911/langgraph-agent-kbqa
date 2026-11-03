"""本地 RAG 检索工具：包装 HybridRetriever（BM25+向量+融合+Rerank）。"""
from __future__ import annotations

from typing import Any


class KBSearchTool:
    """知识库检索（本地 RAG）。"""

    name = "kb_search"
    description = "在企业本地知识库中检索相关文档片段（混合检索 BM25+向量+精排）。适用于企业制度、产品资料、研报等文档类问题。"

    def __init__(self, retriever):
        self._retriever = retriever

    def args_hint(self) -> str:
        return '{"query": "检索问题", "top_k": 5(可选)}'

    def run(self, args: dict[str, Any]) -> dict[str, Any]:
        query = (args.get("query") or args.get("q") or "").strip()
        if not query:
            return {"output": "错误：缺少 query 参数", "sources": [], "error": "missing query"}
        top_k = int(args.get("top_k") or 5)
        result = self._retriever.search(query, top_k=top_k)
        if not result.hits:
            return {"output": "（知识库中没有检索到与该问题相关的内容）", "sources": []}
        lines, sources = [], []
        for i, h in enumerate(result.hits, 1):
            lines.append(f"【片段{i}】来源《{h.doc_name}》(第{h.seq + 1}段, 相关度{h.score:.3f})：{h.text}")
            sources.append(
                {
                    "idx": i,
                    "doc_name": h.doc_name,
                    "doc_id": h.doc_id,
                    "seq": h.seq,
                    "score": round(h.score, 4),
                    "text": h.text[:160] + ("…" if len(h.text) > 160 else ""),
                }
            )
        return {
            "output": "\n\n".join(lines),
            "sources": sources,
            "latency_ms": result.latency_ms,
        }

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "description": self.description, "args_hint": self.args_hint()}
