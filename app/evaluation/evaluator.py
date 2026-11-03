"""自动化评估流水线（对应图片核心特性 4）。

基于 Golden Dataset 对检索链路做离线评估，持续跟踪两个核心指标：
- Hit Rate@K : Top-K 结果中命中标准答案的比例
- MRR       : 第一个命中结果排名倒数的均值
"""
from __future__ import annotations

import json
import time
from pathlib import Path

DATASET_PATH = Path(__file__).parent / "golden_dataset.json"


def load_dataset() -> list[dict]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def _is_hit(hit, item: dict) -> bool:
    """命中判定：来源文档匹配 或 片段包含期望关键词。"""
    if hit.doc_name in item.get("expected_docs", []):
        return True
    return any(kw.lower() in hit.text.lower() for kw in item.get("expected_keywords", []))


def run_evaluation(state) -> dict:
    """state: AppState。返回评估报告。"""
    dataset = load_dataset()
    top_k = 5
    n = len(dataset)
    hits = 0
    rr_sum = 0.0
    details: list[dict] = []
    t0 = time.perf_counter()

    for item in dataset:
        result = state.retriever.search(item["question"], top_k=top_k)
        ranks = [i + 1 for i, h in enumerate(result.hits) if _is_hit(h, item)]
        hit = bool(ranks)
        rr = (1.0 / ranks[0]) if ranks else 0.0
        hits += hit
        rr_sum += rr
        details.append(
            {
                "question": item["question"],
                "hit": hit,
                "first_hit_rank": ranks[0] if ranks else None,
                "top_docs": [h.doc_name for h in result.hits[:3]],
                "latency_ms": result.latency_ms,
            }
        )

    report = {
        "total": n,
        "hit_rate@5": round(hits / n, 4) if n else 0.0,
        "mrr": round(rr_sum / n, 4) if n else 0.0,
        "avg_latency_ms": round(sum(d["latency_ms"] for d in details) / n, 2) if n else 0.0,
        "eval_time_ms": round((time.perf_counter() - t0) * 1000, 1),
        "llm_mode": state.llm.mode,
        "details": details,
    }
    # 持久化最近一次报告
    out = state.index.index_dir / "eval_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
