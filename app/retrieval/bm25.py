"""BM25 稀疏检索（jieba 分词，内存倒排实现）。

对应图片中混合检索的 BM25 通道：对专有名词（如"雇主责任险"）等
字面精确匹配敏感的 query 召回效果显著优于纯向量检索。
"""
from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Iterable

import jieba

jieba.setLogLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# 过滤单字符噪声与停用词
_STOPWORDS = set(
    "的了和是在就都而及与著或一个我们你们他们它这那有没 not the a an of to in".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in jieba.lcut(text.lower()) if t.strip() and t not in _STOPWORDS]


class BM25Index:
    """Okapi BM25: score(q, d) = Σ idf(qi) * (f(qi,d)*(k1+1)) / (f(qi,d) + k1*(1-b+b*|d|/avgdl))"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_freq: Counter[str] = Counter()     # 词 -> 出现过的文档数
        self.doc_tokens: dict[str, list[str]] = {}  # chunk_id -> tokens
        self.doc_len: dict[str, int] = {}
        self.total_len = 0

    def add(self, chunk_id: str, text: str) -> None:
        tokens = tokenize(text)
        self.doc_tokens[chunk_id] = tokens
        self.doc_len[chunk_id] = len(tokens)
        self.total_len += len(tokens)
        for term in set(tokens):
            self.doc_freq[term] += 1

    def remove(self, chunk_id: str) -> None:
        tokens = self.doc_tokens.pop(chunk_id, None)
        if tokens is None:
            return
        self.total_len -= self.doc_len.pop(chunk_id, 0)
        for term in set(tokens):
            if term in self.doc_freq:
                self.doc_freq[term] -= 1
                if self.doc_freq[term] <= 0:
                    del self.doc_freq[term]

    @property
    def n_docs(self) -> int:
        return len(self.doc_tokens)

    def __len__(self) -> int:
        return len(self.doc_tokens)

    def _avgdl(self) -> float:
        return (self.total_len / self.n_docs) if self.n_docs else 0.0

    def _idf(self, term: str) -> float:
        df = self.doc_freq.get(term, 0)
        n = self.n_docs or 1
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if not self.n_docs:
            return []
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        avgdl = self._avgdl()
        scores: dict[str, float] = {}
        for qt in set(q_tokens):
            idf = self._idf(qt)
            if idf <= 0:
                continue
            for cid, tokens in self.doc_tokens.items():
                tf = tokens.count(qt)
                if tf == 0:
                    continue
                dl = self.doc_len.get(cid, 0)
                denom = tf + self.k1 * (1 - self.b + self.b * (dl / avgdl if avgdl else 0))
                scores[cid] = scores.get(cid, 0.0) + idf * tf * (self.k1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        return ranked
