"""向量检索通道。

默认 provider = local_hash：本地哈希特征向量（词级 TF-IDF 哈希到固定维度），
零外部依赖、零模型下载 —— 对应图片中「无向量库轻量化方案」。
接口抽象为 EmbeddingProvider，可切换 OpenAI 兼容 embeddings API（如 BGE 系列托管服务）。
"""
from __future__ import annotations

import hashlib
import math
import threading
from collections import Counter
from collections.abc import Iterable

import numpy as np

from app.retrieval.bm25 import tokenize


class EmbeddingProvider:
    """Embedding 提供方抽象。"""

    name = "base"

    def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]

    def dim(self) -> int:
        raise NotImplementedError


class LocalHashEmbedding(EmbeddingProvider):
    """哈希特征 + TF-IDF 加权 + L2 归一化。

    词 -> md5 哈希 -> 维度索引，语料统计 IDF 后与查询做余弦相似度。
    轻量、可解释，适合中小规模文档集（1000+ 文档）秒级响应。
    """

    name = "local_hash"

    def __init__(self, dim: int = 2048):
        self._dim = dim
        self._lock = threading.Lock()
        self._doc_freq: Counter[int] = Counter()
        self._n_docs = 0

    def dim(self) -> int:
        return self._dim

    def _term_slots(self, token: str) -> tuple[int, int]:
        """每个词映射两个槽位（降低哈希冲突，类似布隆滤波思路）。"""
        h = hashlib.md5(token.encode("utf-8")).digest()
        s1 = int.from_bytes(h[:4], "little") % self._dim
        s2 = int.from_bytes(h[4:8], "little") % self._dim
        return s1, s2

    def _vector(self, tokens: list[str], update_df: bool = False) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        tf = Counter(t for t in tokens if t.strip())
        with self._lock:
            if update_df:
                self._doc_freq.update(set(tf.keys()))
                self._n_docs += 1
            idf = {
                t: math.log(1 + (self._n_docs + 1) / (self._doc_freq.get(t, 0) + 1))
                for t in tf
            }
        for t, cnt in tf.items():
            w = (1.0 + math.log(cnt)) * idf.get(t, 1.0)
            s1, s2 = self._term_slots(t)
            vec[s1] += w
            vec[s2] += w / 2.0  # 第二槽位降权，抑制冲突噪声
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        return [self._vector(tokenize(t), update_df=True) for t in texts]

    def embed_query(self, text: str) -> np.ndarray:
        # 查询不更新语料统计
        return self._vector(tokenize(text), update_df=False)


class OpenAICompatibleEmbedding(EmbeddingProvider):
    """OpenAI 兼容 /embeddings API（DashScope text-embedding-v3 等）。"""

    name = "openai_compatible"

    def __init__(self, base_url: str, api_key: str, model: str, dim: int = 1024):
        import httpx

        self._client = httpx.Client(timeout=30)
        self._url = base_url.rstrip("/") + "/embeddings"
        self._api_key = api_key
        self._model = model
        self._dim = dim

    def dim(self) -> int:
        return self._dim

    def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        resp = self._client.post(
            self._url,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"model": self._model, "input": texts, "dimensions": self._dim},
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        order = sorted(range(len(texts)), key=lambda i: data[i]["index"])
        return [np.asarray(data[i]["embedding"], dtype=np.float32) for i in order]

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]


class VectorIndex:
    """余弦相似度向量检索（向量 L2 归一化后点积即余弦）。"""

    def __init__(self, provider: EmbeddingProvider):
        self.provider = provider
        self._ids: list[str] = []
        self._matrix: np.ndarray | None = None
        self._id2row: dict[str, int] = {}

    def add(self, chunk_id: str, text: str) -> None:
        # 入库走 embed_texts：local_hash 提供方在此更新语料 IDF 统计
        vec = self.provider.embed_texts([text])[0]
        row = self._id2row.get(chunk_id)
        if row is not None:
            self._matrix[row] = vec
            return
        if self._matrix is None:
            self._matrix = np.zeros((0, len(vec)), dtype=np.float32)
        self._id2row[chunk_id] = len(self._ids)
        self._ids.append(chunk_id)
        self._matrix = np.vstack([self._matrix, vec]) if len(self._ids) > 1 else np.asarray([vec])

    def remove(self, chunk_id: str) -> None:
        row = self._id2row.pop(chunk_id, None)
        if row is None or self._matrix is None:
            return
        keep = [i for i in range(len(self._ids)) if i != row]
        self._matrix = self._matrix[keep] if keep else None
        self._ids = [self._ids[i] for i in keep]
        self._id2row = {cid: i for i, cid in enumerate(self._ids)}

    def __len__(self) -> int:
        return len(self._ids)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if self._matrix is None or not len(self._ids):
            return []
        q = self.provider.embed_query(query)
        scores = (self._matrix @ q).astype(np.float64)
        top_k = min(top_k, len(self._ids))
        idx = np.argsort(-scores)[:top_k]
        return [(self._ids[i], float(scores[i])) for i in idx]


def create_embedding_provider(cfg: dict) -> EmbeddingProvider:
    provider = cfg.get("provider", "local_hash")
    if provider == "openai_compatible" and cfg.get("api_key"):
        return OpenAICompatibleEmbedding(
            base_url=cfg.get("base_url", ""),
            api_key=cfg["api_key"],
            model=cfg.get("model", "text-embedding-v3"),
            dim=int(cfg.get("dim", 1024)),
        )
    return LocalHashEmbedding(dim=int(cfg.get("dim", 2048)))
