"""知识库索引器：ChunkStore（哈希索引）+ BM25 + Vector 三路索引的统一管理。

对应图片核心特性 2「无向量库轻量化方案」：
不依赖任何外部数据库服务，采用 文件系统 + 内存哈希索引（doc_id -> chunk ids,
content_hash -> chunk_id 去重），启动时从磁盘 JSON 重建，支持 1000+ 文档。
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict
from pathlib import Path

from app.config import get_config
from app.ingestion.doc_parser import Chunk, ParsedDocument, chunk_document, parse_file
from app.retrieval.bm25 import BM25Index
from app.retrieval.vector import VectorIndex, create_embedding_provider

logger = logging.getLogger(__name__)


class KnowledgeIndex:
    def __init__(self):
        cfg = get_config()
        data_cfg = cfg.sub("data")
        self.documents_dir = Path(data_cfg.get("documents_dir", "data/documents"))
        self.index_dir = Path(data_cfg.get("index_dir", "data/index"))
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._store_file = self.index_dir / "store.json"

        rcfg = cfg.sub("retrieval")
        self._chunk_size = int(rcfg.get("chunk_size", 400))
        self._overlap = int(rcfg.get("chunk_overlap", 80))

        # --- 核心存储结构 ---
        self._chunks: dict[str, Chunk] = {}          # chunk_id -> Chunk
        self._doc_chunks: dict[str, list[str]] = {}  # doc_id -> [chunk_id]（哈希索引）
        self._docs: dict[str, dict] = {}              # doc_id -> 文档元信息
        self._content_hash: dict[str, str] = {}       # content_hash -> chunk_id（去重）
        self._lock = threading.RLock()

        self.bm25_index = BM25Index()
        self.vector_index = VectorIndex(create_embedding_provider(cfg.sub("embedding")))
        self._load()

    # ---------- 持久化 ----------

    def _dump(self) -> None:
        payload = {
            "docs": self._docs,
            "chunks": [
                {**asdict(c), "content_hash": self._hash(c.text)} for c in self._chunks.values()
            ],
        }
        tmp = self._store_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._store_file)

    def _load(self) -> None:
        if not self._store_file.exists():
            return
        try:
            payload = json.loads(self._store_file.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error("索引文件损坏，将重建: %s", e)
            return
        self._docs = payload.get("docs", {})
        with self._lock:
            for c in payload.get("chunks", []):
                chunk = Chunk(
                    chunk_id=c["chunk_id"],
                    doc_id=c["doc_id"],
                    doc_name=c["doc_name"],
                    seq=c["seq"],
                    text=c["text"],
                    start=c.get("start", 0),
                )
                self._register_chunk(chunk, c.get("content_hash"))

    def _register_chunk(self, chunk: Chunk, content_hash: str | None = None) -> None:
        ch = content_hash or self._hash(chunk.text)
        self._chunks[chunk.chunk_id] = chunk
        self._doc_chunks.setdefault(chunk.doc_id, []).append(chunk.chunk_id)
        self._content_hash.setdefault(ch, chunk.chunk_id)
        self.bm25_index.add(chunk.chunk_id, chunk.text)
        self.vector_index.add(chunk.chunk_id, chunk.text)

    @staticmethod
    def _hash(text: str) -> str:
        import hashlib

        return hashlib.md5(text.strip().encode("utf-8")).hexdigest()

    # ---------- 写操作 ----------

    def add_file(self, path: str | Path) -> dict:
        doc = parse_file(path)
        return self.add_document(doc)

    def add_document(self, doc: ParsedDocument) -> dict:
        chunks = chunk_document(doc, self._chunk_size, self._overlap)
        added, skipped = 0, 0
        with self._lock:
            self._docs[doc.doc_id] = {
                "doc_id": doc.doc_id,
                "name": doc.name,
                "source": doc.source,
                "n_chunks": len(chunks),
                "chars": len(doc.text),
            }
            for c in chunks:
                ch = self._hash(c.text)
                if ch in self._content_hash:  # 内容哈希去重
                    skipped += 1
                    continue
                self._register_chunk(c, ch)
                added += 1
            # 修正块计数
            self._docs[doc.doc_id]["n_chunks"] = len(self._doc_chunks.get(doc.doc_id, []))
            self._dump()
        logger.info("入库 %s: %d 块新增, %d 块重复跳过", doc.name, added, skipped)
        return {
            "doc_id": doc.doc_id,
            "name": doc.name,
            "added_chunks": added,
            "skipped_chunks": skipped,
            "total_chunks": len(self._chunks),
        }

    def remove_document(self, doc_id: str) -> bool:
        with self._lock:
            if doc_id not in self._docs:
                return False
            for cid in self._doc_chunks.get(doc_id, []):
                chunk = self._chunks.pop(cid, None)
                if chunk:
                    ch = self._hash(chunk.text)
                    if self._content_hash.get(ch) == cid:
                        del self._content_hash[ch]
                self.bm25_index.remove(cid)
                self.vector_index.remove(cid)
            self._doc_chunks.pop(doc_id, None)
            self._docs.pop(doc_id, None)
            self._dump()
        return True

    # ---------- 读操作 ----------

    def text(self, chunk_id: str) -> str:
        c = self._chunks.get(chunk_id)
        return c.text if c else ""

    def get(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def chunk_ids_of_doc(self, doc_id: str) -> list[str]:
        return list(self._doc_chunks.get(doc_id, []))

    def documents(self) -> list[dict]:
        return [
            {**d, "chunks": len(self._doc_chunks.get(d["doc_id"], []))}
            for d in self._docs.values()
        ]

    @property
    def n_chunks(self) -> int:
        return len(self._chunks)

    @property
    def n_docs(self) -> int:
        return len(self._docs)
