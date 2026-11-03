"""全局配置加载。"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

BASE_DIR = Path(__file__).resolve().parent.parent

_DEFAULTS: dict[str, Any] = {
    "server": {"host": "127.0.0.1", "port": 8000},
    "llm": {
        "api_key": "",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "temperature": 0.3,
        "max_tokens": 1024,
        "timeout": 60,
    },
    "embedding": {
        "provider": "local_hash",
        "api_key": "",
        "base_url": "",
        "model": "",
        "dim": 2048,
    },
    "retrieval": {
        "chunk_size": 400,
        "chunk_overlap": 80,
        "candidate_k": 10,
        "top_k": 5,
        "fusion": {"method": "weighted", "bm25_weight": 0.5, "vector_weight": 0.5, "rrf_k": 60},
        "rerank": {"enabled": True, "provider": "llm"},
    },
    "memory": {
        "short_term_rounds": 6,
        "long_term_max_summaries": 20,
        "compress_threshold": 8,
    },
    "data": {"documents_dir": "data/documents", "index_dir": "data/index", "seed_dir": "seed_docs"},
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class Config:
    """点路径访问的配置对象，如 cfg.get('retrieval.top_k')。"""

    def __init__(self, data: dict[str, Any], path: Path | None = None):
        self._data = data
        self.path = path

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def sub(self, prefix: str) -> dict[str, Any]:
        node = self.get(prefix, {})
        return node if isinstance(node, dict) else {}

    @property
    def raw(self) -> dict[str, Any]:
        return self._data


def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path) if path else BASE_DIR / "config.yaml"
    data: dict[str, Any] = {}
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    merged = _deep_merge(_DEFAULTS, data)
    return Config(merged, cfg_path)


_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(cfg: Config) -> None:
    global _config
    _config = cfg
