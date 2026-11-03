"""联网搜索工具：ddgs（DuckDuckGo 免费，无需 API Key）。"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.config import BASE_DIR, get_config

logger = logging.getLogger(__name__)


def _import_ddgs():
    """优先从项目 libs 目录加载 ddgs（沙箱环境安装位置），再回退全局安装。"""
    libs = BASE_DIR / "libs"
    if libs.exists():
        import sys

        if str(libs) not in sys.path:
            sys.path.insert(0, str(libs))
    try:
        from ddgs import DDGS  # type: ignore

        return DDGS
    except ImportError:
        pass
    try:
        from duckduckgo_search import DDGS  # type: ignore

        return DDGS
    except ImportError:
        return None


class WebSearchTool:
    """联网搜索（DuckDuckGo 免费方案，无需 Key）。"""

    name = "web_search"
    description = "联网搜索最新信息（新闻、时事、实时数据等知识库无法覆盖的内容）。"

    def __init__(self):
        cfg = get_config()
        self._max_results = int(cfg.get("agent.web_search.max_results", 5))

    def args_hint(self) -> str:
        return '{"query": "搜索关键词", "max_results": 5(可选)}'

    def run(self, args: dict[str, Any]) -> dict[str, Any]:
        query = (args.get("query") or args.get("q") or "").strip()
        if not query:
            return {"output": "错误：缺少 query 参数", "sources": [], "error": "missing query"}
        DDGS = _import_ddgs()
        if DDGS is None:
            return {
                "output": "联网搜索暂不可用：未安装 ddgs 包（pip install ddgs）。",
                "sources": [],
                "error": "ddgs not installed",
            }
        max_results = int(args.get("max_results") or self._max_results)
        try:
            with DDGS() as d:
                results = list(d.text(query, max_results=max_results))
        except Exception as e:
            logger.warning("联网搜索失败: %s", e)
            return {"output": f"联网搜索失败：{e}", "sources": [], "error": str(e)}
        if not results:
            return {"output": f"未搜索到与「{query}」相关的结果", "sources": []}
        lines = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            href = r.get("href") or r.get("url", "")
            body = r.get("body") or r.get("snippet", "")
            lines.append(f"【网页{i}】{title}\n链接: {href}\n摘要: {body}")
        return {"output": "\n\n".join(lines), "sources": []}

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "description": self.description, "args_hint": self.args_hint()}
