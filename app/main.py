"""FastAPI 主应用：LangGraph Agent 企业知识库问答系统。

架构：
  用户问题 → 意图识别 → 自主规划 → 工具执行（RAG检索/数据库/联网/MCP/计算器）
                     ↑          ↓
                     ←—— 反思评估 ←——        → 回答生成（带引用来源 + 轨迹）
"""
from __future__ import annotations

import logging
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.agent.action_memory import ActionMemory
from app.agent.graph import build_agent_graph
from app.agent.nodes import AgentNodes
from app.agent.state import initial_state
from app.agent.tools import (
    CalculatorTool,
    DBQueryTool,
    KBSearchTool,
    MCPTool,
    ToolRegistry,
    WebSearchTool,
    build_mcp_tools_from_config,
)
from app.config import BASE_DIR, get_config
from app.evaluation.evaluator import run_evaluation
from app.ingestion.indexer import KnowledgeIndex
from app.llm.client import LLMClient
from app.llm.prompt import SYSTEM_PROMPT, build_rag_prompt
from app.memory.manager import MemoryManager
from app.retrieval.retriever import HybridRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("kbqa")


class AppState:
    """应用级单例：索引、检索器、LLM、分级记忆 + LangGraph Agent。"""

    def __init__(self):
        self.cfg = get_config()
        self.llm = LLMClient(self.cfg.sub("llm"))
        self.index = KnowledgeIndex()
        self.retriever = HybridRetriever(
            self.index.bm25_index,
            self.index.vector_index,
            self.index,
            self.cfg.sub("retrieval"),
            llm_client=self.llm,
        )
        self.memory = MemoryManager(self.llm)

        # ---------- LangGraph Agent ----------
        self.agent_cfg = self.cfg.sub("agent") or {}
        self.action_memory = ActionMemory()
        self.tools = ToolRegistry()
        self.tools.register(KBSearchTool(self.retriever))
        try:
            self.tools.register(DBQueryTool())
        except Exception as e:
            logger.error("数据库工具初始化失败: %s", e)
        self.tools.register(WebSearchTool())
        self.tools.register(CalculatorTool())
        for t in build_mcp_tools_from_config():
            self.tools.register(t)
        self.nodes = AgentNodes(self.llm, self.tools, self.action_memory, self.agent_cfg)

        # ---------- 多模型支持 ----------
        self.model_profiles: dict = self.cfg.sub("llm.models") or {}
        if not self.model_profiles:
            self.model_profiles = {"default": {"model": self.llm.model or "qwen-plus", "description": "默认模型"}}
        self.current_profile = "default"

        self.agent_graph = None
        if self.agent_cfg.get("enabled", True):
            try:
                self.agent_graph = build_agent_graph(self.nodes)
            except Exception as e:
                logger.error("Agent 图构建失败，回退纯 RAG 管道: %s", e)

        self._apply_model_profile()

    # ---------- 多模型 ----------

    def _apply_model_profile(self, profile: str | None = None) -> None:
        if profile:
            self.current_profile = profile
        p = self.model_profiles.get(self.current_profile) or {}
        model_id = p.get("model", self.llm.model)
        if model_id:
            self.llm.set_model(model_id)
            self.nodes.model_override = model_id

    def seed_documents(self) -> None:
        """启动时预加载 seed_docs 中的示例文档（金融研报/条款等）。"""
        seed_dir = BASE_DIR / self.cfg.get("data.seed_dir", "seed_docs")
        if not seed_dir.exists():
            return
        known = {d["name"] for d in self.index.documents()}
        for f in sorted(seed_dir.iterdir()):
            if f.suffix.lower() not in (".txt", ".md", ".pdf", ".docx"):
                continue
            if f.name in known:
                continue
            try:
                result = self.index.add_file(f)
                logger.info("预加载示例文档: %s (%d 块)", f.name, result["added_chunks"])
            except Exception as e:
                logger.error("预加载 %s 失败: %s", f.name, e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state = AppState()
    state.seed_documents()
    app.state.kb = state
    logger.info(
        "系统就绪: %d 文档 / %d 块 | LLM=%s(%s) | Embedding=%s | Agent=%s | 工具=%s",
        state.index.n_docs,
        state.index.n_chunks,
        state.llm.mode,
        state.llm.model,
        state.index.vector_index.provider.name,
        "on" if state.agent_graph else "off",
        ", ".join(state.tools.names()),
    )
    yield


app = FastAPI(title="LangGraph Agent 企业知识库问答系统", version="2.0.0", lifespan=lifespan)


def kb(request: Request) -> AppState:
    return request.app.state.kb


# ---------------- 请求模型 ----------------

class ChatRequest(BaseModel):
    message: str
    session_id: str = ""


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class ModelSwitchRequest(BaseModel):
    profile: str


# ---------------- 对话（Agent 管道） ----------------

@app.post("/api/chat")
def chat(req: ChatRequest, request: Request):
    state = kb(request)
    if not req.message.strip():
        raise HTTPException(400, "问题不能为空")
    session_id = req.session_id or uuid.uuid4().hex[:12]

    # ============ LangGraph Agent 管道 ============
    if state.agent_graph is not None:
        t0 = time.perf_counter()
        # 多轮对话管理：分级记忆 + 行动记忆 注入上下文
        memory_ctx = state.memory.build_context(session_id)
        action_ctx = state.action_memory.recent_context(session_id)
        if action_ctx:
            memory_ctx = (memory_ctx + "\n\n" if memory_ctx else "") + "历史行动轨迹：\n" + action_ctx

        astate = initial_state(
            query=req.message,
            session_id=session_id,
            memory_ctx=memory_ctx,
            model=state.current_profile,
        )
        try:
            result = state.agent_graph.invoke(astate)
        except Exception as e:
            logger.exception("Agent 执行失败: %s", e)
            raise HTTPException(500, f"Agent 执行失败: {e}")

        answer = result.get("final_answer", "")
        state.memory.add_turn(session_id, "user", req.message)
        state.memory.add_turn(session_id, "assistant", answer)
        latency = round((time.perf_counter() - t0) * 1000, 1)
        return {
            "session_id": session_id,
            "answer": answer,
            "sources": result.get("sources", []),
            "intent": result.get("intent", {}),
            "trace": result.get("trace", []),
            "agent": {
                "latency_ms": latency,
                "iterations": result.get("iterations", 0),
                "reflection": result.get("reflection", {}),
                "tools_used": [o["tool"] for o in result.get("observations", [])],
            },
            "llm_mode": state.llm.mode,
            "model": state.llm.model,
            "memory": state.memory.snapshot(session_id),
        }

    # ============ 兼容回退：纯 RAG 管道（agent.enabled=false） ============
    result = state.retriever.search(req.message)
    if not result.hits:
        answer = "知识库中暂无与该问题相关的文档内容，请先上传相关文档或换个问法。"
        state.memory.add_turn(session_id, "user", req.message)
        state.memory.add_turn(session_id, "assistant", answer)
        return {
            "session_id": session_id,
            "answer": answer,
            "sources": [],
            "retrieval": {"latency_ms": result.latency_ms, "debug": result.debug},
            "llm_mode": state.llm.mode,
        }

    hits = [h.to_dict() for h in result.hits]
    memory_ctx = state.memory.build_context(session_id)
    prompt = build_rag_prompt(req.message, hits, memory_ctx)
    answer = state.llm.chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )
    state.memory.add_turn(session_id, "user", req.message)
    state.memory.add_turn(session_id, "assistant", answer)

    sources = [
        {
            "idx": i + 1,
            "doc_name": h["doc_name"],
            "doc_id": h["doc_id"],
            "seq": h["seq"],
            "score": round(h["score"], 4),
            "bm25": h["bm25"],
            "vector": h["vector"],
            "text": h["text"][:160] + ("…" if len(h["text"]) > 160 else ""),
        }
        for i, h in enumerate(hits)
    ]
    return {
        "session_id": session_id,
        "answer": answer,
        "sources": sources,
        "retrieval": {"latency_ms": result.latency_ms, "debug": result.debug},
        "llm_mode": state.llm.mode,
        "memory": state.memory.snapshot(session_id),
    }


# ---------------- Agent 工具 / 模型配置 / 行动记忆 ----------------

@app.get("/api/agent/tools")
def agent_tools(request: Request):
    state = kb(request)
    return {
        "graph": "langgraph" if state.agent_graph is not None else "off",
        "tools": [t.to_dict() for t in state.tools.all()],
    }


@app.get("/api/config/models")
def get_models(request: Request):
    state = kb(request)
    return {
        "profiles": {k: v for k, v in state.model_profiles.items()},
        "current": state.current_profile,
        "current_model": state.llm.model,
    }


@app.put("/api/config/models")
def switch_model(req: ModelSwitchRequest, request: Request):
    state = kb(request)
    if req.profile not in state.model_profiles:
        raise HTTPException(404, f"模型 profile 不存在: {req.profile}")
    state._apply_model_profile(req.profile)
    return {"ok": True, "current": state.current_profile, "current_model": state.llm.model}


@app.get("/api/action_memory/{session_id}")
def get_action_memory(session_id: str, request: Request):
    return {"session_id": session_id, "trace": kb(request).action_memory.trace(session_id)}


@app.delete("/api/action_memory/{session_id}")
def clear_action_memory(session_id: str, request: Request):
    kb(request).action_memory.clear(session_id)
    return {"ok": True}


# ---------------- 检索调试 ----------------

@app.post("/api/search")
def search(req: SearchRequest, request: Request):
    state = kb(request)
    result = state.retriever.search(req.query, top_k=req.top_k)
    return {
        "hits": [h.to_dict() for h in result.hits],
        "latency_ms": result.latency_ms,
        "debug": result.debug,
    }


# ---------------- 文档管理 ----------------

ALLOWED = {".pdf", ".docx", ".txt", ".md", ".markdown"}


@app.post("/api/upload")
def upload(request: Request, files: list[UploadFile] = File(...)):
    state = kb(request)
    results, errors = [], []
    for f in files:
        suffix = Path(f.filename or "").suffix.lower()
        if suffix not in ALLOWED:
            errors.append({"file": f.filename, "error": f"不支持的类型 {suffix}，支持 pdf/docx/txt/md"})
            continue
        try:
            dest = state.index.documents_dir / Path(f.filename).name
            with open(dest, "wb") as out:
                shutil.copyfileobj(f.file, out)
            results.append(state.index.add_file(dest))
        except Exception as e:
            errors.append({"file": f.filename, "error": str(e)})
        finally:
            try:
                f.file.close()
            except Exception:
                pass
    return {"uploaded": results, "errors": errors}


@app.get("/api/documents")
def documents(request: Request):
    return kb(request).index.documents()


@app.delete("/api/documents/{doc_id}")
def remove_document(doc_id: str, request: Request):
    if not kb(request).index.remove_document(doc_id):
        raise HTTPException(404, "文档不存在")
    return {"ok": True}


# ---------------- 会话记忆 ----------------

@app.get("/api/memory/{session_id}")
def get_memory(session_id: str, request: Request):
    return kb(request).memory.snapshot(session_id)


@app.delete("/api/memory/{session_id}")
def clear_memory(session_id: str, request: Request):
    kb(request).memory.clear(session_id)
    kb(request).action_memory.clear(session_id)
    return {"ok": True}


# ---------------- 评估流水线 ----------------

@app.post("/api/eval/run")
def eval_run(request: Request):
    return run_evaluation(kb(request))


# ---------------- 系统状态 ----------------

@app.get("/api/stats")
def stats(request: Request):
    state = kb(request)
    return {
        "documents": state.index.n_docs,
        "chunks": state.index.n_chunks,
        "llm_mode": state.llm.mode,
        "model": state.llm.model,
        "model_profile": state.current_profile,
        "agent": state.agent_graph is not None,
        "agent_backend": getattr(state.agent_graph, "backend", "langgraph") if state.agent_graph else "off",
        "tools": state.tools.names(),
        "embedding_provider": state.index.vector_index.provider.name,
        "reranker": state.retriever._reranker.name,
        "fusion": state.cfg.get("retrieval.fusion"),
    }


# ---------------- 静态页面 ----------------

app.mount("/web", StaticFiles(directory=str(BASE_DIR / "web")), name="web")


@app.get("/")
def index_page():
    return FileResponse(BASE_DIR / "web" / "index.html")


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    logger.exception("内部错误: %s", exc)
    return JSONResponse(status_code=500, content={"detail": f"内部错误: {exc}"})
