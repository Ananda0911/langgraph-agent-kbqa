/* 企业知识库问答系统 - 前端逻辑 */
const $ = (sel) => document.querySelector(sel);

let sessionId = localStorage.getItem("kbqa_session") || "";
let busy = false;

/* ---------------- 初始化 ---------------- */

async function refreshStats() {
  try {
    const s = await (await fetch("/api/stats")).json();
    $("#chip-llm").textContent = s.llm_mode === "mock" ? "LLM: Mock 模式" : `LLM: ${s.model}`;
    $("#chip-llm").className = "chip " + (s.llm_mode === "mock" ? "warn" : "ok");
    $("#chip-docs").textContent = `文档 ${s.documents}`;
    $("#chip-chunks").textContent = `知识块 ${s.chunks}`;
  } catch (e) { /* ignore */ }
}

/* ---------------- 模型切换（多模型支持） ---------------- */

async function loadModels() {
  try {
    const m = await (await fetch("/api/config/models")).json();
    const sel = $("#model-select");
    sel.innerHTML = Object.entries(m.profiles).map(([k, v]) =>
      `<option value="${k}">${v.description || k}</option>`).join("");
    sel.value = m.current;
    sel.title = `当前模型: ${m.current_model}`;
    sel.onchange = async () => {
      const r = await fetch("/api/config/models", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ profile: sel.value }),
      });
      const d = await r.json();
      if (d.ok) { sel.title = `当前模型: ${d.current_model}`; refreshStats(); }
    };
  } catch (e) { /* ignore */ }
}

/* ---------------- Agent 轨迹面板 ---------------- */

const TRACE_LABEL = {
  intent: "🎯 意图识别",
  plan: "📋 自主规划",
  tool_call: "🔧 工具调用",
  observation: "👁️ 执行观察",
  reflection: "🤔 反思评估",
  replan: "🔁 补充计划",
  answer: "✅ 生成回答",
};

const TOOL_NAME = {
  kb_search: "知识库检索",
  db_query: "数据库查询",
  web_search: "联网搜索",
  calculator: "计算器",
  mcp_contacts: "MCP 通讯录",
};

function renderTrace(trace, agent) {
  if (!trace || !trace.length) return "";
  const steps = trace.map(t => {
    const label = TRACE_LABEL[t.ts_type] || t.ts_type;
    let body = "";
    const c = t.content;
    if (t.ts_type === "intent") {
      body = `类型: <b>${c.type}</b> · 工具: ${(c.tools || []).map(x => TOOL_NAME[x] || x).join("、") || "无需工具"}<br><span class="dim">${escapeHtml(c.reason || "")}</span>`;
    } else if (t.ts_type === "plan") {
      body = (Array.isArray(c) ? c : []).map((s, i) =>
        `<div class="plan-step">Step ${i + 1}: <b>${TOOL_NAME[s.tool] || s.tool}</b> — ${escapeHtml(s.description || "")}</div>`).join("");
    } else if (t.ts_type === "tool_call") {
      body = `Step ${c.step}: <b>${TOOL_NAME[c.tool] || c.tool}</b><br><span class="dim">${escapeHtml(JSON.stringify(c.args || {}))}</span>`;
    } else if (t.ts_type === "observation") {
      body = `<div class="obs ${t.meta && t.meta.ok ? "ok" : "fail"}">${escapeHtml(String(c).slice(0, 400))}</div>`;
    } else if (t.ts_type === "reflection") {
      body = `评估: <b>${c.pass ? "通过 ✅" : "需补充 ❌"}</b>${c.feedback ? " — " + escapeHtml(c.feedback) : ""}`;
    } else if (t.ts_type === "replan") {
      body = `补充工具: ${(Array.isArray(c) ? c : []).map(x => TOOL_NAME[x] || x).join("、")}`;
    } else {
      body = escapeHtml(String(c).slice(0, 200));
    }
    return `<div class="trace-step"><div class="trace-label">${label}</div><div class="trace-body">${body}</div></div>`;
  }).join("");
  const meta = agent ? `<div class="trace-meta">Agent 耗时 ${agent.latency_ms}ms · 反思轮次 ${agent.iterations} · 工具链 ${(agent.tools_used || []).map(x => TOOL_NAME[x] || x).join(" → ") || "无"}</div>` : "";
  return `<details class="agent-trace">
    <summary>🤖 Agent 执行轨迹（点击展开）${agent ? ` · ${(agent.tools_used || []).map(x => TOOL_NAME[x] || x).join(" → ") || "直接回答"}` : ""}</summary>
    <div class="trace-steps">${steps}</div>${meta}
  </details>`;
}

/* ---------------- 聊天 ---------------- */

function renderSources(sources) {
  if (!sources || !sources.length) return "";
  const items = sources.map(s => `
    <div class="source-item">
      <div class="src-head">
        <span class="src-tag">[${s.idx}]</span>
        <span class="src-name">${s.doc_name}</span>
        <span class="src-meta">第${(s.seq ?? 0) + 1}段</span>
        ${s.bm25 != null ? `<span class="score-pill b" title="BM25 分数">B ${s.bm25}</span>` : ""}
        ${s.vector != null ? `<span class="score-pill v" title="向量分数">V ${s.vector}</span>` : ""}
        <span class="score-pill" title="相关度">R ${s.score}</span>
      </div>
      <div class="src-text">${s.text}</div>
    </div>`).join("");
  return `<div class="sources">
    <div class="sources-title" onclick="this.nextElementSibling.classList.toggle('source-list')">📎 引用来源（点击展开/收起）</div>
    <div class="source-list" style="display:none">${items}</div>
  </div>`;
}

function addMessage(role, html) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  wrap.innerHTML = `
    <div class="avatar">${role === "user" ? "我" : "AI"}</div>
    <div class="bubble">${html}</div>`;
  $("#messages").appendChild(wrap);
  $("#messages").scrollTop = $("#messages").scrollHeight;
  return wrap;
}

function showTyping() {
  const wrap = addMessage("assistant", `<div class="typing"><span></span><span></span><span></span></div>`);
  return wrap.querySelector(".bubble");
}

async function send() {
  const input = $("#input");
  const msg = input.value.trim();
  if (!msg || busy) return;
  busy = true;
  $("#send").disabled = true;
  input.value = "";
  input.style.height = "auto";
  addMessage("user", msg);

    const typing = showTyping();
  try {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: msg, session_id: sessionId }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "请求失败");
    sessionId = data.session_id;
    localStorage.setItem("kbqa_session", sessionId);
    const isAgent = !!data.trace;
    const metaLine = isAgent
      ? `Agent 耗时 ${data.agent.latency_ms}ms · 模型 ${data.model}${data.llm_mode === "mock" ? " · Mock 模式" : ""}`
      : `检索耗时 ${data.retrieval.latency_ms}ms · 来源 ${data.sources.length} 块${data.llm_mode === "mock" ? " · Mock 模式" : ""}`;
    typing.innerHTML = (isAgent ? renderTrace(data.trace, data.agent) : "") +
      escapeHtml(data.answer) + renderSources(data.sources) +
      `<div class="retrieval-meta">${metaLine}</div>`;
  } catch (e) {
    typing.innerHTML = `<span style="color:var(--danger)">出错了：${escapeHtml(e.message)}</span>`;
  } finally {
    busy = false;
    $("#send").disabled = false;
    $("#messages").scrollTop = $("#messages").scrollHeight;
    loadMemory();
  }
}

function escapeHtml(s) {
  return (s || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

/* ---------------- 文档库 ---------------- */

async function loadDocs() {
  const docs = await (await fetch("/api/documents")).json();
  const list = $("#doc-list");
  if (!docs.length) {
    list.innerHTML = `<div class="empty-tip">暂无文档</div>`;
    return;
  }
  list.innerHTML = docs.map(d => `
    <div class="doc-item">
      <div>
        <div class="doc-name">${escapeHtml(d.name)}</div>
        <div class="doc-meta">${d.chunks} 块 · ${d.chars} 字</div>
      </div>
      <button class="doc-del" onclick="delDoc('${d.doc_id}')">删除</button>
    </div>`).join("");
}

async function delDoc(id) {
  if (!confirm("确认删除该文档及其索引？")) return;
  await fetch(`/api/documents/${id}`, { method: "DELETE" });
  loadDocs();
  refreshStats();
}

async function uploadFiles(files) {
  if (!files.length) return;
  const fd = new FormData();
  for (const f of files) fd.append("files", f);
  const zone = $("#drop-zone");
  zone.innerHTML = `<p style="color:var(--accent)">⏳ 解析入库中…</p>`;
  try {
    const resp = await fetch("/api/upload", { method: "POST", body: fd });
    const data = await resp.json();
    const ok = (data.uploaded || []).map(u => `✅ ${u.name}（${u.added_chunks} 块）`).join("<br>");
    const err = (data.errors || []).map(e => `❌ ${e.file}: ${e.error}`).join("<br>");
    alert((ok || "无成功入库项") + (err ? "\n\n" + err : ""));
  } catch (e) {
    alert("上传失败: " + e.message);
  }
  restoreZone();
  loadDocs();
  refreshStats();
}

function restoreZone() {
  $("#drop-zone").innerHTML = `<p>拖拽或点击上传</p><p class="hint">支持 PDF / DOCX / TXT / MD</p>`;
}

/* ---------------- 评估 ---------------- */

async function runEval() {
  const btn = $("#run-eval");
  const out = $("#eval-result");
  btn.disabled = true;
  btn.textContent = "评估中…";
  try {
    const r = await (await fetch("/api/eval/run", { method: "POST" })).json();
    const hitPct = (r["hit_rate@5"] * 100).toFixed(1);
    out.innerHTML = `
      <div class="metric-grid">
        <div class="metric"><div class="val">${hitPct}%</div><div class="label">Hit Rate@5</div></div>
        <div class="metric"><div class="val green">${r.mrr.toFixed(3)}</div><div class="label">MRR</div></div>
        <div class="metric"><div class="val">${r.avg_latency_ms}ms</div><div class="label">平均检索延迟</div></div>
        <div class="metric"><div class="val">${r.total}</div><div class="label">Golden 题目数</div></div>
      </div>
      ${r.details.map(d => `
        <div class="eval-detail">
          <div class="q">${escapeHtml(d.question)}<span class="badge ${d.hit ? "hit" : "miss"}">${d.hit ? "命中 @" + d.first_hit_rank : "未命中"}</span></div>
          <div class="r">Top 检索: ${d.top_docs.map(escapeHtml).join(" / ") || "无"}</div>
        </div>`).join("")}
    `;
  } catch (e) {
    out.innerHTML = `<div class="empty-tip">评估失败: ${escapeHtml(e.message)}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "运行评估";
  }
}

/* ---------------- 会话记忆 ---------------- */

async function loadMemory() {
  if (!sessionId) {
    $("#memory-view").innerHTML = `<div class="empty-tip">暂无会话</div>`;
    return;
  }
  const [m, am] = await Promise.all([
    (await fetch(`/api/memory/${sessionId}`)).json(),
    (await fetch(`/api/action_memory/${sessionId}`)).json(),
  ]);
  const short = (m.short_term || []).map(t =>
    `<div class="mem-item"><span class="who">${t.role === "user" ? "用户" : "助手"}:</span> ${escapeHtml(t.content.slice(0, 120))}</div>`).join("");
  const long = (m.long_term_summaries || []).map(s =>
    `<div class="mem-item sum">📝 ${escapeHtml(s)}</div>`).join("");
  const actions = (am.trace || []).map(t =>
    `<div class="mem-item act">${TRACE_LABEL[t.type] || t.type}: ${escapeHtml(typeof t.content === "string" ? t.content.slice(0, 100) : JSON.stringify(t.content).slice(0, 100))}</div>`).join("");
  $("#memory-view").innerHTML = `
    <div class="mem-section">
      <h4>行动记忆（Agent 轨迹 · ${(am.trace || []).length} 条）</h4>
      ${actions || `<div class="empty-tip">暂无 Agent 执行轨迹</div>`}
    </div>
    <div class="mem-section">
      <h4>长期记忆（压缩摘要 · ${m.long_term_summaries.length} 条）</h4>
      ${long || `<div class="empty-tip">暂无压缩摘要（轮数超过 ${m.compress_threshold} 后自动生成）</div>`}
    </div>
    <div class="mem-section">
      <h4>短期记忆（最近 ${m.short_term_rounds} 轮 · 当前 ${m.short_term_count} 轮）</h4>
      ${short || `<div class="empty-tip">暂无对话</div>`}
    </div>`;
}

/* ---------------- 事件绑定 ---------------- */

document.querySelectorAll(".tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
    tab.classList.add("active");
    $(`#panel-${tab.dataset.tab}`).classList.add("active");
    if (tab.dataset.tab === "memory") loadMemory();
  });
});

$("#send").addEventListener("click", send);
$("#input").addEventListener("input", function () {
  this.style.height = "auto";
  this.style.height = Math.min(this.scrollHeight, 140) + "px";
  $("#send").disabled = !this.value.trim() || busy;
});
$("#input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
});

const zone = $("#drop-zone");
const fileInput = $("#file-input");
zone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => { uploadFiles(fileInput.files); fileInput.value = ""; });
zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("drag"); });
zone.addEventListener("dragleave", () => zone.classList.remove("drag"));
zone.addEventListener("drop", (e) => {
  e.preventDefault();
  zone.classList.remove("drag");
  uploadFiles(e.dataTransfer.files);
});

$("#run-eval").addEventListener("click", runEval);
$("#clear-memory").addEventListener("click", async () => {
  if (!sessionId) return;
  await fetch(`/api/memory/${sessionId}`, { method: "DELETE" });
  loadMemory();
});

/* ---------------- 启动 ---------------- */
refreshStats();
loadModels();
loadDocs();
