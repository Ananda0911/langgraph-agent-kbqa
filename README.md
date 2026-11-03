# LangGraph Agent 企业文档知识库问答系统

基于 **LangGraph** 搭建的企业级 Agent 问答系统：实现了**意图识别、自主规划、工具调用、行动记忆、反思评估**，以及**多轮对话管理、MCP 服务集成、多模型支持、本地 RAG 检索、数据库查询、联网搜索**等能力。

```
用户问题
   │
   ▼
🎯 意图识别 ──(复杂任务)──▶ 📋 自主规划 ──┐
   │ (单工具/闲聊)                        │
   │                                     ▼
   │            ┌──────────── ◀── 🔁 反思评估
   │            ▼                          │ 不通过(≤3轮)
   │        🔧 工具执行 ◀── 补充步骤 ──────┘
   │            │
   ▼            ▼
        ✅ 回答生成（带引用来源 + 全程轨迹）
```

## 核心功能

| 能力 | 说明 |
|---|---|
| **意图识别** | LLM 将问题路由到最合适的工具（知识库/数据库/联网/通讯录/计算器/闲聊），无 Key 时降级关键词规则 |
| **自主规划** | 复杂任务自动拆解为多步工具调用计划（LangGraph planner 节点） |
| **工具调用** | 5 类内置工具：`kb_search` / `db_query` / `web_search` / `calculator` / `mcp_contacts` |
| **行动记忆** | 每个会话记录完整行动轨迹（意图→计划→工具→观察→反思），注入后续决策上下文，前端可视化展示 |
| **反思评估** | LLM 检查观察结果是否足以回答；不足时自动生成补充步骤重试（上限 3 轮防死循环） |
| **多轮对话管理** | 分级记忆：短期 List 保留最近 N 轮原文，超阈值自动 LLM 压缩为长期摘要（支持 Redis 切换） |
| **MCP 服务集成** | 轻量 MCP 客户端（JSON-RPC over stdio），启动时自动发现外部 MCP Server 工具清单；内置企业通讯录演示服务 |
| **多模型支持** | qwen-plus / qwen-turbo / qwen-max 等 profiles 配置化，Web 界面运行时热切换 |
| **本地 RAG 检索** | BM25 + 向量（本地哈希 Embedding）双路召回 → 加权/RRF 融合 → LLM Rerank 精排 → Top-K |
| **数据库查询** | SQLite 只读查询，仅允许 SELECT + 表白名单（employees / orders 示例库自动初始化） |
| **联网搜索** | DuckDuckGo 免费搜索（ddgs，无需 API Key） |
| **评估流水线** | Golden Dataset 自动化评估：Hit Rate@5 / MRR / 检索延迟 |

## 技术栈

- **Agent 编排**: LangGraph（StateGraph 条件边状态机；未安装时自动降级内置 MiniGraph）
- **后端**: FastAPI + Uvicorn
- **LLM**: OpenAI 兼容 API（DashScope Qwen 等），无 Key 自动 Mock 降级，全链路可演示
- **检索**: jieba BM25 + 本地哈希向量 + 融合排序 + LLM Rerank
- **文档解析**: PDF / DOCX / TXT / Markdown，分块入库、增量索引、持久化
- **MCP**: JSON-RPC over stdio（initialize / tools/list / tools/call）

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> 提示：若 pip 安装 `ddgs` 报权限错误，可装到项目本地目录（已配好自动加载）：
> `pip install ddgs --target ./libs`

### 2. 配置

```bash
copy config.example.yaml config.yaml
# 编辑 config.yaml，填入 llm.api_key（如阿里 DashScope Key）
# 不填 Key 也能跑：系统自动降级 Mock 模式，全流程可演示
```

### 3. 启动

```bash
python run.py
# 或直接双击 启动系统.bat（Windows 一键启动）
```

浏览器打开 <http://127.0.0.1:8000>，系统预加载金融示例文档（研报/保险条款/报销制度/基金说明）与企业示例数据库（10 员工 / 12 订单）。

### 试试这些问题

- `雇主责任险哪些情况不赔？` → 本地 RAG 检索，带文档引用
- `哪个客户累计销售金额最高？` → 数据库聚合查询（GROUP BY + SUM）
- `最近固态电池行业有什么最新进展？` → 联网搜索
- `查一下技术部周杰的联系方式` → MCP 通讯录服务
- `(450000+380000)*0.13 等于多少` → 计算器
- `那它的保费怎么算？` → 多轮对话指代消解（接上文的"雇主责任险"）

## API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/chat` | Agent 对话（响应含 answer / sources / intent / trace 轨迹） |
| GET | `/api/agent/tools` | 已注册工具清单 |
| GET / PUT | `/api/config/models` | 查询/切换模型 profile（多模型热切换） |
| GET / DELETE | `/api/action_memory/{sid}` | 行动记忆轨迹查看/清空 |
| POST | `/api/search` | 检索调试（BM25+向量+融合+精排明细） |
| POST | `/api/upload` | 上传文档（pdf/docx/txt/md）自动解析入库 |
| POST | `/api/eval/run` | 运行 Golden Dataset 评估（Hit Rate@5 / MRR） |
| GET | `/api/stats` | 系统状态（文档数/模型/Agent 后端/工具列表） |
| GET / DELETE | `/api/memory/{sid}` | 分级记忆查看/清空 |

## 项目结构

```
├── app/
│   ├── agent/                 # LangGraph Agent
│   │   ├── state.py           # AgentState（共享状态）
│   │   ├── nodes.py           # 意图/规划/执行/反思/回答 5 大节点
│   │   ├── graph.py           # LangGraph StateGraph 组装（含 MiniGraph 降级）
│   │   ├── action_memory.py   # 行动记忆
│   │   └── tools/             # kb_search / db_query / web_search / calculator / mcp_client
│   ├── retrieval/             # BM25 + 向量 + 混合融合 + Rerank
│   ├── ingestion/             # 文档解析分块 + 增量索引持久化
│   ├── memory/                # 分级记忆（短期 List + 长期 LLM 摘要，Redis 可选）
│   ├── llm/                   # OpenAI 兼容客户端（Mock 降级）
│   ├── evaluation/            # Golden Dataset 评估流水线
│   └── main.py                # FastAPI 入口
├── mcp_servers/
│   └── contacts_server.py     # MCP 企业通讯录演示服务（stdio JSON-RPC）
├── seed_docs/                 # 金融示例文档
├── web/                       # 聊天界面（Agent 轨迹面板 + 模型切换下拉）
├── config.example.yaml        # 配置模板（复制为 config.yaml 使用）
├── config.yaml                # 本地配置（含 Key，已 gitignore）
└── run.py / 启动系统.bat       # 一键启动
```

## 效果评估

Golden Dataset（12 题）实测：

| 指标 | 结果 |
|---|---|
| Hit Rate@5 | **100%**（12/12 命中） |
| MRR | **1.0**（全部 rank=1） |

Agent 路由回归：6 类意图（RAG/数据库/联网/通讯录/计算/闲聊）100% 正确路由；多步任务经 planner 或反思补充完成；数据库白名单拦截全部危险 SQL。

## 安全设计

- `config.yaml` 含真实 API Key，已通过 `.gitignore` 排除，仓库仅提供 `config.example.yaml` 模板
- 数据库工具：只读连接（`mode=ro`）+ 仅 SELECT + 表白名单 + 禁多语句
- 计算器：ast 白名单沙箱求值，仅算术运算
- 上传文件类型白名单，重命名落盘

## License

MIT
