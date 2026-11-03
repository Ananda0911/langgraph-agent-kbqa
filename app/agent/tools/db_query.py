"""数据库查询工具：SQLite 只读查询（仅 SELECT + 表白名单）。

启动时若 data/company.db 不存在，自动创建并写入示例企业数据
（员工表 + 销售订单表）。
"""
from __future__ import annotations

import logging
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any

from app.config import BASE_DIR, get_config

logger = logging.getLogger(__name__)

ALLOWED_TABLES = {"employees", "orders"}

EMPLOYEES = [
    (1, "张伟", "市场部", "市场总监", "13801010101", "zhangwei@corp.com", "2019-03-12"),
    (2, "李娜", "财务部", "财务经理", "13801010102", "lina@corp.com", "2018-07-01"),
    (3, "王强", "技术部", "架构师", "13801010103", "wangqiang@corp.com", "2020-05-18"),
    (4, "赵敏", "人事部", "HRBP", "13801010104", "zhaomin@corp.com", "2021-09-05"),
    (5, "陈涛", "市场部", "渠道经理", "13801010105", "chentao@corp.com", "2022-01-10"),
    (6, "孙丽", "技术部", "前端工程师", "13801010106", "sunli@corp.com", "2023-04-22"),
    (7, "周杰", "技术部", "算法工程师", "13801010107", "zhoujie@corp.com", "2023-08-15"),
    (8, "吴静", "财务部", "会计", "13801010108", "wujing@corp.com", "2022-11-30"),
    (9, "郑凯", "市场部", "内容策划", "13801010109", "zhengkai@corp.com", "2024-02-26"),
    (10, "冯雪", "人事部", "招聘专员", "13801010110", "fengxue@corp.com", "2024-06-03"),
]

ORDERS = [
    ("SO-2025-001", "华东数控", "工业机器人 R6", 3, 450000.00, "2025-01-08", "华东", 1),
    ("SO-2025-002", "蓝海医疗", "监护仪 VM-8", 20, 380000.00, "2025-01-22", "华北", 5),
    ("SO-2025-003", "星辉物流", "AGV 导航套件", 50, 295000.00, "2025-02-14", "华南", 1),
    ("SO-2025-004", "蓝海医疗", "彩超机 P40", 2, 610000.00, "2025-03-02", "华北", 5),
    ("SO-2025-005", "云顶新能源", "储能控制器 ESS-3", 15, 520000.00, "2025-03-19", "西北", 3),
    ("SO-2025-006", "银河证券", "行情终端 V9", 100, 250000.00, "2025-04-07", "华东", 1),
    ("SO-2025-007", "天工汽车", "车载雷达 DK2", 30, 720000.00, "2025-04-25", "华中", 9),
    ("SO-2025-008", "星辉物流", "分拣机器人 S2", 10, 430000.00, "2025-05-11", "华南", 1),
    ("SO-2025-009", "云顶新能源", "逆变器 INV-5", 25, 310000.00, "2025-05-30", "西北", 5),
    ("SO-2025-010", "银河证券", "量化数据平台", 1, 880000.00, "2025-06-18", "华东", 3),
    ("SO-2025-011", "天工汽车", "域控制器 DC-X", 12, 560000.00, "2025-07-02", "华中", 9),
    ("SO-2025-012", "华东数控", "数控系统 NC-9", 8, 390000.00, "2025-07-21", "华东", 1),
]


def ensure_company_db() -> Path:
    """确保示例企业数据库存在（不存在则自动建库+写入数据）。"""
    data_dir = (BASE_DIR / get_config().get("data.index_dir", "data/index")).parent
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "company.db"
    if db_path.exists():
        return db_path
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """CREATE TABLE employees (
                id INTEGER PRIMARY KEY, name TEXT, department TEXT, title TEXT,
                phone TEXT, email TEXT, hire_date TEXT)"""
        )
        cur.execute(
            """CREATE TABLE orders (
                order_id TEXT PRIMARY KEY, customer TEXT, product TEXT,
                qty INTEGER, amount REAL, order_date TEXT, region TEXT, salesperson_id INTEGER)"""
        )
        cur.executemany("INSERT INTO employees VALUES (?,?,?,?,?,?,?)", EMPLOYEES)
        cur.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?)", ORDERS)
        conn.commit()
        logger.info("已初始化示例企业数据库: %s（employees=%d, orders=%d）", db_path, len(EMPLOYEES), len(ORDERS))
    finally:
        conn.close()
    return db_path


_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|pragma|attach|detach|vacuum|reindex)\b",
    re.I,
)
_TABLE_RE = re.compile(r"\b(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_.]*)", re.I)


class DBQueryTool:
    """企业数据库只读查询（SQLite，仅允许 SELECT employees/orders）。"""

    name = "db_query"
    description = (
        "查询企业数据库（只读）。表 employees(员工: id,name,department,title,phone,email,hire_date) 与 "
        "orders(销售订单: order_id,customer,product,qty,amount,order_date,region,salesperson_id)。"
        "仅允许 SELECT，可使用聚合函数（SUM/COUNT/AVG等）。"
    )

    def __init__(self, db_path: Path | None = None):
        self.db_path = Path(db_path) if db_path else ensure_company_db()
        self._lock = threading.Lock()

    def args_hint(self) -> str:
        return '{"sql": "SELECT ... FROM employees WHERE department=..."}'

    def _validate(self, sql: str) -> str | None:
        s = sql.strip().rstrip(";").strip()
        if not s.lower().startswith("select") and not s.lower().startswith("with"):
            return "仅允许 SELECT 查询"
        if ";" in s:
            return "不允许多语句"
        if _FORBIDDEN.search(s):
            return "包含被禁止的 SQL 关键字"
        for t in _TABLE_RE.findall(s):
            if t.lower().split(".")[-1] not in ALLOWED_TABLES:
                return f"表 {t} 不在白名单（仅允许 {sorted(ALLOWED_TABLES)}）"
        return None

    def run(self, args: dict[str, Any]) -> dict[str, Any]:
        sql = (args.get("sql") or "").strip()
        if not sql:
            return {"output": "错误：缺少 sql 参数", "sources": [], "error": "missing sql"}
        err = self._validate(sql)
        if err:
            return {"output": f"SQL 校验失败：{err}", "sources": [], "error": err}
        try:
            with self._lock:
                conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, timeout=5)
                try:
                    cur = conn.execute(sql)
                    cols = [d[0] for d in cur.description] if cur.description else []
                    rows = cur.fetchmany(50)
                finally:
                    conn.close()
        except sqlite3.Error as e:
            return {"output": f"SQL 执行错误：{e}", "sources": [], "error": str(e)}
        if not rows:
            return {"output": "查询结果为空（0 行）", "sources": []}
        md = "| " + " | ".join(cols) + " |\n|" + "|".join(["---"] * len(cols)) + "|\n"
        md += "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in rows)
        if len(rows) == 50:
            md += "\n（最多展示 50 行）"
        return {"output": f"SQL: {sql}\n共 {len(rows)} 行：\n{md}", "sources": []}

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "description": self.description, "args_hint": self.args_hint()}
