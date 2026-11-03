"""启动脚本：python run.py"""
import sys
from pathlib import Path

# Windows 控制台中文输出兼容
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config import get_config  # noqa: E402

if __name__ == "__main__":
    import uvicorn

    cfg = get_config()
    uvicorn.run(
        "app.main:app",
        host=cfg.get("server.host", "127.0.0.1"),
        port=int(cfg.get("server.port", 8000)),
        reload=False,
        log_level="info",
    )
