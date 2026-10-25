"""启动 API 服务：python scripts/serve.py [--reload]"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn

from app.config import settings


def main() -> None:
    reload = "--reload" in sys.argv
    uvicorn.run(
        "app.api.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=reload,
    )


if __name__ == "__main__":
    main()
