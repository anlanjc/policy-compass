"""练习场夹具：把项目根加入 sys.path + 强制 mock 模式（与 tests/conftest.py 同思路）。

注意：e1 练习本体是自包含的（不 import app），此夹具为后续练习与"对照 app 源码"环节预留。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("POLICY_COMPASS_FORCE_MOCK", "1")
