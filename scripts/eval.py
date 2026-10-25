"""评测入口：python scripts/eval.py [--retrieval]"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.eval.runner import main

if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
