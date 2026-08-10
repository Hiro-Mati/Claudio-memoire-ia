from __future__ import annotations

import sys
from pathlib import Path

ARK4_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ARK4_DIR.parents[1]
for path in (REPO_ROOT, ARK4_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
