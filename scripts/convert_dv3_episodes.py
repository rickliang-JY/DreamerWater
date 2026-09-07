"""CLI：上游 dv3 episode npz → UWM recorder schema（SPEC_M2 §2.4）。

用法：
    python scripts/convert_dv3_episodes.py <dv3_logdir 或 run_dir>

核心逻辑在 uwm/eval/dv3_episodes.py（schema 对照见该模块 docstring）。
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from uwm.eval.dv3_episodes import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
