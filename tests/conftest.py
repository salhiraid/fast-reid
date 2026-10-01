import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE):  # repo root (reid_data, reid_eval, scripts) and this folder (helpers)
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
