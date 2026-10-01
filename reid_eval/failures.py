"""failures/: highest-similarity negatives and lowest-similarity near positives, with contact sheets.

These are mostly annotation errors (a vehicle split in two tracklets, an identity switch): they must be reviewed
by a person before the numbers are trusted. Nothing is removed automatically.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def collect(accs, kind, n=200):
    """kind: 'neg' (highest sims first) or 'pos' (lowest first). Returns a DataFrame of the n worst pairs over all videos."""
    rows = []
    for a in accs:
        f = a[f"fail_{kind}"]
        for k in range(len(f)):
            rows.append({"similarity": float(f[k, 0]), "video_id": a["meta"]["video_id"],
                         "tracklet_a": str(a[f"fail_{kind}_trk"][k, 0]), "tracklet_b": str(a[f"fail_{kind}_trk"][k, 1]),
                         "crop_a": str(a[f"fail_{kind}_uid"][k, 0]), "crop_b": str(a[f"fail_{kind}_uid"][k, 1]),
                         "frame_a": int(a[f"fail_{kind}_frame"][k, 0]), "frame_b": int(a[f"fail_{kind}_frame"][k, 1]),
                         "delta_position_m": float(a[f"fail_{kind}_dpos"][k])})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("similarity", ascending=(kind == "pos"), kind="stable").head(n).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def contact_sheets(df, data_root, out_prefix, per_sheet=20, cols=4, thumb=112):
    """Write PNG sheets of side-by-side crop pairs. Returns the written paths (empty if images are unavailable)."""
    out = []
    if df.empty:
        return out
    rows = int(np.ceil(per_sheet / cols))
    gap = 18  # background gap between two pairs, so a pair is visually one unit
    tw, th = 2 * thumb + 4 + gap, thumb + 18
    for s0 in range(0, len(df), per_sheet):
        sheet = np.full((rows * th, cols * tw, 3), 40, np.uint8)
        for k, (_, r) in enumerate(df.iloc[s0:s0 + per_sheet].iterrows()):
            ims = []
            for c in (r.crop_a, r.crop_b):
                img = cv2.imread(str(Path(data_root) / "videos" / c), cv2.IMREAD_COLOR)
                if img is None:
                    return out
                ims.append(cv2.resize(img, (thumb, thumb)))
            y, x = (k // cols) * th, (k % cols) * tw
            sheet[y + 16:y + 16 + thumb, x:x + thumb] = ims[0]
            sheet[y + 16:y + 16 + thumb, x + thumb + 4:x + 2 * thumb + 4] = ims[1]
            cv2.rectangle(sheet, (x - 1, y + 15), (x + 2 * thumb + 4, y + 16 + thumb), (200, 200, 200), 1)
            dp = "" if np.isnan(r.delta_position_m) else f" d={r.delta_position_m:.1f}m"
            cv2.putText(sheet, f"#{r['rank']} s={r.similarity:.3f}{dp}", (x + 2, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        p = Path(f"{out_prefix}_{s0 // per_sheet + 1:02d}.png")
        cv2.imwrite(str(p), sheet)
        out.append(p)
    return out
