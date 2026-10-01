"""Paired comparison of two evaluated models on the same split (spec section 8, last paragraph / Task 8).

The SAME bootstrap resample of videos is applied to both models and the CI is taken on the per-resample
difference (B - A), so video difficulty cancels out. Each model keeps its own validation thresholds.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from .aggregate import AXES, Aggregation, ci, named, reduce_pack, resample_pack, stack_packs
from .bins import Bins
from .common import read_json

METRICS = ("tar_t2", "tar_t3", "far_t2", "far_t3", "auc", "eer")


def _load(results_dir, bins_path):
    d = Path(results_dir)
    th = read_json(d / "thresholds.json")
    bins = Bins.load(bins_path)
    if th["bins_sha256"] != bins.sha256:
        raise ValueError(f"{d} was evaluated with a different bins file than {bins_path}")
    paths = sorted((d / "acc" / "test").glob("*.npz"))
    return th, bins, Aggregation(paths, bins, th)


def compare(dir_a, dir_b, bins_path="configs/bins_v1.yaml"):
    th_a, bins, a = _load(dir_a, bins_path)
    th_b, _, b = _load(dir_b, bins_path)
    if th_a["split_sha256"] != th_b["split_sha256"]:
        raise ValueError("the two models were evaluated on different split files")
    ids_a = [x["meta"]["video_id"] for x in a.accs]
    if ids_a != [x["meta"]["video_id"] for x in b.accs]:
        raise ValueError("the two models do not cover the same test videos")
    ft = bins.far_targets[0]
    counts = a.counts  # identical for both (same V and seed)
    rows = []
    for keep in ((),) + tuple((k,) for k in range(4)):
        labels = [""] if not keep else bins.labels(AXES[keep[0]])
        sa = stack_packs([reduce_pack(c, bins, keep) for c in a.cells_per_video])
        sb = stack_packs([reduce_pack(c, bins, keep) for c in b.cells_per_video])
        ra, rb = resample_pack(sa, counts), resample_pack(sb, counts)
        for version, bal in (("object_balanced", True), ("pooled", False)):
            ma = named(M.compute(reduce_pack(a.pooled_cells, bins, keep), bal, ft), bins)
            mb = named(M.compute(reduce_pack(b.pooled_cells, bins, keep), bal, ft), bins)
            ba, bb = named(M.compute(ra, bal, ft), bins), named(M.compute(rb, bal, ft), bins)
            for k in METRICS:
                va, vb = np.atleast_1d(ma[k]), np.atleast_1d(mb[k])
                lo, hi = ci(bb[k] - ba[k], bins.boot_level)
                lo, hi = np.atleast_1d(lo), np.atleast_1d(hi)
                for i, lab in enumerate(labels):
                    rows.append({"axis": AXES[keep[0]] if keep else "all", "bin": lab or "all", "version": version, "metric": k,
                                 "model_a": va[i], "model_b": vb[i], "diff_b_minus_a": vb[i] - va[i], "ci_lo": lo[i], "ci_hi": hi[i],
                                 "ci_excludes_0": bool(lo[i] > 0 or hi[i] < 0), "n_pos_a": float(np.atleast_1d(ma["n_pos"])[i])})
    df = pd.DataFrame(rows)
    meta = {"model_a": f"{th_a['model_name']}__{th_a['preproc_mode']}", "model_b": f"{th_b['model_name']}__{th_b['preproc_mode']}",
            "split_version": th_a["split_version"], "n_videos": a.V, "resamples": bins.boot_resamples, "level": bins.boot_level}
    return df, meta


def to_markdown(df, meta):
    g = df[(df.axis == "all") & (df.version == "object_balanced")]
    L = [f"# Model comparison: {meta['model_a']} (A) vs {meta['model_b']} (B)", "",
         f"Split `{meta['split_version']}`, {meta['n_videos']} test videos, paired bootstrap over videos "
         f"({meta['resamples']} resamples, {int(meta['level'] * 100)}% CI on B - A). Each model uses its own validation thresholds, "
         "so FAR is measured, not matched.", "", "## Headline (object-balanced)", "",
         "| metric | A | B | B - A [CI] | CI excludes 0 |", "|---|---|---|---|---|"]
    for _, r in g.iterrows():
        L.append(f"| {r.metric} | {r.model_a:.4f} | {r.model_b:.4f} | {r.diff_b_minus_a:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] | "
                 f"{'yes' if r.ci_excludes_0 else 'no'} |")
    L += ["", "## TAR @ t(1e-3) difference per difficulty bin (object-balanced)", "",
          "| axis | bin | A | B | B - A [CI] | pos pairs |", "|---|---|---|---|---|---|"]
    t = df[(df.axis != "all") & (df.version == "object_balanced") & (df.metric == "tar_t3") & (df.n_pos_a > 0)]
    for _, r in t.iterrows():
        L.append(f"| {r.axis} | {r.bin} | {r.model_a:.3f} | {r.model_b:.3f} | {r.diff_b_minus_a:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] | {int(r.n_pos_a):,} |")
    return "\n".join(L) + "\n"
