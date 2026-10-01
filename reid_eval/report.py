"""report.md from the saved result files only (summary.json, bins_*.csv, heatmap_*.csv, per_video.csv, failures/)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .aggregate import HEATMAPS
from .bins import AXES
from .common import read_json
from .failures import contact_sheets
from .plots import plot_axis, plot_heatmap


def _f(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def _ci(e, d=3):
    if e is None:
        return "n/a"
    s = _f(e["value"], d)
    return s + (f" [{_f(e['ci_lo'], d)}, {_f(e['ci_hi'], d)}]" if "ci_lo" in e else "")


def _md_table(df):
    head = "| " + " | ".join(df.columns) + " |\n|" + "|".join("---" for _ in df.columns) + "|\n"
    return head + "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)) + "\n"


def build_report_from_dir(results_dir, data_root=None):
    out = Path(results_dir)
    s = read_json(out / "summary.json")
    th = read_json(out / "thresholds.json")
    fig = out / "figures"
    fig.mkdir(exist_ok=True)
    h = s["headline"]
    t2, t3 = th["names"][0], th["names"][1]
    L = [f"# Verification report: {s['model_name']} / {s['preproc_mode']}", "",
         f"- split `{s['split_version']}` ({s['n_videos']} test videos), bins `{s['bins_version']}`",
         f"- thresholds from **validation only**: t(1e-2) = {th['thresholds'][0]:.4f}, t(1e-3) = {th['thresholds'][1]:.4f} "
         f"({th['n_negative_pairs']:,} negative pairs, {th['n_distinct_object_pairs']:,} distinct object pairs, "
         f"{th['n_positive_pairs']:,} positive pairs)",
         f"- CIs: percentile bootstrap over videos, {s['bootstrap']['resamples']} resamples, {int(s['bootstrap']['level'] * 100)}% "
         "(pairs and crops are never resampled)", "",
         "## Headline", "",
         "Main score = **object-balanced** (every object and every object pair counts equally). "
         "FAR is measured on test at the validation thresholds; it is not forced to the target.", ""]
    rows = []
    for name in ("object_balanced", "pooled", "video_averaged"):
        d = h[name]
        rows.append({"version": name, "TAR@t(1e-2)": _ci(d["tar_t2"]), "TAR@t(1e-3)": _ci(d["tar_t3"]),
                     "FAR@t(1e-2) (target 0.01)": _ci(d["far_t2"], 4), "FAR@t(1e-3) (target 0.001)": _ci(d["far_t3"], 5),
                     "AUC": _ci(d["auc"], 4), "EER": _ci(d["eer"], 4)})
    bb = h.get("bin_balanced", {})
    for name in ("object_balanced", "pooled"):
        if name in bb:
            rows.append({"version": f"bin_balanced ({name})", "TAR@t(1e-2)": _ci(bb[name]["tar_t2"]), "TAR@t(1e-3)": _ci(bb[name]["tar_t3"]),
                         "FAR@t(1e-2) (target 0.01)": "", "FAR@t(1e-3) (target 0.001)": "", "AUC": "", "EER": ""})
    L += [_md_table(pd.DataFrame(rows)),
          f"Video-averaged uses {h['video_averaged']['n_videos_used']} videos with enough pairs. Bin-balanced = mean TAR over the "
          f"supported delta-position bins ({', '.join(bb.get('bins_used', []))}).",
          f"Support: {h['support']['n_pos']:,} positive pairs, {h['support']['n_neg']:,} negative pairs, "
          f"{h['support']['n_objects']:,} objects, {h['support']['n_object_pairs']:,} distinct object pairs.", ""]
    if "validation" in s:
        v = s["validation"]
        L += [f"Validation check (pooled): FAR@t(1e-2) = {v['pooled_far_t2']:.4f}, FAR@t(1e-3) = {v['pooled_far_t3']:.5f}, "
              f"AUC = {v['pooled_auc']:.4f}.", ""]

    L += ["## Difficulty axes", "",
          "Each figure: object-balanced TAR at both thresholds per bin. Hollow grey markers: fewer than the minimum positive pairs / objects. "
          "AUC/EER/best-TAR in the CSVs come from 200-bin histograms (about 0.01 similarity resolution); TAR/FAR at t are exact.", ""]
    for axis in AXES:
        df = pd.read_csv(out / f"bins_{axis}.csv")
        plot_axis(df, axis, fig / f"curve_{axis}.png")
        L += [f"### {axis.replace('_', ' ')}", "", f"![{axis}](figures/curve_{axis}.png)", ""]
        t = pd.DataFrame({"bin": df[f"bin_{axis}"], "pos pairs": df.n_pos.astype(int), "neg pairs": df.n_neg.astype(int),
                          "objects": df.n_objects.astype(int), "TAR@t(1e-3) bal.": df.balanced_tar_t3.map(_f),
                          "TAR@t(1e-3) pooled": df.pooled_tar_t3.map(_f), "AUC bal.": df.balanced_auc.map(_f),
                          "best TAR@FAR1e-2": df.balanced_best_tar_far.map(_f),
                          "supported": np.where(df.supported, "yes", "**no (grey)**")})
        L += [_md_table(t)]
    L += ["## Heatmaps", ""]
    for a, b in HEATMAPS:
        df = pd.read_csv(out / f"heatmap_{a}_x_{b}.csv")
        name = f"heatmap_{a}_x_{b}"
        plot_heatmap(df, a, b, out / f"{name}.png")
        L += [f"### {a.replace('_', ' ')} x {b.replace('_', ' ')}", "", f"![{name}]({name}.png)", ""]
    pv = pd.read_csv(out / "per_video.csv")
    L += ["## Per video", "", _md_table(pv.assign(**{c: pv[c].map(lambda x: _f(x, 4)) for c in ("auc", "eer", "tar_t2", "tar_t3", "far_t2", "far_t3", "balanced_tar_t3")}))]
    fdir = out / "failures"
    L += ["## Failures to review before trusting the numbers", "",
          "Mostly annotation errors (one vehicle split in two tracklets = false negative; identity switch = false positive). "
          "Nothing is removed automatically.", ""]
    for name, title in (("negatives_highest_similarity", "200 negative pairs with the highest similarity"),
                        ("positives_lowest_similarity_dpos_lt_1m", "200 positive pairs with the lowest similarity and delta position < 1 m")):
        L += [f"### {title}", "", f"CSV: `failures/{name}.csv`", ""]
        if data_root is not None and not list(fdir.glob(f"{name}_*.png")):
            contact_sheets(pd.read_csv(fdir / f"{name}.csv"), data_root, fdir / name)
        for p in sorted(fdir.glob(f"{name}_*.png"))[:3]:
            L += [f"![{p.stem}](failures/{p.name})", ""]
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    return out / "report.md"
