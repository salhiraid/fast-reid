"""report.md + figures from the saved files of ONE results folder (global, a site, or a video). Nothing is recomputed."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .aggregate import HEATMAPS
from .bins import AXES
from .common import read_json
from .failures import contact_sheets
from .plots import plot_axis, plot_groups, plot_heatmap, plot_roc, pretty


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


def _link(df, col="report"):
    return df[col].map(lambda p: f"[report]({p})") if col in df else None


def build_report_from_dir(results_dir, data_root=None):
    out = Path(results_dir)
    s = read_json(out / "summary.json")
    names = s["far_names"]                                     # e.g. ['0.1pct', '1pct', '2pct', '5pct', '10pct'] (config order)
    order = sorted(names, key=lambda n: float(n.replace("pct", "")))
    h = s["headline"]
    fig = out / "figures"
    fig.mkdir(exist_ok=True)
    th = s["thresholds"]
    L = [f"# {s['title']}", "",
         f"- evaluation: **{s['kind']}**" + (" (all difficulty criteria: delta position, delta azimuth, occlusion, keypoints)"
                                              if s["kind"] == "full" else " (no pose / occlusion / keypoint criteria: every pair counts, one global result)"),
         f"- split `{s['split_version']}`, bins `{s['bins_version']}`, {s['n_videos']} test video(s)",
         "- thresholds from **validation only**: " + ", ".join(f"t({pretty(n)}) = {th[n]:.4f}" for n in order),
         f"- CIs: percentile bootstrap over videos ({s['bootstrap']['resamples']} resamples, {int(s['bootstrap']['level'] * 100)}%); "
         + ("" if s["has_ci"] else "**not computed here (fewer than 5 videos)**"), ""]

    # ---------------- headline
    L += ["## TAR at fixed FAR", "",
          "Main score = **object-balanced** (every object and object pair counts equally). The global thresholds come from validation; "
          "the FAR they give here is measured, not forced.", ""]
    rows = []
    for name in ("object_balanced", "pooled", "video_averaged"):
        d = h[name]
        if "tar_at_" + order[0] not in d:
            continue
        rows.append({"version": name, **{f"TAR @ {pretty(n)}": _ci(d[f"tar_at_{n}"]) for n in order}})
    for name in ("object_balanced", "pooled"):
        if name in h.get("bin_balanced", {}):
            rows.append({"version": f"bin_balanced ({name})", **{f"TAR @ {pretty(n)}": _ci(h["bin_balanced"][name][f"tar_at_{n}"]) for n in order}})
    L += [_md_table(pd.DataFrame(rows))]
    L += ["Measured FAR at those thresholds (target in the header):", "",
          _md_table(pd.DataFrame([{"version": name, **{f"{pretty(n)}": _ci(h[name][f"far_at_{n}"], 4) for n in order}}
                                  for name in ("object_balanced", "pooled")])),
          _md_table(pd.DataFrame([{"version": name, "AUC": _ci(h[name]["auc"], 4), "EER": _ci(h[name]["eer"], 4),
                                   "best TAR @ FAR 1% (threshold tuned here)": _ci(h[name]["best_tar_far"]), "d'": _f(h[name]["dprime"]["value"], 2)}
                                  for name in ("object_balanced", "pooled")])),
          f"Support: {h['support']['n_pos']:,} positive pairs, {h['support']['n_neg']:,} negative pairs, "
          f"{h['support']['n_objects']:,} objects, {h['support']['n_object_pairs']:,} distinct object pairs.", ""]
    if "validation" in s:
        v = s["validation"]
        L += ["Validation check (pooled): " + ", ".join(f"FAR {pretty(n)} = {v[f'pooled_far_at_{n}']:.5f}" for n in order)
              + f", AUC = {v['pooled_auc']:.4f}.", ""]

    roc = pd.read_csv(out / "roc.csv")
    plot_roc(roc, h, fig / "roc.png", names, title=s.get("short_title", ""))
    L += ["## ROC", "", "![roc](figures/roc.png)", ""]

    # ---------------- difficulty axes (full evaluation only)
    if s["kind"] == "full":
        L += ["## Difficulty axes", "",
              "TAR per bin at every FAR target (strict = dark, loose = light). Hollow grey markers: fewer than the minimum positive pairs / "
              "objects. AUC/EER/best-TAR come from 200-bin histograms (0.01 resolution); TAR/FAR at the thresholds are exact.", ""]
        for axis in AXES:
            df = pd.read_csv(out / f"bins_{axis}.csv")
            plot_axis(df, axis, fig / f"curve_{axis}.png", names)
            t = {"bin": df[f"bin_{axis}"], "pos pairs": df.n_pos.astype(int), "neg pairs": df.n_neg.astype(int),
                 "objects": df.n_objects.astype(int)}
            for n in order:
                t[f"TAR {pretty(n)}"] = df[f"balanced_tar_at_{n}"].map(_f)
            t.update({"AUC": df.balanced_auc.map(_f), "supported": np.where(df.supported, "yes", "**no (grey)**")})
            L += [f"### {axis.replace('_', ' ')}", "", f"![{axis}](figures/curve_{axis}.png)", "", _md_table(pd.DataFrame(t))]
        L += ["## Heatmaps", ""]
        for a, b in HEATMAPS:
            df = pd.read_csv(out / f"heatmap_{a}_x_{b}.csv")
            L += [f"### {a.replace('_', ' ')} x {b.replace('_', ' ')}", ""]
            for n in s["heat_names"]:
                png = f"heatmap_{a}_x_{b}__tar_{n}.png"
                plot_heatmap(df, a, b, fig / png, n)
                L += [f"![{png}](figures/{png})", ""]

    # ---------------- per site / per video (global report only)
    if (out / "per_site.csv").exists():
        ps = pd.read_csv(out / "per_site.csv")
        plot_groups(ps, "site", fig / "per_site_tar.png", names, title="TAR per site (object-balanced)")
        t = {"site": ps.site, "videos": ps.n_videos, "objects": ps.n_objects, "crops": ps.n_crops, "AUC": ps.auc.map(_f)}
        for n in order:
            t[f"TAR {pretty(n)}"] = ps[f"balanced_tar_at_{n}"].map(_f)
        t["confused obj. pairs"] = ps.confused_object_pairs
        if "report" in ps:
            t["figures"] = _link(ps)
        L += ["## Per site", "", "Same figures per site: follow the links. Sites with fewer than 5 videos have no CI.", "",
              "![per site](figures/per_site_tar.png)", "", _md_table(pd.DataFrame(t))]
    if (out / "per_video.csv").exists():
        pv = pd.read_csv(out / "per_video.csv")
        t = {"video": pv.video_id, "site": pv.site, "pov": pv.pov, "objects": pv.n_objects, "crops": pv.n_crops, "AUC": pv.auc.map(lambda x: _f(x, 4)),
             "EER": pv.eer.map(lambda x: _f(x, 4))}
        for n in order:
            t[f"TAR {pretty(n)}"] = pv[f"tar_at_{n}"].map(_f)
        t[f"FAR {pretty(order[0])}"] = pv[f"far_at_{order[0]}"].map(lambda x: _f(x, 5))
        t["confused obj. pairs"] = pv.confused_object_pairs
        if "report" in pv:
            t["figures"] = _link(pv)
        L += ["## Per video", "", "Same figures per video: follow the links (most bins of a single video are below minimum support).", "",
              _md_table(pd.DataFrame(t))]

    # ---------------- failures (global only)
    fdir = out / "failures"
    if fdir.is_dir():
        L += ["## Failures to review before trusting the numbers", "",
              "Mostly annotation errors (one vehicle split in two tracklets = false negative; identity switch = false positive). "
              "Nothing is removed automatically.", ""]
        for csv in sorted(fdir.glob("*.csv")):
            name = csv.stem
            L += [f"### {name.replace('_', ' ')}", "", f"CSV: `failures/{csv.name}`", ""]
            if data_root is not None and not list(fdir.glob(f"{name}_*.png")):
                contact_sheets(pd.read_csv(csv), data_root, fdir / name)
            for p in sorted(fdir.glob(f"{name}_*.png"))[:3]:
                L += [f"![{p.stem}](failures/{p.name})", ""]
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    return out / "report.md"
