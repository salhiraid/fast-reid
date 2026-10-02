"""report.md + figures from the saved files of ONE results folder (global, a site, or a video). Nothing is recomputed."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .aggregate import HEATMAPS
from .bins import AXES
from .common import read_json
from .failures import contact_sheets
from .plots import plot_accuracy, plot_axis, plot_groups, plot_heatmap, plot_roc, pretty


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
    proto = s.get("threshold_protocol")                           # per-video / per-site threshold evaluations
    fixed = s.get("fixed_names", [])
    ops = order + fixed                                           # every operating point: FAR thresholds, then fixed thresholds
    L = [f"# {s['title']}", "",
         *(["> **ORACLE evaluation: the thresholds were tuned on the same videos they are measured on. The numbers are optimistic; "
            "use them only as an upper bound for what per-video / per-site calibration could give.**", ""] if s.get("oracle") else []),
         f"- evaluation: **{s['kind']}**" + (" (all difficulty criteria: delta position, delta azimuth, occlusion, keypoints)"
                                              if s["kind"] == "full" else " (no pose / occlusion / keypoint criteria: every pair counts, one global result)"),
         f"- split `{s['split_version']}`, bins `{s['bins_version']}`, {s['n_videos']} test video(s)",
         *([f"- **threshold protocol**: {proto}",
            f"- {s['threshold_units_used']} video(s) evaluated, {len(s['threshold_units_skipped'])} skipped; median number of calibration "
            f"negative pairs per threshold: {s['n_calibration_negatives_median']:,.0f} (few negatives make a strict-FAR threshold noisy). "
            "The FAR below is MEASURED" + (" only in the held-out protocols; here it is forced to the target by construction." if s.get("oracle") else
                                           " on pairs that were not used to set the threshold."),
            "- per-unit thresholds: `thresholds_per_unit.csv`"] if proto else
           ["- thresholds from **validation only**: " + ", ".join(f"t({pretty(n)}) = {th[n]:.4f}" for n in order)
            + ("; fixed (not tuned): " + ", ".join(f"{pretty(n)}" for n in fixed) if fixed else "")]),
         (f"- CIs: percentile bootstrap over videos ({s['bootstrap']['resamples']} resamples, {int(s['bootstrap']['level'] * 100)}%)"
          if s["has_ci"] else "- CIs: **not computed here (fewer than 5 videos)**"), ""]

    # ---------------- headline
    L += ["## TAR at fixed FAR", "",
          "Main score = **object-balanced** (every object and object pair counts equally). "
          + ("Each video / site has its own threshold (see the protocol above)." if proto else
             "The global thresholds come from validation; the FAR they give here is measured, not forced."), ""]
    rows = []
    for name in ("object_balanced", "pooled", "video_averaged", "site_averaged"):
        d = h.get(name, {})
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
          "Versions: **object_balanced** and **pooled** (defined above); **video_averaged** = metric per video, then the mean over videos with enough "
          "pairs; **site_averaged** = metric per site (pooled over its videos), then the mean over sites (CI only with >= 5 sites); "
          "**bin_balanced** = mean TAR over the supported delta-position bins.", "",
          f"Support: {h['support']['n_pos']:,} positive pairs, {h['support']['n_neg']:,} negative pairs, "
          f"{h['support']['n_objects']:,} objects, {h['support']['n_object_pairs']:,} distinct object pairs.", ""]
    if "validation" in s:
        v = s["validation"]
        L += ["Validation check (pooled): " + ", ".join(f"FAR {pretty(n)} = {v[f'pooled_far_at_{n}']:.5f}" for n in order)
              + f", AUC = {v['pooled_auc']:.4f}.", ""]

    roc = pd.read_csv(out / "roc.csv")
    plot_roc(roc, h, fig / "roc.png", names, title=s.get("short_title", ""))
    L += ["## ROC", "", "![roc](figures/roc.png)", ""]

    # ---------------- accuracy at the fixed threshold and at every FAR threshold
    if f"acc_at_{ops[0]}" in h["pooled"]:
        npos, nneg = h["support"]["n_pos"], h["support"]["n_neg"]
        hdr = lambda n: pretty(n) if (proto and n not in fixed) else f"{pretty(n)} (t = {th[n]:.3f})"
        L += ["## Accuracy", "",
              "Accuracy = (positive pairs accepted + negative pairs rejected) / all pairs, at each threshold. "
              f"Here {nneg / max(npos + nneg, 1):.0%} of the pairs are negatives, so plain accuracy mostly measures the negatives; "
              "**balanced accuracy** = (TAR + (1 - FAR)) / 2 is the comparable number. Fixed-threshold counts are exact.", ""]
        for key, label in (("bacc", "Balanced accuracy"), ("acc", "Accuracy")):
            rows = [{"version": name, **{hdr(n): _ci(h[name][f"{key}_at_{n}"], 4) for n in ops}}
                    for name in ("object_balanced", "pooled", "video_averaged", "site_averaged") if f"{key}_at_{ops[0]}" in h.get(name, {})]
            L += [f"**{label}**", "", _md_table(pd.DataFrame(rows))]
        plot_accuracy(roc, {n: th[n] for n in (fixed if proto else ops)}, fig / "accuracy_vs_threshold.png", title=s.get("short_title", ""))
        L += ["![accuracy vs threshold](figures/accuracy_vs_threshold.png)", ""]

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
            for n in fixed:
                t[f"bal.acc {pretty(n)}"] = df[f"balanced_bacc_at_{n}"].map(_f)
                t[f"acc {pretty(n)}"] = df[f"pooled_acc_at_{n}"].map(_f)
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

    if s.get("site_matches"):
        L += ["## Site-level matching", "", f"Objects of the site against the gallery of all the site's videos (sampled queries, object x object "
              f"similarity matrix, cross-video candidates; matches between videos have no ground truth): [{s['site_matches']}]({s['site_matches']})", ""]
    if s.get("matches"):
        L += ["## Object matches", "", f"One image per object with its top-10 positive and negative matches: [{s['matches']}]({s['matches']})", ""]
    elif (out / "matches").is_dir():
        L += ["## Object matches", "", "One image per object (query = medoid crop) with its top-10 positive and negative matches, per video: "
              "`matches/<video>/index.md` (images + table, most confusable objects first) and `matches/<video>/index.csv`.", ""]

    if proto and s.get("threshold_units_skipped") and (out / "per_video.csv").exists():
        L += ["## Videos that could not be evaluated with this protocol", "",
              _md_table(pd.DataFrame([{"video": u["unit"], "site": u["site"], "reason": u["status"]} for u in s["threshold_units_skipped"]])), ""]

    # ---------------- spread across videos and sites
    def spread(csv, unit):
        st = pd.read_csv(out / csv, index_col="metric")
        pick = (["auc", "eer"] + [f"tar_at_{n}" for n in order] + [f"far_at_{n}" for n in order]
                + [f"bacc_at_{n}" for n in ops] + [f"acc_at_{n}" for n in ops])
        rows = [{"metric": m.replace("_at_", " @ "), f"{unit}s": int(st.loc[m, "count"]),
                 "mean +- std": f"{st.loc[m, 'mean']:.4f} +- {0 if np.isnan(st.loc[m, 'std']) else st.loc[m, 'std']:.4f}",
                 "median": _f(st.loc[m, "median"], 4), "min - max": f"{st.loc[m, 'min']:.4f} - {st.loc[m, 'max']:.4f}"}
                for m in pick if m in st.index]
        return _md_table(pd.DataFrame(rows))
    if (out / "per_video_stats.csv").exists() and (out / "per_video.csv").exists():
        L += ["## Spread across videos and sites", "",
              "Every number below is computed inside one video (or one site) first, then summarised over the videos (sites): "
              "how much the result varies from one camera to another.", "", "**Over videos**", "", spread("per_video_stats.csv", "video")]
        if (out / "per_site_stats.csv").exists():
            L += ["**Over sites**", "", spread("per_site_stats.csv", "site")]

    # ---------------- per site / per video (global report only)
    if (out / "per_site.csv").exists():
        ps = pd.read_csv(out / "per_site.csv")
        plot_groups(ps, "site", fig / "per_site_tar.png", names, title="TAR per site (object-balanced)")
        t = {"site": ps.site, "videos": ps.n_videos, "objects": ps.n_objects, "crops": ps.n_crops, "AUC": ps.auc.map(_f)}
        for n in order:
            t[f"TAR {pretty(n)}"] = ps[f"balanced_tar_at_{n}"].map(_f)
        for n in fixed:
            t[f"bal.acc {pretty(n)}"] = ps[f"balanced_bacc_at_{n}"].map(_f)
            t[f"acc {pretty(n)}"] = ps[f"acc_at_{n}"].map(_f)
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
        for n in fixed:
            t[f"bal.acc {pretty(n)}"] = pv[f"balanced_bacc_at_{n}"].map(_f)
            t[f"acc {pretty(n)}"] = pv[f"acc_at_{n}"].map(_f)
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


VARIANTS = (("", "global threshold from validation (all difficulty criteria)"), ("__plain", "global threshold from validation"),
            ("__thr-video", "threshold per video, held out (2 folds of objects)"), ("__thr-site", "threshold per site, held out (leave-one-video-out)"),
            ("__thr-video-oracle", "threshold per video, ORACLE (tuned on the evaluated video)"),
            ("__thr-site-oracle", "threshold per site, ORACLE (tuned on the evaluated site)"))


def write_variant_comparison(model_dir, split_version):
    """<model_dir>/<split>__threshold_comparison.md/.csv: object-balanced TAR at every FAR for each threshold protocol that was run."""
    model_dir = Path(model_dir)
    rows, seen_global = [], False
    for suffix, label in VARIANTS:
        d = model_dir / f"{split_version}{suffix}"
        if not (d / "summary.json").exists():
            continue
        if suffix in ("", "__plain"):          # same global thresholds and pairs: identical numbers, show once
            if seen_global:
                continue
            seen_global = True
        s = read_json(d / "summary.json")
        h = s["headline"]
        names = sorted(s["far_names"], key=lambda n: float(n.replace("pct", "")))
        row = {"variant": label, "folder": d.name, "videos evaluated": s["n_videos"],
               "pos pairs": h["support"]["n_pos"], "neg pairs": h["support"]["n_neg"]}
        for n in names:
            row[f"TAR @ {pretty(n)}"] = h["object_balanced"][f"tar_at_{n}"]["value"]
        for n in names:
            row[f"measured FAR @ {pretty(n)}"] = h["pooled"][f"far_at_{n}"]["value"]
        for n in s.get("fixed_names", []):
            row[f"balanced acc @ {pretty(n)}"] = h["object_balanced"][f"bacc_at_{n}"]["value"]
        rows.append(row)
    if len(rows) < 2:
        return None
    df = pd.DataFrame(rows)
    df.to_csv(model_dir / f"{split_version}__threshold_comparison.csv", index=False)
    show = df.copy()
    for c in show.columns[5:]:
        show[c] = show[c].map(lambda x: f"{x:.4f}")
    show["pos pairs"], show["neg pairs"] = show["pos pairs"].map("{:,}".format), show["neg pairs"].map("{:,}".format)
    text = ["# Threshold protocols side by side", "",
            "Object-balanced TAR at each FAR target, and the FAR actually measured (pooled). **Read with care**: the rows do not evaluate exactly "
            "the same pairs. The held-out per-video protocol evaluates only pairs inside each half of a video; the per-site protocol skips "
            "sites with a single test video; ORACLE rows tune the threshold on the data they are measured on (FAR forced to the target, "
            "optimistic). Differences between rows therefore mix the effect of the threshold with a change of the evaluated pairs.", "",
            _md_table(show.drop(columns=["folder"]))]
    (model_dir / f"{split_version}__threshold_comparison.md").write_text("\n".join(text), encoding="utf-8")
    return model_dir / f"{split_version}__threshold_comparison.md"
