"""One report comparing EVERY evaluation method of a model: tables (site-averaged first) and curve figures (no heatmaps).

Methods (a folder `<split>[__variant]` of the model's results folder; missing ones are skipped):
  global              one threshold per FAR target set on the validation videos, matching WITHIN each video      (v1 / v1__plain)
  thr-video           threshold per video, held out (2 folds of objects), matching within the video              (v1__thr-video)
  thr-site            threshold per site, held out (leave-one-video-out), matching within the video              (v1__thr-site)
  site-gallery        matching against the WHOLE SITE (other videos too; cross-video pairs assumed negative),
                      threshold per site, held out (2 folds of the site's objects)                               (v1__site-gallery)
  *-oracle            the same thresholds tuned on the evaluated data: optimistic upper bounds                  (v1__thr-*-oracle ...)
Everything is read from the saved summaries / per-site tables / accumulators of those folders.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import metrics as M
from .common import read_json
from .plots import GREY, INK, INK2, _style, pretty
from .report import _f, _md_table

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
# key, folder suffixes (first existing wins), label, short label, matching, colour, oracle
METHODS = (
    ("global", ("__plain", ""), "Global threshold (validation)", "Global threshold", "within each video", BLUE, False),
    ("thr-video", ("__thr-video",), "Threshold per video (held out)", "Threshold per video", "within each video", ORANGE, False),
    ("thr-site", ("__thr-site",), "Threshold per site (held out, leave-one-video-out)", "Threshold per site", "within each video", AQUA, False),
    ("site-gallery", ("__site-gallery",), "Site matching: gallery = whole site, threshold per site (held out)", "Site matching", "whole-site gallery (cross-video pairs = negatives)", VIOLET, False),
    ("thr-video-oracle", ("__thr-video-oracle",), "Threshold per video ORACLE (optimistic)", "Per video, oracle", "within each video", ORANGE, True),
    ("thr-site-oracle", ("__thr-site-oracle",), "Threshold per site ORACLE (optimistic)", "Per site, oracle", "within each video", AQUA, True),
    ("site-gallery-oracle", ("__site-gallery-oracle",), "Site matching ORACLE (optimistic)", "Site matching, oracle", "whole-site gallery (cross-video pairs = negatives)", VIOLET, True),
)


class Method:
    def __init__(self, key, folder, label, short, matching, color, oracle):
        self.key, self.dir, self.label, self.short, self.matching, self.color, self.oracle = key, Path(folder), label, short, matching, color, oracle
        self.summary = read_json(self.dir / "summary.json")
        self.per_site = pd.read_csv(self.dir / "per_site.csv")
        self.far_names = self.summary["far_names"]
        self.fixed = self.summary.get("fixed_names", [])
        self.ops = self.far_names + self.fixed
        self.far_order = sorted(self.far_names, key=lambda n: float(n.replace("pct", "")))
        self._curves = None

    @property
    def style(self):
        return dict(color=self.color, ls="--" if self.oracle else "-", marker="o", mfc="white" if self.oracle else self.color)

    # ---- site-averaged value of a per-site metric (the headline `site_averaged` when it exists, else the plain mean over sites)
    def site_mean(self, col, metric):
        sa = self.summary["headline"].get("site_averaged") or {}
        if metric in sa:
            return sa[metric]["value"]
        return float(self.per_site[col].mean())

    def site_std(self, col):
        return float(self.per_site[col].std(ddof=1)) if len(self.per_site) > 1 else float("nan")

    def pooled(self, version, metric):
        return self.summary["headline"][version][metric]["value"]

    def curves(self):
        """Per-site ROC and accuracy-vs-threshold curves from the accumulators' 2,000-bin histograms."""
        if self._curves is None:
            hist = {}
            for p in sorted((self.dir / "acc" / "test").glob("*.npz")):
                with np.load(p, allow_pickle=False) as z:
                    site = json.loads(str(z["meta"])).get("site") or "unknown"
                    hist[site] = hist.get(site, 0) + z["fine"].astype(np.float64)
            out = {}
            for site, h in hist.items():
                P, N = h[0].sum(), h[1].sum()
                far, tar = M.roc(h[0], h[1])
                tp = np.concatenate([np.cumsum(h[0][::-1])[::-1], [0]])
                fp = np.concatenate([np.cumsum(h[1][::-1])[::-1], [0]])
                tn = N - fp
                out[site] = {"far": far, "tar": tar, "acc": (tp + tn) / (P + N), "bacc": (tp / P + tn / N) / 2}
            self._curves = out
        return self._curves


def discover(model_dir, split_version):
    model_dir = Path(model_dir)
    found = []
    for key, suffixes, label, short, matching, color, oracle in METHODS:
        for suf in suffixes:
            d = model_dir / f"{split_version}{suf}"
            if (d / "summary.json").exists() and (d / "per_site.csv").exists():
                found.append(Method(key, d, label, short, matching, color, oracle))
                break
    return found


# ------------------------------------------------------------------ tables
def _cell(m, col, metric, d=3, std=True):
    v, s = m.site_mean(col, metric), m.site_std(col)
    return f"{v:.{d}f}" + (f" ± {s:.{d}f}" if std and not np.isnan(s) else "")


def tables(methods):
    """Returns (list of (title, DataFrame of strings) for markdown, one long numeric DataFrame for CSV)."""
    m0 = methods[0]
    order = m0.far_order
    fixed = m0.fixed
    ops = order + fixed
    t_main = []
    for m in methods:
        row = {"method": m.short, "matching": m.matching, "sites": int(m.summary["headline"].get("site_averaged", {}).get("n_sites_used", len(m.per_site)))}
        for n in order:
            row[f"TAR @ {pretty(n)}"] = _cell(m, f"tar_at_{n}", f"tar_at_{n}")
        row["AUC"], row["EER"] = _cell(m, "auc", "auc", 4), _cell(m, "eer", "eer", 4)
        for n in fixed:
            row[f"bal. acc @ {pretty(n)}"] = _cell(m, f"bacc_at_{n}", f"bacc_at_{n}")
            row[f"acc @ {pretty(n)}"] = _cell(m, f"acc_at_{n}", f"acc_at_{n}")
        t_main.append(row)
    out = [("Site-averaged results: TAR at each FAR, AUC, EER, accuracy", pd.DataFrame(t_main))]
    for key, title in (("bacc", "Balanced accuracy at every operating point (site-averaged)"), ("acc", "Accuracy at every operating point (site-averaged)")):
        rows = [{"method": m.short, **{pretty(n): _cell(m, f"{key}_at_{n}", f"{key}_at_{n}") for n in ops}} for m in methods]
        out.append((title, pd.DataFrame(rows)))
    rows = [{"method": m.short, **{pretty(n): _cell(m, f"far_at_{n}", f"far_at_{n}", 4) for n in order}} for m in methods]
    out.append(("Measured FAR at each target (site-averaged; the target is in the header)", pd.DataFrame(rows)))
    rows = []
    for m in methods:
        for ver in ("object_balanced", "pooled"):
            row = {"method": m.short, "version": ver}
            row.update({f"TAR @ {pretty(n)}": _f(m.pooled(ver, f"tar_at_{n}")) for n in order})
            row["AUC"], row["EER"] = _f(m.pooled(ver, "auc"), 4), _f(m.pooled(ver, "eer"), 4)
            row.update({f"bal. acc @ {pretty(n)}": _f(m.pooled(ver, f"bacc_at_{n}")) for n in fixed})
            rows.append(row)
    out.append(("Pooled over all pairs (not averaged over sites): object-balanced and pooled", pd.DataFrame(rows)))
    # ---- numeric long table for CSV
    longrows = []
    for m in methods:
        for site, r in m.per_site.set_index("site").iterrows():
            longrows.append({"method": m.key, "label": m.label, "site": site, **{c: r[c] for c in m.per_site.columns
                                                                             if c not in ("site", "report") and not c.endswith(("_lo", "_hi"))}})
    return out, pd.DataFrame(longrows)


def summary_csv(methods):
    rows = []
    for m in methods:
        row = {"method": m.key, "label": m.label, "matching": m.matching, "oracle": m.oracle, "n_sites": len(m.per_site)}
        for n in m.ops:
            for b in ("tar", "far", "acc", "bacc"):
                row[f"{b}_at_{n}"] = m.site_mean(f"{b}_at_{n}", f"{b}_at_{n}")
                row[f"{b}_at_{n}_std_over_sites"] = m.site_std(f"{b}_at_{n}")
        for k in ("auc", "eer"):
            row[k], row[f"{k}_std_over_sites"] = m.site_mean(k, k), m.site_std(k)
        rows.append(row)
    return pd.DataFrame(rows)


def per_site_tables(methods):
    out = []
    order = methods[0].far_order
    fixed = methods[0].fixed
    picks = [(f"tar_at_{n}", f"TAR @ {pretty(n)}") for n in order[:2]] + [(f"bacc_at_{n}", f"Balanced accuracy @ {pretty(n)}") for n in fixed] \
        + [("auc", "AUC"), ("eer", "EER")]
    sites = sorted({s for m in methods for s in m.per_site.site})
    for col, title in picks:
        rows = []
        for s in sites:
            row = {"site": s}
            for m in methods:
                r = m.per_site[m.per_site.site == s]
                row[m.short] = _f(float(r[col].iloc[0]), 4 if col in ("auc", "eer") else 3) if len(r) else "n/a"
            rows.append(row)
        out.append((f"Per site: {title}", pd.DataFrame(rows)))
    return out


# ------------------------------------------------------------------ figures (curves only)
def _legend(ax, methods, loc="lower right"):
    ax.legend(frameon=False, fontsize=7, loc=loc)


def fig_roc(methods, path):
    """Mean over sites of TAR at a log grid of FAR + the five FAR targets as dots (the spread over sites is in the TAR-vs-target figure)."""
    grid = np.logspace(-4, 0, 300)             # below 1e-4 a site has too few negatives for a stable curve
    fig, ax = plt.subplots(figsize=(6.6, 4.6), dpi=130)
    _style(ax)
    ax.grid(axis="x", color="#e6e5e0", lw=0.6)
    for m in methods:
        tars = []
        for site, c in m.curves().items():
            tars.append(np.interp(grid, c["far"][::-1], c["tar"][::-1]))
        tars = np.array(tars)
        ax.plot(grid, tars.mean(0), ls="--" if m.oracle else "-", color=m.color, lw=1.5, label=m.short)
        far = [m.site_mean(f"far_at_{n}", f"far_at_{n}") for n in m.far_order]
        tar = [m.site_mean(f"tar_at_{n}", f"tar_at_{n}") for n in m.far_order]
        ax.plot(far, tar, "o", color=m.color, mfc="white" if m.oracle else m.color, ms=4.5, zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("FAR", fontsize=8, color=INK2)
    ax.set_ylabel("TAR (mean over sites)", fontsize=8, color=INK2)
    ax.set_title("ROC averaged over sites; dots = the five FAR targets at their measured (FAR, TAR)", fontsize=8.5, loc="left", color=INK)
    _legend(ax, methods)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_tar_vs_target(methods, path):
    order = methods[0].far_order
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=130)
    _style(ax)
    for i, m in enumerate(methods):
        dx = (i - (len(methods) - 1) / 2) * 0.05
        mu = np.array([m.site_mean(f"tar_at_{n}", f"tar_at_{n}") for n in order])
        sd = np.array([m.site_std(f"tar_at_{n}") for n in order])
        ax.errorbar(x + dx, mu, yerr=np.nan_to_num(sd), color=m.color, ls="--" if m.oracle else "-", marker="o", ms=4.5, lw=1.5, capsize=2,
                    mfc="white" if m.oracle else m.color, label=m.short)
    ax.set_xticks(x)
    ax.set_xticklabels([pretty(n).replace("FAR ", "") for n in order], fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("FAR target", fontsize=8, color=INK2)
    ax.set_ylabel("TAR (mean over sites, bars = std over sites)", fontsize=8, color=INK2)
    ax.set_title("TAR at each FAR target", fontsize=8.5, loc="left", color=INK)
    _legend(ax, methods, "lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_calibration(methods, path):
    order = methods[0].far_order
    tgt = np.array([float(n.replace("pct", "")) / 100 for n in order])
    fig, ax = plt.subplots(figsize=(5.4, 4.4), dpi=130)
    _style(ax)
    ax.grid(axis="x", color="#e6e5e0", lw=0.6)
    ax.plot(tgt, tgt, color=GREY, lw=1, ls=":", label="measured = target")
    for m in methods:
        mu = np.array([m.site_mean(f"far_at_{n}", f"far_at_{n}") for n in order])
        ax.plot(tgt, mu, ls="--" if m.oracle else "-", marker="o", ms=4.5, lw=1.5, color=m.color, mfc="white" if m.oracle else m.color, label=m.short)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("FAR target", fontsize=8, color=INK2)
    ax.set_ylabel("FAR actually measured (mean over sites)", fontsize=8, color=INK2)
    ax.set_title("How close each threshold protocol gets to its FAR target", fontsize=8.5, loc="left", color=INK)
    _legend(ax, methods, "upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_accuracy_vs_threshold(methods, path):
    t = np.linspace(-1, 1, 2001)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), dpi=130, sharex=True)
    for ax, key, title in ((axes[0], "acc", "accuracy (dominated by negatives)"), (axes[1], "bacc", "balanced accuracy")):
        _style(ax)
        ax.grid(axis="x", color="#e6e5e0", lw=0.6)
        for m in methods:
            mu = np.mean([c[key] for c in m.curves().values()], axis=0)
            ax.plot(t, mu, ls="--" if m.oracle else "-", color=m.color, lw=1.5, label=m.short)
        for n in methods[0].fixed:
            v = methods[0].summary["thresholds"][n]
            ax.axvline(v, color=INK, lw=1)
            ax.text(v, 0.02, f" {pretty(n).replace('threshold ', 'th ')}", transform=ax.get_xaxis_transform(), fontsize=7, color=INK, rotation=90, va="bottom", ha="left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0.4, 1.0)
        ax.set_xlabel("cosine similarity threshold", fontsize=8, color=INK2)
        ax.set_title(title + " (mean over sites)", fontsize=8.5, loc="left", color=INK)
    axes[0].set_ylabel("over all pairs", fontsize=8, color=INK2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def fig_per_site(methods, out_prefix, per_page=12, cols=4):
    """Small multiples, one panel per site: TAR vs FAR target, one line per method. Returns the written paths."""
    order = methods[0].far_order
    x = np.arange(len(order))
    sites = sorted({s for m in methods for s in m.per_site.site})
    paths = []
    for p0 in range(0, len(sites), per_page):
        chunk = sites[p0:p0 + per_page]
        rows = int(np.ceil(len(chunk) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.5 * rows + 1.0), dpi=130, sharey=True, squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, site in zip(axes.ravel(), chunk):
            ax.axis("on")
            _style(ax)
            for m in methods:
                r = m.per_site[m.per_site.site == site]
                if len(r):
                    ax.plot(x, [float(r[f"tar_at_{n}"].iloc[0]) for n in order], ls="--" if m.oracle else "-", marker="o", ms=3, lw=1.2, color=m.color,
                            mfc="white" if m.oracle else m.color, label=m.short)
            ax.set_title(site, fontsize=8, loc="left", color=INK)
            ax.set_xticks(x)
            ax.set_xticklabels([pretty(n).replace("FAR ", "") for n in order], fontsize=6.5)
            ax.set_ylim(0, 1.02)
        handles, labels = axes.ravel()[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), frameon=False, fontsize=7)
        fig.suptitle("TAR at each FAR target, per site", fontsize=9, x=0.01, ha="left", color=INK)
        fig.tight_layout(rect=(0, 0.13, 1, 0.97))
        p = Path(f"{out_prefix}_{p0 // per_page + 1:02d}.png")
        fig.savefig(p)
        plt.close(fig)
        paths.append(p)
    return paths


def fig_auc_eer(methods, path):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), dpi=130)
    for ax, col, title in ((axes[0], "auc", "AUC"), (axes[1], "eer", "EER")):
        _style(ax)
        ax.grid(axis="y", color="#e6e5e0", lw=0.6)
        for i, m in enumerate(methods):
            v = m.per_site[col].to_numpy(float)
            jit = (np.random.RandomState(i).rand(len(v)) - 0.5) * 0.25
            ax.scatter(np.full(len(v), i) + jit, v, s=14, color=m.color, alpha=0.45, edgecolors="none")
            ax.errorbar([i], [m.site_mean(col, col)], yerr=[np.nan_to_num(m.site_std(col))], fmt="D", color=INK2, ms=5, capsize=3, lw=0.8, zorder=4)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m.short for m in methods], rotation=25, ha="right", fontsize=7)
        ax.set_title(f"{title}: one dot per site, diamond = mean +- std over sites", fontsize=8.5, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------ the report
def build_all_methods(model_dir, split_version="v1"):
    """Write <model_dir>/<split>__all_methods/{report.md, all_methods_summary.csv, all_methods_per_site.csv, figures/}."""
    methods = discover(model_dir, split_version)
    if not methods:
        return None
    out = Path(model_dir) / f"{split_version}__all_methods"
    fig = out / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    tabs, long = tables(methods)
    long.to_csv(out / "all_methods_per_site.csv", index=False)
    summary_csv(methods).to_csv(out / "all_methods_summary.csv", index=False)
    fig_roc(methods, fig / "roc_mean_over_sites.png")
    fig_tar_vs_target(methods, fig / "tar_vs_far_target.png")
    fig_calibration(methods, fig / "measured_far_vs_target.png")
    fig_accuracy_vs_threshold(methods, fig / "accuracy_vs_threshold.png")
    fig_auc_eer(methods, fig / "auc_eer_by_method.png")
    site_figs = fig_per_site(methods, fig / "per_site_tar")

    s0 = methods[0].summary
    L = [f"# All evaluation methods side by side: {s0.get('model_name')} / {s0.get('preproc_mode')}", "",
         f"Split `{s0.get('split_version')}`. Averages are over **sites** (each site counts once; ± = std over sites); every site's number is first "
         "computed on all its pairs. Curves only (no heatmaps). Dashed lines / hollow markers = ORACLE methods (thresholds tuned on the data they are "
         "measured on: optimistic upper bounds).", "",
         "## Methods", "",
         _md_table(pd.DataFrame([{"method": m.short, "threshold": ("per site" if "site" in m.key else "per video" if "video" in m.key else "one global, from validation")
                                  + (" (ORACLE)" if m.oracle else " (held out)" if m.key != "global" else ""),
                                  "matching (pairs)": m.matching, "folder": m.dir.name} for m in methods])),
         "**Read the comparison with care.** The methods do not evaluate exactly the same pairs: the held-out per-video protocol only evaluates pairs "
         "inside each half of a video; *site matching* adds every cross-video pair of the site as a NEGATIVE (the dataset has no identity link between "
         "videos: if a vehicle reappears in two videos, FAR is overstated); the per-video thresholds are noisy because one video has few negatives. "
         "So differences mix the effect of the threshold with a change of pairs and of difficulty.", ""]
    for title, df in tabs:
        L += [f"## {title}", ""]
        if title.startswith("Accuracy at every"):
            L += ["Plain accuracy depends on the share of positive pairs, which differs between the methods (the whole-site gallery adds many "
                  "cross-video negatives; the per-video protocol removes pairs): **compare balanced accuracy across methods**, use plain accuracy only within a method.", ""]
        L += [_md_table(df)]
    L += ["## Curves", "",
          "### ROC averaged over sites", "", "![roc](figures/roc_mean_over_sites.png)", "",
          "### TAR at each FAR target", "", "![tar](figures/tar_vs_far_target.png)", "",
          "### Measured FAR vs target (does the threshold protocol hit its FAR?)", "", "![calibration](figures/measured_far_vs_target.png)", "",
          "### Accuracy and balanced accuracy at every threshold", "", "![accuracy](figures/accuracy_vs_threshold.png)", "",
          "### AUC and EER by method", "", "![auc eer](figures/auc_eer_by_method.png)", "",
          "### Per site: TAR at each FAR target", ""] + [f"![per site {p.stem}](figures/{p.name})\n" for p in site_figs]
    for title, df in per_site_tables(methods):
        L += [f"## {title}", "", _md_table(df)]
    L += ["## Files", "", "- `all_methods_summary.csv`: site-averaged value and std over sites of every metric, one row per method",
          "- `all_methods_per_site.csv`: every metric of every site for every method (long format)",
          "- each method's own folder (`" + "`, `".join(m.dir.name for m in methods) + "`) has its full report"]
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    return out / "report.md"
