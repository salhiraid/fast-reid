"""Compare several MODELS (template folders evaluated with the same split and bins) in tables and curves.

For every evaluation method that was run for at least two models (global threshold, per video, per site, site matching, oracles) it writes
a table with one row per model: TAR at each FAR, AUC, EER, accuracy / balanced accuracy (averaged over sites, ± std over sites), the same
pooled over all pairs, the best value of every column in bold, and (global method) a paired bootstrap of the difference to a reference model.
Plus a models x methods matrix and curve overlays (ROC, TAR vs FAR target). Reads only saved result folders.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import compare as pair_compare
from .all_methods import METHODS, Method, discover
from .plots import GREY, INK, INK2, _style, pretty
from .report import _f, _md_table

MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e87ba4", "#eda100", "#008300", "#e34948"]   # categorical order of the palette


def find_model_dirs(results_root, split_version="v1"):
    """{label: results/<model>__<mode>} for every model folder holding at least one evaluation of this split."""
    out = {}
    for d in sorted(Path(results_root).iterdir()):
        if d.is_dir() and not d.name.startswith("comparison") and discover(d, split_version):
            out[d.name] = d
    return out


def _bold_best(df, higher, lower=()):
    """Bold the best value of each numeric-looking column (the string before ' ±')."""
    df = df.copy()
    for col in df.columns:
        vals = []
        for v in df[col]:
            try:
                vals.append(float(str(v).split(" ")[0]))
            except ValueError:
                vals.append(np.nan)
        arr = np.array(vals)
        if np.isnan(arr).all():
            continue
        if col in higher:
            k = int(np.nanargmax(arr))
        elif col in lower:
            k = int(np.nanargmin(arr))
        else:
            continue
        df.iloc[k, df.columns.get_loc(col)] = f"**{df.iloc[k, df.columns.get_loc(col)]}**"
    return df


def _method_table(key, entries):
    """entries: {model: Method}. Site-averaged table and pooled table, one row per model."""
    m0 = next(iter(entries.values()))
    order, fixed = m0.far_order, m0.fixed
    rows, prow = [], []
    for model, m in entries.items():
        sites = int(m.summary["headline"].get("site_averaged", {}).get("n_sites_used", len(m.per_site)))
        row = {"model": model, "sites": sites}
        for n in order:
            v, s = m.site_mean(f"tar_at_{n}", f"tar_at_{n}"), m.site_std(f"tar_at_{n}")
            row[f"TAR @ {pretty(n)}"] = f"{v:.3f}" + ("" if np.isnan(s) else f" ± {s:.3f}")
        for k, d in (("auc", 4), ("eer", 4)):
            v, s = m.site_mean(k, k), m.site_std(k)
            row[k.upper()] = f"{v:.{d}f}" + ("" if np.isnan(s) else f" ± {s:.{d}f}")
        for n in fixed:
            for b, label in (("bacc", "bal. acc"), ("acc", "acc")):
                v, s = m.site_mean(f"{b}_at_{n}", f"{b}_at_{n}"), m.site_std(f"{b}_at_{n}")
                row[f"{label} @ {pretty(n)}"] = f"{v:.3f}" + ("" if np.isnan(s) else f" ± {s:.3f}")
        rows.append(row)
        h = m.summary["headline"]["object_balanced"]
        p = {"model": model}
        for n in order:
            e = h[f"tar_at_{n}"]
            p[f"TAR @ {pretty(n)}"] = f"{e['value']:.3f}" + (f" [{e['ci_lo']:.3f}, {e['ci_hi']:.3f}]" if "ci_lo" in e else "")
        p["AUC"], p["EER"] = f"{h['auc']['value']:.4f}", f"{h['eer']['value']:.4f}"
        for n in fixed:
            p[f"bal. acc @ {pretty(n)}"] = f"{h[f'bacc_at_{n}']['value']:.3f}"
        prow.append(p)
    higher = [c for c in rows[0] if c.startswith(("TAR", "AUC", "bal. acc", "acc"))]
    return _bold_best(pd.DataFrame(rows), higher, lower=["EER"]), _bold_best(pd.DataFrame(prow), [c for c in prow[0] if c.startswith(("TAR", "AUC", "bal"))], lower=["EER"])


def _paired(entries, reference):
    """Paired bootstrap (over videos, same resamples) of the difference to the reference model, object-balanced. {model: DataFrame}."""
    out, notes = {}, []
    ref = entries[reference]
    for model, m in entries.items():
        if model == reference:
            continue
        try:
            df, _ = pair_compare.compare(ref.dir, m.dir)
            out[model] = df[(df.axis == "all") & (df.version == "object_balanced")].set_index("metric")
        except Exception as e:      # different units, < 5 units, ...
            notes.append(f"{model} vs {reference}: paired bootstrap not possible ({type(e).__name__}: {e})")
    return out, notes


def _fig_overlay(key, entries, fig_dir):
    """ROC averaged over sites and TAR-vs-target, one line per model (curves only)."""
    order = next(iter(entries.values())).far_order
    models = list(entries)
    grid = np.logspace(-4, 0, 300)
    paths = []
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=130)
    _style(ax)
    ax.grid(axis="x", color="#e6e5e0", lw=0.6)
    for i, (model, m) in enumerate(entries.items()):
        col = MODEL_COLORS[i % len(MODEL_COLORS)]
        tars = np.array([np.interp(grid, c["far"][::-1], c["tar"][::-1]) for c in m.curves().values()])
        ax.plot(grid, tars.mean(0), color=col, lw=1.6, label=model)
        far = [m.site_mean(f"far_at_{n}", f"far_at_{n}") for n in order]
        tar = [m.site_mean(f"tar_at_{n}", f"tar_at_{n}") for n in order]
        ax.plot(far, tar, "o", color=col, ms=4.5, zorder=4)
    ax.set_xscale("log"); ax.set_xlim(1e-4, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("FAR", fontsize=8, color=INK2); ax.set_ylabel("TAR (mean over sites)", fontsize=8, color=INK2)
    ax.set_title(f"{key}: ROC averaged over sites; dots = the FAR targets", fontsize=8.5, loc="left", color=INK)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout(); p = fig_dir / f"roc_{key}.png"; fig.savefig(p); plt.close(fig); paths.append(p.name)
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=130)
    _style(ax)
    for i, (model, m) in enumerate(entries.items()):
        col = MODEL_COLORS[i % len(MODEL_COLORS)]
        mu = np.array([m.site_mean(f"tar_at_{n}", f"tar_at_{n}") for n in order])
        sd = np.nan_to_num(np.array([m.site_std(f"tar_at_{n}") for n in order]))
        ax.errorbar(x + (i - (len(models) - 1) / 2) * 0.05, mu, yerr=sd, color=col, marker="o", ms=4.5, lw=1.5, capsize=2, label=model)
    ax.set_xticks(x); ax.set_xticklabels([pretty(n).replace("FAR ", "") for n in order], fontsize=8)
    ax.set_ylim(0, 1.02); ax.set_xlabel("FAR target", fontsize=8, color=INK2)
    ax.set_ylabel("TAR (mean over sites, bars = std over sites)", fontsize=8, color=INK2)
    ax.set_title(f"{key}: TAR at each FAR target", fontsize=8.5, loc="left", color=INK)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout(); p = fig_dir / f"tar_vs_target_{key}.png"; fig.savefig(p); plt.close(fig); paths.append(p.name)
    return paths


def build_model_comparison(results_root, split_version="v1", models=None, methods=None, reference=None, out=None):
    """Write <results_root>/comparison_<split>/{report.md, models_summary.csv, models_per_site.csv, figures/}. Returns the report path."""
    dirs = find_model_dirs(results_root, split_version)
    if models:
        dirs = {k: v for k, v in dirs.items() if k in models or any(k.startswith(m + "__") for m in models)}
    if len(dirs) < 1:
        return None
    out = Path(out) if out else Path(results_root) / f"comparison_{split_version}"
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    found = {label: {m.key: m for m in discover(d, split_version)} for label, d in dirs.items()}
    keys = [k for k, *_ in METHODS if any(k in f for f in found.values()) and (not methods or k in methods)]
    reference = reference if reference in dirs else next(iter(dirs))
    lines = [f"# Model comparison (split `{split_version}`)", "",
             "Models: " + ", ".join(f"`{k}`" for k in dirs) + f". Reference for the paired differences: `{reference}`.", "",
             "Every table has one row per model; the best value of each column is **bold** (highest, lowest for EER). Averages are over **sites** "
             "(± std over sites). All models must come from the same split and bins file. The evaluation methods are defined in each model's "
             "`<split>__all_methods/report.md`: *global* = one threshold per FAR target set on the validation videos (matching within each video); "
             "*per video / per site* = thresholds held out per video / site; *site matching* = gallery is the whole site (cross-video pairs assumed "
             "negative); *oracle* = tuned on the evaluated data (optimistic).", ""]
    long, per_site = [], []
    matrix = {}
    for key in keys:
        entries = {label: f[key] for label, f in found.items() if key in f}
        label_m = next(iter(entries.values())).label
        lines += [f"## {label_m}", ""]
        if len(entries) < len(dirs):
            lines += [f"*Only {len(entries)} of {len(dirs)} models have this method.*", ""]
        site_t, pooled_t = _method_table(key, entries)
        lines += ["**Averaged over sites**", "", _md_table(site_t), "**Pooled over all pairs (object-balanced, 95% CI over videos)**", "", _md_table(pooled_t)]
        if key == "global" and len(entries) >= 2:
            diffs, notes = _paired(entries, reference)
            if diffs:
                order = next(iter(entries.values())).far_order
                rows = []
                for model, d in diffs.items():
                    row = {"model": f"{model} − {reference}"}
                    for n in order:
                        r = d.loc[f"tar_at_{n}"]
                        row[f"ΔTAR @ {pretty(n)}"] = f"{r.diff_b_minus_a:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]" + (" *" if r.ci_excludes_0 else "")
                    r = d.loc["auc"]
                    row["ΔAUC"] = f"{r.diff_b_minus_a:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]" + (" *" if r.ci_excludes_0 else "")
                    rows.append(row)
                lines += [f"**Paired bootstrap vs `{reference}`** (same video resamples; * = 95% CI excludes 0)", "", _md_table(pd.DataFrame(rows))]
            lines += [f"*{n}*" for n in notes] + ([""] if notes else [])
        figs = _fig_overlay(key, entries, fig_dir)
        lines += [f"![roc {key}](figures/{figs[0]}) ![tar {key}](figures/{figs[1]})", ""]
        for model, m in entries.items():
            matrix.setdefault(model, {})[key] = m
            row = {"model": model, "method": key, "label": m.label, "n_sites": len(m.per_site)}
            for n in m.ops:
                for b in ("tar", "far", "acc", "bacc"):
                    row[f"{b}_at_{n}"] = m.site_mean(f"{b}_at_{n}", f"{b}_at_{n}")
                    row[f"{b}_at_{n}_std_over_sites"] = m.site_std(f"{b}_at_{n}")
            for k in ("auc", "eer"):
                row[k], row[f"{k}_std_over_sites"] = m.site_mean(k, k), m.site_std(k)
            long.append(row)
            for _, r in m.per_site.iterrows():
                per_site.append({"model": model, "method": key, **{c: r[c] for c in m.per_site.columns if c != "report" and not c.endswith(("_lo", "_hi"))}})
    # ---- models x methods matrix
    if keys:
        n1 = next(iter(next(iter(found.values())).values())).far_order
        pick = "1pct" if "1pct" in n1 else n1[0]
        mats = []
        for title, col, d in ((f"TAR @ {pretty(pick)} (mean over sites)", f"tar_at_{pick}", 3), ("AUC (mean over sites)", "auc", 4)):
            rows = []
            for model in dirs:
                row = {"model": model}
                for key in keys:
                    m = matrix.get(model, {}).get(key)
                    row[key] = f"{m.site_mean(col, col):.{d}f}" if m else "n/a"
                rows.append(row)
            mats.append((title, _bold_best(pd.DataFrame(rows), higher=keys)))
        lines += ["## Models × methods", ""] + [x for t, df in mats for x in (f"**{t}**", "", _md_table(df))]
    lines += ["## Files", "", "- `models_summary.csv`: site-averaged value and std over sites of every metric, one row per model and method",
              "- `models_per_site.csv`: every metric of every site, model and method (long format)"]
    pd.DataFrame(long).to_csv(out / "models_summary.csv", index=False)
    pd.DataFrame(per_site).to_csv(out / "models_per_site.csv", index=False)
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return out / "report.md"
