"""Static figures for the reports (matplotlib, Agg).

Colours: ordered operating points (FAR 0.1 % ... 10 %) use ONE hue, dark = strict FAR, light = loose FAR (an ordinal ramp);
blue / orange are the two aggregation versions in the ROC; hollow grey markers / grey cells = below minimum support.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

BLUE, ORANGE, GREY, INK, INK2 = "#2a78d6", "#eb6834", "#b9b8b2", "#0b0b0b", "#52514e"
RAMP = ["#0d366b", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef", "#b7d3f6"]   # strict -> loose FAR (blue 700 ... 150)
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def pretty(name):
    """'0.1pct' -> 'FAR 0.1%'"""
    return "FAR " + name.replace("pct", "%")


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GREY)
    ax.grid(axis="y", color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=8)


def _ramp(n):
    if n <= len(RAMP):
        idx = np.linspace(0, len(RAMP) - 2, n).round().astype(int) if n > 1 else [2]
        return [RAMP[i] for i in idx]
    return [RAMP[int(i)] for i in np.linspace(0, len(RAMP) - 1, n).round()]


def plot_axis(df: pd.DataFrame, axis: str, path, thr_names, prefix="balanced", title_suffix=""):
    """TAR at every FAR target per bin; bootstrap CIs when present; bins below minimum support drawn grey and hollow."""
    col = f"bin_{axis}"
    x = np.arange(len(df))
    order = sorted(thr_names, key=lambda n: float(n.replace("pct", "")))     # strict first
    colors = dict(zip(order, _ramp(len(order))))
    sup = df["supported"].to_numpy(bool)
    fig, ax = plt.subplots(figsize=(7.4, 3.8), dpi=130)
    _style(ax)
    for n in order:
        y = df[f"{prefix}_tar_at_{n}"].to_numpy(float)
        lo, hi = df.get(f"{prefix}_tar_at_{n}_lo"), df.get(f"{prefix}_tar_at_{n}_hi")
        ok = ~np.isnan(y)
        # draw the line through supported bins only (grey points are not connected)
        ax.plot(x[ok & sup], y[ok & sup], "-", color=colors[n], lw=1.8, zorder=2)
        for i in np.where(ok)[0]:
            if lo is not None and not np.isnan(lo.iloc[i]):
                ax.plot([x[i]] * 2, [lo.iloc[i], hi.iloc[i]], color=colors[n] if sup[i] else GREY, lw=1.1, zorder=1)
            ax.plot(x[i], y[i], "o", ms=4.5, mfc=colors[n] if sup[i] else "white", mec=colors[n] if sup[i] else GREY, mew=1.2, zorder=3)
        last = np.where(ok & sup)[0]
        if len(last):
            ax.annotate(pretty(n), (x[last[-1]], y[last[-1]]), xytext=(6, 0), textcoords="offset points", fontsize=7,
                        color=INK2, va="center")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\nn+={int(n):,}" for l, n in zip(df[col], df["n_pos"])], fontsize=7)
    ax.set_xlim(-0.4, len(df) - 0.2 + 0.9)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("TAR at the global threshold", color=INK2, fontsize=8)
    ax.set_title(f"{axis.replace('_', ' ')}: TAR per bin ({'object-balanced' if prefix == 'balanced' else 'pooled'}){title_suffix}; "
                 "hollow grey = low support", fontsize=8.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_heatmap(df: pd.DataFrame, a: str, b: str, path, thr_name, prefix="balanced"):
    ca, cb = f"bin_{a}", f"bin_{b}"
    metric = f"{prefix}_tar_at_{thr_name}"
    ra, rb = list(dict.fromkeys(df[ca])), list(dict.fromkeys(df[cb]))
    val = np.full((len(ra), len(rb)), np.nan)
    sup = np.zeros_like(val, bool)
    npos = np.zeros_like(val)
    for _, r in df.iterrows():
        i, j = ra.index(r[ca]), rb.index(r[cb])
        val[i, j], sup[i, j], npos[i, j] = r[metric], r["supported"], r["n_pos"]
    fig, ax = plt.subplots(figsize=(1.0 + 0.85 * len(rb), 0.9 + 0.5 * len(ra)), dpi=130)
    ax.imshow(np.ones_like(val), cmap="Greys", vmin=0, vmax=4, aspect="auto")  # grey base for low-support / empty cells
    shown = np.ma.masked_where(~sup | np.isnan(val), val)
    im = ax.imshow(shown, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    for i in range(len(ra)):
        for j in range(len(rb)):
            if npos[i, j] > 0:
                dark = sup[i, j] and not np.isnan(val[i, j]) and val[i, j] > 0.55
                ax.text(j, i, f"{val[i, j]:.2f}\n{int(npos[i, j]):,}" if not np.isnan(val[i, j]) else f"\n{int(npos[i, j]):,}",
                        ha="center", va="center", fontsize=6.5, color="white" if dark else (INK if sup[i, j] else INK2))
    ax.set_xticks(range(len(rb)))
    ax.set_xticklabels(rb, fontsize=7, rotation=30, ha="right")
    ax.set_yticks(range(len(ra)))
    ax.set_yticklabels(ra, fontsize=7)
    ax.set_xlabel(b.replace("_", " "), fontsize=8, color=INK2)
    ax.set_ylabel(a.replace("_", " "), fontsize=8, color=INK2)
    ax.set_title(f"{'object-balanced' if prefix == 'balanced' else 'pooled'} TAR at {pretty(thr_name)}\n"
                 "cell: value / positive pairs; grey = below minimum support", fontsize=7.5, loc="left", color=INK)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_roc(roc: pd.DataFrame, headline: dict, path, thr_names, title=""):
    """TAR vs FAR (log x). Dots = the global thresholds from validation, at the FAR/TAR they actually give here."""
    fig, ax = plt.subplots(figsize=(5.6, 4.0), dpi=130)
    _style(ax)
    ax.grid(axis="x", color="#e6e5e0", lw=0.6)
    for pre, c, lab, ls in (("balanced", BLUE, "object-balanced", "-"), ("pooled", ORANGE, "pooled", "--")):
        far, tar = roc[f"far_{pre}"].to_numpy(), roc[f"tar_{pre}"].to_numpy()
        ok = (far > 0) & ~np.isnan(tar)
        ax.plot(far[ok], tar[ok], ls, color=c, lw=1.8, label=lab)
    hv = headline["object_balanced"]
    for n in thr_names:
        fx, ty = hv[f"far_at_{n}"]["value"], hv[f"tar_at_{n}"]["value"]
        if fx > 0 and not np.isnan(ty):
            ax.plot(fx, ty, "o", color=INK, ms=4.5, zorder=4)
            ax.annotate(n.replace("pct", "%"), (fx, ty), xytext=(4, -10), textcoords="offset points", fontsize=7, color=INK2)
    ax.set_xscale("log")
    ax.set_xlim(1e-5, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("FAR (false accept rate)", fontsize=8, color=INK2)
    ax.set_ylabel("TAR (true accept rate)", fontsize=8, color=INK2)
    ax.set_title((title + "  " if title else "") + "ROC; dots = global thresholds from validation", fontsize=8.5, loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_groups(df: pd.DataFrame, label_col: str, path, thr_names, title=""):
    """One row per group (e.g. site): TAR at the strictest and loosest-but-useful FAR targets, sorted."""
    order = sorted(thr_names, key=lambda n: float(n.replace("pct", "")))
    show = [order[0]] + ([n for n in order if n == "1pct"] or order[1:2])
    show = list(dict.fromkeys(show))
    col = {n: c for n, c in zip(show, (RAMP[0], RAMP[3]))}
    d = df.sort_values(f"balanced_tar_at_{show[-1]}").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.4, 0.9 + 0.26 * len(d)), dpi=130)
    _style(ax)
    ax.grid(axis="x", color="#e6e5e0", lw=0.6)
    ax.grid(axis="y", visible=False)
    y = np.arange(len(d))
    for n in show:
        v = d[f"balanced_tar_at_{n}"].to_numpy(float)
        ax.plot(v, y, "o", color=col[n], ms=5, label=pretty(n))
        if f"balanced_tar_at_{n}_lo" in d:
            for i in y:
                if not np.isnan(d[f"balanced_tar_at_{n}_lo"].iloc[i]):
                    ax.plot([d[f"balanced_tar_at_{n}_lo"].iloc[i], d[f"balanced_tar_at_{n}_hi"].iloc[i]], [i, i], color=col[n], lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{s}  ({int(k)} video{'' if int(k) == 1 else 's'})" for s, k in zip(d[label_col], d["n_videos"])], fontsize=7)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("object-balanced TAR at the global threshold", fontsize=8, color=INK2)
    ax.set_title(title, fontsize=8.5, loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
