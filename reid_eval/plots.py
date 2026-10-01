"""Static figures for report.md (matplotlib, Agg). Colours: blue = primary series, orange = second; greyed = low support."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

BLUE, ORANGE, GREY, INK, INK2 = "#2a78d6", "#eb6834", "#b9b8b2", "#0b0b0b", "#52514e"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GREY)
    ax.grid(axis="y", color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=8)


def plot_axis(df: pd.DataFrame, axis: str, path, prefix="balanced", thr_names=("t2", "t3"), far_labels=("1e-2", "1e-3")):
    """TAR at both thresholds per bin with bootstrap CIs; bins below minimum support drawn grey and hollow."""
    col = f"bin_{axis}"
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=130)
    _style(ax)
    for t, far, c, dx in zip(thr_names, far_labels, (ORANGE, BLUE), (-0.08, 0.08)):
        y = df[f"{prefix}_tar_{t}"].to_numpy(float)
        lo, hi = df.get(f"{prefix}_tar_{t}_lo"), df.get(f"{prefix}_tar_{t}_hi")
        sup = df["supported"].to_numpy(bool)
        ok = ~np.isnan(y)
        ax.plot(x[ok & sup] + dx, y[ok & sup], "-", color=c, lw=2, label=f"TAR @ FAR {far}", zorder=2)
        for i in np.where(ok)[0]:
            if lo is not None and not np.isnan(lo.iloc[i]):
                ax.plot([x[i] + dx] * 2, [lo.iloc[i], hi.iloc[i]], color=c if sup[i] else GREY, lw=1.2, zorder=1)
            ax.plot(x[i] + dx, y[i], "o", ms=5.5, mfc=c if sup[i] else "white", mec=c if sup[i] else GREY, mew=1.4, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\nn+={int(n):,}" for l, n in zip(df[col], df["n_pos"])], fontsize=7)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("TAR at global threshold", color=INK2, fontsize=8)
    ax.set_title(f"{axis.replace('_', ' ')}: object-balanced TAR, 95% CI over videos (hollow grey = low support)", fontsize=8.5,
                 color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_heatmap(df: pd.DataFrame, a: str, b: str, path, metric="balanced_tar_t3", title_metric="object-balanced TAR @ t(1e-3)"):
    ca, cb = f"bin_{a}", f"bin_{b}"
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
    ax.set_xticks(range(len(rb)), rb, fontsize=7, rotation=30, ha="right")
    ax.set_yticks(range(len(ra)), ra, fontsize=7)
    ax.set_xlabel(b.replace("_", " "), fontsize=8, color=INK2)
    ax.set_ylabel(a.replace("_", " "), fontsize=8, color=INK2)
    ax.set_title(f"{title_metric}\ncell: value / positive pairs; grey = below minimum support", fontsize=7.5, loc="left", color=INK)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
