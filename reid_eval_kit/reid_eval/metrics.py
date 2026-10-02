"""Verification metrics computed from similarity histograms (vectorised over leading dimensions).

A *pack* is a dict of arrays with arbitrary leading dims L:
  h, hb    (L, 2, B)   similarity histograms, plain / object-balanced; axis -2: 0 = positive pairs, 1 = negative
  cnt,cntb (L, 2, Q)   pairs with similarity >= t_q (exact, counted with the real threshold)
  m1, m2   (L, 2)      sum of s and s^2 (plain);  m1b, m2b, nb: the same weighted (object-balanced)
Per-cell histograms have B=200 bins over [-1, 1]; AUC/EER/best-TAR/quantiles are therefore accurate to
about one bin (0.01); TAR/FAR at the thresholds are exact. The global 2,000-bin histogram is used for
thresholds and for the global AUC.
"""
from __future__ import annotations

import numpy as np

PACK_KEYS = ("h", "hb", "cnt", "cntb", "m1", "m2", "m1b", "m2b", "nb")
# number of trailing dims of each array that are NOT part of the cell index
TRAIL = {"h": 2, "hb": 2, "cnt": 2, "cntb": 2, "m1": 1, "m2": 1, "m1b": 1, "m2b": 1, "nb": 1}


def _div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)


def hist_quantile(h, q, lo=-1.0, hi=1.0):
    """q-quantile of a histogram (...,B) assuming uniform density inside a bin. NaN if empty."""
    B = h.shape[-1]
    c = np.cumsum(h, -1)
    tot = c[..., -1]
    target = q * tot
    k = np.minimum((c < target[..., None]).sum(-1), B - 1)
    prev = np.take_along_axis(c, np.maximum(k - 1, 0)[..., None], -1)[..., 0] * (k > 0)
    cur = np.take_along_axis(h, k[..., None], -1)[..., 0]
    frac = np.clip(_div(target - prev, cur), 0, 1)
    out = lo + (k + np.nan_to_num(frac)) * (hi - lo) / B
    return np.where(tot > 0, out, np.nan)


def roc(p, n):
    """FAR and TAR at the B+1 bin edges (thresholds), decreasing in the edge index. Shapes (...,B+1)."""
    P, N = p.sum(-1, keepdims=True), n.sum(-1, keepdims=True)
    tail = lambda h: np.concatenate([np.cumsum(h[..., ::-1], -1)[..., ::-1], np.zeros_like(h[..., :1])], -1)
    return _div(tail(n), N), _div(tail(p), P)


def auc(p, n):
    P, N = p.sum(-1), n.sum(-1)
    below = np.cumsum(n, -1) - n
    return _div((p * (below + 0.5 * n)).sum(-1), P * N)


def eer(p, n):
    far, tar = roc(p, n)
    d = far - (1 - tar)  # +1 at the lowest edge, -1 at the highest
    k = np.argmax(d <= 0, -1)
    k0 = np.maximum(k - 1, 0)
    g = lambda a, i: np.take_along_axis(a, i[..., None], -1)[..., 0]
    d0, d1 = g(d, k0), g(d, k)
    w = np.clip(_div(d0, d0 - d1), 0, 1)
    e = (1 - w) * g(far, k0) + w * g(far, k)
    return np.where(np.isnan(far[..., 0]) | np.isnan(tar[..., 0]), np.nan, e)


def best_tar_at_far(p, n, target):
    """TAR when the threshold is chosen inside the bin so that FAR = target (interpolated between edges)."""
    far, tar = roc(p, n)
    k = np.argmax(far <= target, -1)
    k0 = np.maximum(k - 1, 0)
    g = lambda a, i: np.take_along_axis(a, i[..., None], -1)[..., 0]
    f0, f1 = g(far, k0), g(far, k)
    w = np.clip(_div(f0 - target, f0 - f1), 0, 1)
    t = (1 - w) * g(tar, k0) + w * g(tar, k)
    return np.where(np.isnan(far[..., 0]) | np.isnan(tar[..., 0]), np.nan, t)


def threshold_for_far(neg_hist, target, lo=-1.0, hi=1.0):
    """Similarity t with FAR(t) = target on a negative histogram (linear inside the bin)."""
    B = len(neg_hist)
    N = neg_hist.sum()
    if N == 0:
        raise ValueError("no negative pairs")
    want = target * N
    tail = np.concatenate([np.cumsum(neg_hist[::-1])[::-1], [0]])  # tail[k] = negatives in bins >= k
    k = int(np.argmax(tail <= want))  # first edge whose tail is <= want
    if k == 0:
        return lo
    in_bin = neg_hist[k - 1]
    frac_above = (want - tail[k]) / in_bin if in_bin > 0 else 0.0
    width = (hi - lo) / B
    return float(lo + k * width - frac_above * width)


def compute(pack, balanced=False, far_target=1e-2):
    """All headline metrics of a pack (any leading dims). `balanced` selects the object-balanced version."""
    h = pack["hb" if balanced else "h"]
    cnt = pack["cntb" if balanced else "cnt"]
    p, n = h[..., 0, :], h[..., 1, :]
    P, N = p.sum(-1), n.sum(-1)
    m1, m2 = (pack["m1b"], pack["m2b"]) if balanced else (pack["m1"], pack["m2"])
    W = pack["nb"] if balanced else np.stack([P, N], -1)
    mean = _div(m1, W)
    var = np.maximum(_div(m2, W) - mean ** 2, 0)
    out = {
        "n_pos": P, "n_neg": N,
        "auc": auc(p, n), "eer": eer(p, n), "best_tar_far": best_tar_at_far(p, n, far_target),
        "dprime": _div(mean[..., 0] - mean[..., 1], np.sqrt((var[..., 0] + var[..., 1]) / 2)),
        "pos_mean": mean[..., 0], "neg_mean": mean[..., 1],
    }
    for name, hh in (("pos", p), ("neg", n)):
        out[f"{name}_median"] = hist_quantile(hh, 0.5)
        out[f"{name}_p5"] = hist_quantile(hh, 0.05)
        out[f"{name}_p95"] = hist_quantile(hh, 0.95)
    Wp, Wn = (pack["nb"][..., 0], pack["nb"][..., 1]) if balanced else (P, N)
    for q in range(cnt.shape[-1]):
        out[f"tar_q{q}"] = _div(cnt[..., 0, q], Wp)
        out[f"far_q{q}"] = _div(cnt[..., 1, q], Wn)
        out[f"frr_q{q}"] = 1 - out[f"tar_q{q}"]
        # accuracy over all pairs (positives accepted + negatives rejected) and balanced accuracy = (TAR + TNR) / 2.
        # Plain accuracy is dominated by the (much more numerous) negatives; balanced accuracy is not.
        out[f"acc_q{q}"] = _div(cnt[..., 0, q] + (Wn - cnt[..., 1, q]), Wp + Wn)
        out[f"bacc_q{q}"] = (out[f"tar_q{q}"] + (1 - out[f"far_q{q}"])) / 2
        # precision = accepted pairs that are truly the same vehicle = TP / (TP + FP); recall = TAR. Precision depends on the share of
        # negative pairs (a whole-site gallery has many more), so compare it across methods with care.
        out[f"prec_q{q}"] = _div(cnt[..., 0, q], cnt[..., 0, q] + cnt[..., 1, q])
    return out
