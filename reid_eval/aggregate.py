"""Aggregation of per-video accumulators into bins, heatmaps, ROC, per-video / per-site / per-object tables and CIs.

Four aggregation versions are produced for every bin:
  pooled          sum of histograms/counts over videos (every pair counts equally)
  balanced        sum of the object-balanced histograms (every object / object pair counts equally)  <- main score
  video_averaged  metric per video, then mean over videos with enough pairs in that bin
  bin_balanced    mean of TAR@FAR over the supported delta-position bins
Confidence intervals: percentile bootstrap over VIDEOS only (pairs and crops are never resampled), recomputed from the
per-video summaries. They are only computed when at least MIN_VIDEOS_FOR_CI videos are aggregated (a single video, or a
site with 2 videos, has no meaningful video-level bootstrap).

Every aggregation works on a list of `VideoSummary`, so the same code gives the global result, one site, or one video.
"""
from __future__ import annotations

import itertools
import json
import warnings

import numpy as np
import pandas as pd

from . import metrics as M
from .bins import AXES, Bins, N_FINE

HEATMAPS = (("delta_position", "delta_azimuth"), ("delta_position", "keypoint_iou"),
            ("delta_azimuth", "keypoint_iou"), ("occlusion", "keypoint_iou"))
AXIS_KEEPS = [(a,) for a in range(4)]
HEAT_KEEPS = [(AXES.index(a), AXES.index(b)) for a, b in HEATMAPS]
PACK_KEEPS = [()] + AXIS_KEEPS + HEAT_KEEPS          # reduced packs stored per video (small)
FULL = (0, 1, 2, 3)                                  # all joint cells (pooled only: cells.csv)
MIN_VIDEOS_FOR_CI = 5


# ------------------------------------------------------------------ packs
def load_acc(path):
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["meta"] = json.loads(str(d["meta"]))
    return d


def cells_pack(acc, bins: Bins):
    """Dense pack over all joint cells (leading dim n_cells)."""
    NC, ids = bins.n_cells, acc["cell_ids"]

    def dense(a):
        out = np.zeros((NC,) + a.shape[1:], np.float64)
        out[ids] = a
        return out
    return {"h": dense(acc["hist"]), "hb": dense(acc["hist_bal"]), "cnt": dense(acc["cnt"]), "cntb": dense(acc["cnt_bal"]),
            "m1": dense(acc["m1"]), "m2": dense(acc["m2"]), "m1b": dense(acc["m1_bal"]), "m2b": dense(acc["m2_bal"]),
            "nb": dense(acc["n_bal"])}


def reduce_pack(cells, bins: Bins, keep):
    """Sum joint cells over every axis not in `keep` (tuple of axis indices, ascending). Leading dims = kept axes."""
    shape = bins.cell_shape
    drop = tuple(a for a in range(4) if a not in keep)
    return {k: v.reshape(shape + v.shape[1:]).sum(axis=drop) for k, v in cells.items()}


def add_packs(a, b):
    return b if a is None else {k: a[k] + b[k] for k in a}


def stack_packs(packs):
    return {k: np.stack([p[k] for p in packs]) for k in packs[0]}


def objects_support(acc, bins: Bins, keep):
    """(distinct objects with positives, distinct negative object pairs) per bin of the kept axes, for one video."""
    shape = bins.cell_shape
    sel_shape = tuple(shape[a] for a in keep) or (1,)
    T = int(acc["n_objects"])

    def flat(cells):
        if not keep:
            return np.zeros(len(cells), np.int64)
        idx = np.unravel_index(cells, shape)
        return np.ravel_multi_index(tuple(idx[a] for a in keep), sel_shape)
    pk, nk = acc["pos_keys"], acc["neg_keys"]
    n_pos = np.zeros(int(np.prod(sel_shape)), np.int64)
    n_neg = np.zeros_like(n_pos)
    if len(pk):
        u = np.unique(flat(pk[:, 0]) * T + pk[:, 1])
        n_pos = np.bincount(u // T, minlength=len(n_pos))
    if len(nk):
        u = np.unique((flat(nk[:, 0]) * T + nk[:, 1]) * T + nk[:, 2])
        n_neg = np.bincount(u // (T * T), minlength=len(n_neg))
    return n_pos.reshape(sel_shape), n_neg.reshape(sel_shape)


class VideoSummary:
    """Everything the aggregation needs from one video; the big per-cell arrays are reduced and dropped."""

    def __init__(self, acc, bins: Bins, cells):
        self.meta = acc["meta"]
        self.video_id = acc["meta"]["video_id"]
        self.site = acc["meta"].get("site") or "unknown"
        self.n_crops, self.n_objects = int(acc["n_crops"]), int(acc["n_objects"])
        self.tracklet_ids = acc["tracklet_ids"]
        self.packs = {k: reduce_pack(cells, bins, k) for k in PACK_KEEPS}
        self.objs = {k: objects_support(acc, bins, k) for k in PACK_KEEPS + [FULL]}
        for k in ("fine", "fine_bal", "kpmin_hist", "kpmin_cnt", "pair_a", "pair_b", "pair_med", "pair_max", "obj_n",
                  "obj_tar_num", "obj_tar_den", "obj_low_sim", "obj_low_uid", "thresholds"):
            setattr(self, k, acc[k])
        self.fail = {k: v for k, v in acc.items() if k.startswith("fail_")}


def load_summaries(paths, bins: Bins, thresholds: dict):
    """Returns (list of VideoSummary, pooled dense cells pack)."""
    out, pooled = [], None
    for p in paths:
        acc = load_acc(p)
        if not np.allclose(acc["thresholds"], thresholds["thresholds"]):
            raise ValueError(f"{acc['meta']['video_id']}: accumulated with other thresholds than thresholds.json")
        if str(acc["bins_sha256"]) != bins.sha256:
            raise ValueError(f"{acc['meta']['video_id']}: accumulated with a different bins file")
        cells = cells_pack(acc, bins)
        out.append(VideoSummary(acc, bins, cells))
        pooled = add_packs(pooled, cells)
    return out, pooled


# ------------------------------------------------------------------ bootstrap
def bootstrap_counts(V, B, seed):
    rng = np.random.default_rng(seed)
    return rng.multinomial(V, np.full(V, 1.0 / V), size=B).astype(np.float64)  # (B, V)


def resample_pack(stacked, counts):
    return {k: np.tensordot(counts, v, axes=(1, 0)) for k, v in stacked.items()}


def ci(x, level):
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanpercentile(x, [lo, hi], axis=0)


def named(m, bins: Bins):
    """Rename tar_q0 -> tar_at_0.1pct, acc_q5 -> acc_at_th0.5, etc. (FAR thresholds first, then fixed thresholds)."""
    out = dict(m)
    for q, t in enumerate(bins.op_names):
        for base in ("tar", "far", "frr", "acc", "bacc"):
            out[f"{base}_at_{t}"] = out.pop(f"{base}_q{q}")
    return out


def video_avg(per_video_metrics, gate, counts=None):
    """Mean over videos where `gate` holds. per_video_metrics: (V, ...) ; counts (B, V) resample weights or None."""
    x = np.where(gate, per_video_metrics, np.nan)
    valid = (~np.isnan(x)).astype(np.float64)
    if counts is None:
        n = valid.sum(0)
        with np.errstate(all="ignore"):
            return np.nansum(x, 0) / np.where(n > 0, n, np.nan), n.astype(int)
    num = np.tensordot(counts, np.nan_to_num(x), axes=(1, 0))
    den = np.tensordot(counts, valid, axes=(1, 0))
    with np.errstate(all="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def metric_names(bins: Bins):
    ops = bins.op_names
    return ([f"{b}_at_{n}" for b in ("tar", "far", "acc", "bacc") for n in ops] + ["auc", "eer"])


# ------------------------------------------------------------------ tables
def bin_table(bins: Bins, keep, pooled, stacked=None, counts=None, n_obj_pos=None, n_obj_neg=None):
    """One row per bin (or per bin combination when len(keep) > 1). Returns a DataFrame."""
    ft = bins.best_far
    mp, mb = named(M.compute(pooled, False, ft), bins), named(M.compute(pooled, True, ft), bins)
    shape = mp["n_pos"].shape
    names = bins.op_names
    base = ([f"{b}_at_{n}" for b in ("tar", "far", "frr", "acc", "bacc") for n in names]
            + ["auc", "eer", "best_tar_far", "dprime", "pos_mean", "pos_median", "pos_p5", "pos_p95",
               "neg_mean", "neg_median", "neg_p5", "neg_p95"])
    cols = {"n_pos": mp["n_pos"], "n_neg": mp["n_neg"], "balanced_n_pos": mb["n_pos"], "balanced_n_neg": mb["n_neg"]}
    for pre, m in (("pooled", mp), ("balanced", mb)):
        for k in base:
            cols[f"{pre}_{k}"] = m[k]
    if n_obj_pos is not None:
        cols["n_objects"], cols["n_object_pairs"] = n_obj_pos.reshape(shape), n_obj_neg.reshape(shape)
        cols["supported"] = (mp["n_pos"] >= bins.min_pos_pairs) & (n_obj_pos.reshape(shape) >= bins.min_objects)
    else:
        cols["supported"] = mp["n_pos"] >= bins.min_pos_pairs
    if stacked is not None:
        pv = named(M.compute(stacked, False, ft), bins)  # (V, *shape)
        gate = (pv["n_pos"] >= bins.vavg_pos) & (pv["n_neg"] >= bins.vavg_neg)
        for k in metric_names(bins):
            cols[f"video_avg_{k}"], nv = video_avg(pv[k], gate)
        cols["video_avg_n_videos"] = nv
        if counts is not None:
            rs = resample_pack(stacked, counts)
            for pre, bal in (("pooled", False), ("balanced", True)):
                rm = named(M.compute(rs, bal, ft), bins)
                for k in metric_names(bins):
                    cols[f"{pre}_{k}_lo"], cols[f"{pre}_{k}_hi"] = ci(rm[k], bins.boot_level)
            for k in [f"tar_at_{n}" for n in names] + ["auc"]:
                cols[f"video_avg_{k}_lo"], cols[f"video_avg_{k}_hi"] = ci(video_avg(pv[k], gate, counts), bins.boot_level)
    if keep:
        labels = [bins.labels(AXES[a]) for a in keep]
        rows = list(itertools.product(*[range(len(l)) for l in labels]))  # C order = reshape(-1) order of the metrics
        df = pd.DataFrame({f"bin_{AXES[a]}": [labels[n][r[n]] for r in rows] for n, a in enumerate(keep)})
    else:
        df = pd.DataFrame({"bin": ["all"]})
    return pd.concat([df, pd.DataFrame({k: np.asarray(v).reshape(-1) for k, v in cols.items()})], axis=1)


def kpmin_table(bins: Bins, summaries):
    """Secondary keypoint breakdown (plain only): by min visible keypoints of the two crops."""
    h = sum(s.kpmin_hist.astype(np.float64) for s in summaries)
    cnt = sum(s.kpmin_cnt.astype(np.float64) for s in summaries)
    p, n = h[:, 0], h[:, 1]
    df = pd.DataFrame({"bin_min_visible_keypoints": bins.labels("min_visible_keypoints"), "n_pos": p.sum(1), "n_neg": n.sum(1)})
    for q, t in enumerate(bins.op_names):
        df[f"tar_at_{t}"] = M._div(cnt[:, 0, q], p.sum(1))
        df[f"far_at_{t}"] = M._div(cnt[:, 1, q], n.sum(1))
        df[f"acc_at_{t}"] = M._div(cnt[:, 0, q] + n.sum(1) - cnt[:, 1, q], p.sum(1) + n.sum(1))
    df["auc"], df["eer"] = M.auc(p, n), M.eer(p, n)
    df["supported"] = df["n_pos"] >= bins.min_pos_pairs
    return df


# ------------------------------------------------------------------ the whole thing
class Aggregation:
    """Aggregates a list of VideoSummary: all test videos, the videos of one site, or a single video."""

    def __init__(self, summaries, bins: Bins, thresholds: dict, pooled_cells=None):
        assert summaries, "nothing to aggregate"
        self.s, self.bins, self.thresholds, self.pooled_cells = list(summaries), bins, thresholds, pooled_cells
        self.V = len(self.s)
        self.use_ci = self.V >= MIN_VIDEOS_FOR_CI
        self.counts = bootstrap_counts(self.V, bins.boot_resamples, bins.boot_seed) if self.use_ci else None
        self._cache = {}

    def pooled(self, keep):
        return add_all([s.packs[keep] for s in self.s])

    def stacked(self, keep):
        return stack_packs([s.packs[keep] for s in self.s])

    def objects(self, keep):
        return (sum(s.objs[keep][0] for s in self.s), sum(s.objs[keep][1] for s in self.s))

    def table(self, keep, ci_=True):
        keep = tuple(keep)
        ci_ = ci_ and self.use_ci
        key = (keep, ci_)
        if key not in self._cache:
            pooled = self.pooled_cells if keep == FULL else self.pooled(keep)
            stacked = self.stacked(keep) if (ci_ and keep != FULL) else None
            op, on = self.objects(keep)
            self._cache[key] = bin_table(self.bins, keep, pooled, stacked, self.counts if ci_ else None, op, on)
        return self._cache[key]

    def axis_tables(self):
        return {AXES[a]: self.table((a,)) for a in range(4)}

    def heatmap_tables(self):
        return {(a, b): self.table(k, ci_=False) for (a, b), k in zip(HEATMAPS, HEAT_KEEPS)}

    def cells_table(self):
        return self.table(FULL, ci_=False)

    # --- headline numbers (four aggregation versions) with CIs
    def headline(self):
        row = self.table(()).iloc[0]
        names = self.bins.op_names
        out = {"object_balanced": {}, "pooled": {}, "video_averaged": {}}
        for k in metric_names(self.bins) + ["best_tar_far", "dprime"] + [f"frr_at_{n}" for n in names]:
            for name, pre in (("object_balanced", "balanced"), ("pooled", "pooled")):
                e = {"value": float(row[f"{pre}_{k}"])}
                if f"{pre}_{k}_lo" in row:
                    e["ci_lo"], e["ci_hi"] = float(row[f"{pre}_{k}_lo"]), float(row[f"{pre}_{k}_hi"])
                out[name][k] = e
        if "video_avg_n_videos" in row:
            for k in metric_names(self.bins):
                e = {"value": float(row[f"video_avg_{k}"])}
                if f"video_avg_{k}_lo" in row:
                    e["ci_lo"], e["ci_hi"] = float(row[f"video_avg_{k}_lo"]), float(row[f"video_avg_{k}_hi"])
                out["video_averaged"][k] = e
            out["video_averaged"]["n_videos_used"] = int(row["video_avg_n_videos"])
        out["bin_balanced"] = self.bin_balanced()
        out["support"] = {k: (int(row[k]) if k in row else None) for k in ("n_pos", "n_neg", "n_objects", "n_object_pairs")}
        out["far_targets"] = dict(zip(self.bins.thr_names, self.bins.far_targets))
        out["fixed_thresholds"] = dict(zip(self.bins.fixed_names, self.bins.fixed_thresholds))
        return out

    def bin_balanced(self):
        """Mean of TAR@FAR over the supported, known delta-position bins (pooled and object-balanced), CI if available."""
        df = self.table((0,))
        P = self.bins.n_known[0]
        use = df["supported"].to_numpy()[:P]
        res = {"bins_used": [l for l, u in zip(self.bins.labels("delta_position")[:P], use) if u]}
        if not use.any():
            return res
        rs = None
        if self.use_ci:
            rs = resample_pack(self.stacked((0,)), self.counts)
        for name, bal in (("pooled", False), ("object_balanced", True)):
            pre = "balanced" if bal else "pooled"
            m = named(M.compute(rs, bal, self.bins.best_far), self.bins) if rs is not None else None
            for t in self.bins.thr_names:
                e = {"value": float(np.nanmean(df[f"{pre}_tar_at_{t}"].to_numpy()[:P][use]))}
                if m is not None:
                    e["ci_lo"], e["ci_hi"] = (float(x) for x in ci(np.nanmean(m[f"tar_at_{t}"][:, :P][:, use], axis=1), self.bins.boot_level))
                res.setdefault(name, {})[f"tar_at_{t}"] = e
        return res

    # --- ROC (from the 2,000-bin fine histograms)
    def roc(self):
        fine = sum(s.fine.astype(np.float64) for s in self.s)
        fineb = sum(s.fine_bal for s in self.s)
        far_p, tar_p = M.roc(fine[0], fine[1])
        far_b, tar_b = M.roc(fineb[0], fineb[1])
        out = {"threshold": np.linspace(-1, 1, N_FINE + 1), "far_pooled": far_p, "tar_pooled": tar_p,
               "far_balanced": far_b, "tar_balanced": tar_b}
        for pre, hist in (("pooled", fine), ("balanced", fineb)):    # accuracy at EVERY threshold (edge k = similarity >= edge k)
            P, N = hist[0].sum(), hist[1].sum()
            tp, fp = (np.concatenate([np.cumsum(h[::-1])[::-1], [0]]) for h in (hist[0], hist[1]))
            tn = N - fp
            out[f"acc_{pre}"] = M._div(tp + tn, P + N)
            out[f"bacc_{pre}"] = (M._div(tp, P) + M._div(tn, N)) / 2
        return pd.DataFrame(out)

    # --- per video / per site / per object
    def _summary_row(self, summaries):
        """One-line summary of a group of videos (a video or a site) from its global pack."""
        agg = Aggregation(summaries, self.bins, self.thresholds)
        t = agg.table(()).iloc[0]
        names = self.bins.op_names
        row = {"n_videos": len(summaries), "n_objects": int(sum(s.n_objects for s in summaries)),
               "n_crops": int(sum(s.n_crops for s in summaries)), "n_pos": int(t["n_pos"]), "n_neg": int(t["n_neg"]),
               "auc": t["pooled_auc"], "eer": t["pooled_eer"]}
        for n in names:
            row[f"tar_at_{n}"] = t[f"pooled_tar_at_{n}"]
            if f"balanced_tar_at_{n}_lo" in t:
                row[f"balanced_tar_at_{n}"] = t[f"balanced_tar_at_{n}"]
                row[f"balanced_tar_at_{n}_lo"], row[f"balanced_tar_at_{n}_hi"] = t[f"balanced_tar_at_{n}_lo"], t[f"balanced_tar_at_{n}_hi"]
            else:
                row[f"balanced_tar_at_{n}"] = t[f"balanced_tar_at_{n}"]
        for n in names:
            row[f"far_at_{n}"] = t[f"pooled_far_at_{n}"]
            row[f"acc_at_{n}"], row[f"bacc_at_{n}"] = t[f"pooled_acc_at_{n}"], t[f"pooled_bacc_at_{n}"]
            row[f"balanced_acc_at_{n}"], row[f"balanced_bacc_at_{n}"] = t[f"balanced_acc_at_{n}"], t[f"balanced_bacc_at_{n}"]
        thr = self.thresholds["thresholds"][self.bins.strict_idx]
        row["confused_object_pairs"] = int(sum((s.pair_med > thr).sum() for s in summaries))
        row["n_object_pairs"] = int(sum(len(s.pair_med) for s in summaries))
        return row

    def per_video(self):
        rows = []
        for s in self.s:
            rows.append({"video_id": s.video_id, "site": s.site, "pov": s.meta.get("pov"), **self._summary_row([s])})
        return pd.DataFrame(rows).drop(columns=["n_videos"])

    def per_site(self):
        rows = []
        for site in sorted({s.site for s in self.s}):
            sub = [s for s in self.s if s.site == site]
            rows.append({"site": site, **self._summary_row(sub)})
        return pd.DataFrame(rows)

    def per_object(self):
        thr = self.thresholds["thresholds"][self.bins.strict_idx]
        n = self.bins.strict_name
        P1 = self.bins.cell_shape[0]
        pos_labels = self.bins.labels("delta_position")
        rows = []
        for s in self.s:
            T = s.n_objects
            hi_neg = np.full(T, -np.inf)
            nfalse = np.zeros(T, np.int64)
            for x, y, mx in zip(s.pair_a, s.pair_b, s.pair_max):
                hi_neg[x] = max(hi_neg[x], mx); hi_neg[y] = max(hi_neg[y], mx)
                if mx >= thr:
                    nfalse[x] += 1; nfalse[y] += 1
            tar = np.divide(s.obj_tar_num[self.bins.strict_idx], s.obj_tar_den, out=np.full((T, P1), np.nan), where=s.obj_tar_den > 0)
            for t in range(T):
                low = s.obj_low_sim[t]
                r = {"video_id": s.video_id, "tracklet_id": str(s.tracklet_ids[t]), "n_crops": int(s.obj_n[t]),
                     "lowest_positive_sim": None if np.isnan(low) else float(low),
                     "lowest_positive_crop_a": str(s.obj_low_uid[t, 0]), "lowest_positive_crop_b": str(s.obj_low_uid[t, 1]),
                     f"n_other_objects_falsely_matched_at_{n}": int(nfalse[t]),
                     "highest_negative_sim": None if np.isinf(hi_neg[t]) else float(hi_neg[t]),
                     "margin": None if (np.isnan(low) or np.isinf(hi_neg[t])) else float(low - hi_neg[t])}
                for k, lab in enumerate(pos_labels):
                    r[f"tar_at_{n}_dpos{lab}"] = None if np.isnan(tar[t, k]) else float(tar[t, k])
                rows.append(r)
        return pd.DataFrame(rows)

    def kpmin(self):
        return kpmin_table(self.bins, self.s)

    def has_keypoints(self):
        return any(s.meta.get("keypoints_present") for s in self.s)


def add_all(packs):
    out = None
    for p in packs:
        out = add_packs(out, p)
    return out
