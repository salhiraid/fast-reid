"""Aggregation of per-video accumulators into bins, heatmaps, per-video / per-object tables and CIs (spec section 8).

Four aggregation versions are produced for every bin:
  pooled          sum of histograms/counts over videos (every pair counts equally)
  balanced        sum of the object-balanced histograms (every object / object pair counts equally)  <- main score
  video_averaged  metric per video, then mean over videos with enough pairs in that bin
  bin_balanced    mean of TAR@t over the supported delta-position bins
Confidence intervals: percentile bootstrap over VIDEOS only (pairs and crops are never resampled), recomputed from
the per-video accumulators; the same resample counts are used for every table so results are consistent.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from .bins import AXES, Bins, N_HIST

HEATMAPS = (("delta_position", "delta_azimuth"), ("delta_position", "keypoint_iou"),
            ("delta_azimuth", "keypoint_iou"), ("occlusion", "keypoint_iou"))
CI_METRICS = ("tar_t2", "tar_t3", "far_t2", "far_t3", "auc", "eer")


# ------------------------------------------------------------------ loading / packs
def load_acc(path):
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["meta"] = json.loads(str(d["meta"]))
    return d


def cells_pack(acc, bins: Bins):
    """Dense pack over all joint cells (leading dim n_cells)."""
    NC, ids = bins.n_cells, acc["cell_ids"]

    def dense(a, dtype=np.float64):
        out = np.zeros((NC,) + a.shape[1:], dtype)
        out[ids] = a
        return out
    return {"h": dense(acc["hist"]), "hb": dense(acc["hist_bal"]), "cnt": dense(acc["cnt"]), "cntb": dense(acc["cnt_bal"]),
            "m1": dense(acc["m1"]), "m2": dense(acc["m2"]), "m1b": dense(acc["m1_bal"]), "m2b": dense(acc["m2_bal"]),
            "nb": dense(acc["n_bal"])}


def reduce_pack(cells, bins: Bins, keep):
    """Sum joint cells over every axis not in `keep` (tuple of axis indices, ascending). Leading dims = kept axes."""
    shape = bins.cell_shape
    drop = tuple(a for a in range(4) if a not in keep)
    out = {}
    for k, v in cells.items():
        out[k] = v.reshape(shape + v.shape[1:]).sum(axis=drop)
    return out


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


# ------------------------------------------------------------------ bootstrap
def bootstrap_counts(V, B, seed):
    rng = np.random.default_rng(seed)
    return rng.multinomial(V, np.full(V, 1.0 / V), size=B).astype(np.float64)  # (B, V)


def resample_pack(stacked, counts):
    return {k: np.tensordot(counts, v, axes=(1, 0)) for k, v in stacked.items()}


def ci(x, level):
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanpercentile(x, [lo, hi], axis=0)


def named(m, bins: Bins):
    """Rename tar_q0 -> tar_t2 etc."""
    out = dict(m)
    for q, t in enumerate(bins.thr_names):
        for base in ("tar", "far", "frr"):
            out[f"{base}_{t}"] = out.pop(f"{base}_q{q}")
    return out


def video_avg(per_video_metrics, gate, counts=None):
    """Mean over videos where `gate` holds. per_video_metrics: (V, ...) ; counts (B, V) resample weights or None."""
    x = np.where(gate, per_video_metrics, np.nan)
    if counts is None:
        with np.errstate(all="ignore"):
            valid = ~np.isnan(x)
            return np.nansum(x, 0) / np.where(valid.sum(0) > 0, valid.sum(0), np.nan), valid.sum(0)
    valid = (~np.isnan(x)).astype(np.float64)
    num = np.tensordot(counts, np.nan_to_num(x), axes=(1, 0))
    den = np.tensordot(counts, valid, axes=(1, 0))
    with np.errstate(all="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


# ------------------------------------------------------------------ tables
def bin_table(bins: Bins, keep, pooled, stacked=None, counts=None, n_obj_pos=None, n_obj_neg=None):
    """One row per bin (or per bin combination when len(keep) > 1). Returns a DataFrame."""
    ft = bins.far_targets[0]
    mp, mb = named(M.compute(pooled, False, ft), bins), named(M.compute(pooled, True, ft), bins)
    shape = mp["n_pos"].shape
    cols = {}
    base = ["tar_t2", "tar_t3", "far_t2", "far_t3", "frr_t2", "frr_t3", "auc", "eer", "best_tar_far", "dprime",
            "pos_mean", "pos_median", "pos_p5", "pos_p95", "neg_mean", "neg_median", "neg_p5", "neg_p95"]
    for pre, m in (("pooled", mp), ("balanced", mb)):
        for k in base:
            cols[f"{pre}_{k}"] = m[k]
    cols["balanced_n_pos"], cols["balanced_n_neg"] = mb["n_pos"], mb["n_neg"]
    cols["n_pos"], cols["n_neg"] = mp["n_pos"], mp["n_neg"]
    if n_obj_pos is not None:
        cols["n_objects"], cols["n_object_pairs"] = n_obj_pos.reshape(shape), n_obj_neg.reshape(shape)
        sup = (mp["n_pos"] >= bins.min_pos_pairs) & (n_obj_pos.reshape(shape) >= bins.min_objects)
    else:
        sup = mp["n_pos"] >= bins.min_pos_pairs
    cols["supported"] = sup
    if stacked is not None:
        pv = named(M.compute(stacked, False, ft), bins)  # (V, *shape)
        gate = (pv["n_pos"] >= bins.vavg_pos) & (pv["n_neg"] >= bins.vavg_neg)
        for k in ("tar_t2", "tar_t3", "far_t2", "far_t3", "auc", "eer"):
            cols[f"video_avg_{k}"], nv = video_avg(pv[k], gate)
        cols["video_avg_n_videos"] = nv
        if counts is not None:
            rs = resample_pack(stacked, counts)
            for pre, bal in (("pooled", False), ("balanced", True)):
                rm = named(M.compute(rs, bal, ft), bins)
                for k in CI_METRICS:
                    lo, hi = ci(rm[k], bins.boot_level)
                    cols[f"{pre}_{k}_lo"], cols[f"{pre}_{k}_hi"] = lo, hi
            for k in ("tar_t2", "tar_t3", "auc"):
                lo, hi = ci(video_avg(pv[k], gate, counts), bins.boot_level)
                cols[f"video_avg_{k}_lo"], cols[f"video_avg_{k}_hi"] = lo, hi
    if keep:
        labels = [bins.labels(AXES[a]) for a in keep]
        rows = list(itertools.product(*[range(len(l)) for l in labels]))  # C order = reshape(-1) order of the metrics
        df = pd.DataFrame({f"bin_{AXES[a]}": [labels[n][r[n]] for r in rows] for n, a in enumerate(keep)})
    else:
        df = pd.DataFrame({"bin": ["all"]})
    for k, v in cols.items():
        df[k] = np.asarray(v).reshape(-1)
    return df


def kpmin_table(bins: Bins, accs):
    """Secondary keypoint breakdown (plain only): by min visible keypoints of the two crops."""
    h = sum(a["kpmin_hist"].astype(np.float64) for a in accs)
    cnt = sum(a["kpmin_cnt"].astype(np.float64) for a in accs)
    p, n = h[:, 0], h[:, 1]
    df = pd.DataFrame({"bin_min_visible_keypoints": bins.labels("min_visible_keypoints"), "n_pos": p.sum(1), "n_neg": n.sum(1)})
    for q, t in enumerate(bins.thr_names):
        df[f"tar_{t}"] = M._div(cnt[:, 0, q], p.sum(1))
        df[f"far_{t}"] = M._div(cnt[:, 1, q], n.sum(1))
    df["auc"], df["eer"] = M.auc(p, n), M.eer(p, n)
    df["supported"] = df["n_pos"] >= bins.min_pos_pairs
    return df


# ------------------------------------------------------------------ the whole thing
class Aggregation:
    """Reads per-video accumulators (test videos) and computes every table of the results folder."""

    def __init__(self, acc_paths, bins: Bins, thresholds: dict):
        self.bins, self.thresholds = bins, thresholds
        self.accs = [load_acc(p) for p in acc_paths]
        assert self.accs, "no accumulators to aggregate"
        for a in self.accs:
            if not np.allclose(a["thresholds"], thresholds["thresholds"]):
                raise ValueError(f"{a['meta']['video_id']}: accumulated with other thresholds than thresholds.json")
            if str(a["bins_sha256"]) != bins.sha256:
                raise ValueError(f"{a['meta']['video_id']}: accumulated with a different bins file")
        self.V = len(self.accs)
        self.counts = bootstrap_counts(self.V, bins.boot_resamples, bins.boot_seed)
        cells = [cells_pack(a, bins) for a in self.accs]
        self.pooled_cells = None
        for c in cells:
            self.pooled_cells = add_packs(self.pooled_cells, c)
        self.cells_per_video = cells
        self._tables = {}

    # --- one table for a set of axes
    def table(self, keep, ci_=True):
        key = (tuple(keep), ci_)
        if key in self._tables:
            return self._tables[key]
        pooled = reduce_pack(self.pooled_cells, self.bins, keep)
        stacked = stack_packs([reduce_pack(c, self.bins, keep) for c in self.cells_per_video]) if ci_ else None
        op = on = None
        for a in self.accs:
            p, n = objects_support(a, self.bins, keep)
            op = p if op is None else op + p
            on = n if on is None else on + n
        df = bin_table(self.bins, keep, pooled, stacked, self.counts if ci_ else None, op, on)
        self._tables[key] = df
        return df

    def axis_tables(self):
        return {AXES[a]: self.table((a,)) for a in range(4)}

    def heatmap_tables(self):
        return {(a, b): self.table((AXES.index(a), AXES.index(b)), ci_=False) for a, b in HEATMAPS}

    def cells_table(self):
        return self.table((0, 1, 2, 3), ci_=False)

    # --- headline numbers (four aggregation versions) with CIs
    def headline(self):
        g = self.table(())
        row = g.iloc[0]
        out = {"object_balanced": {}, "pooled": {}, "video_averaged": {}}
        for k in CI_METRICS + ("best_tar_far", "dprime", "frr_t2", "frr_t3"):
            for name, pre in (("object_balanced", "balanced"), ("pooled", "pooled")):
                e = {"value": float(row[f"{pre}_{k}"])}
                if f"{pre}_{k}_lo" in row:
                    e["ci_lo"], e["ci_hi"] = float(row[f"{pre}_{k}_lo"]), float(row[f"{pre}_{k}_hi"])
                out[name][k] = e
        for k in ("tar_t2", "tar_t3", "far_t2", "far_t3", "auc", "eer"):
            e = {"value": float(row[f"video_avg_{k}"])}
            if f"video_avg_{k}_lo" in row:
                e["ci_lo"], e["ci_hi"] = float(row[f"video_avg_{k}_lo"]), float(row[f"video_avg_{k}_hi"])
            out["video_averaged"][k] = e
        out["video_averaged"]["n_videos_used"] = int(row["video_avg_n_videos"])
        out["bin_balanced"] = self.bin_balanced()
        out["support"] = {k: (int(row[k]) if k in row else None) for k in ("n_pos", "n_neg", "n_objects", "n_object_pairs")}
        out["far_targets"] = {t: ft for t, ft in zip(self.bins.thr_names, self.bins.far_targets)}
        return out

    def bin_balanced(self):
        """Mean of TAR@t over the supported, known delta-position bins (pooled and object-balanced), with bootstrap CI."""
        df = self.table((0,))
        P = self.bins.n_known[0]
        use = df["supported"].to_numpy()[:P]
        res = {"bins_used": [l for l, u in zip(self.bins.labels("delta_position")[:P], use) if u]}
        if not use.any():
            return res
        stacked = stack_packs([reduce_pack(c, self.bins, (0,)) for c in self.cells_per_video])
        rs = resample_pack(stacked, self.counts)
        for name, bal in (("pooled", False), ("object_balanced", True)):
            m = named(M.compute(rs, bal, self.bins.far_targets[0]), self.bins)
            for t in self.bins.thr_names:
                pre = "balanced" if bal else "pooled"
                val = float(np.nanmean(df[f"{pre}_tar_{t}"].to_numpy()[:P][use]))
                lo, hi = ci(np.nanmean(m[f"tar_{t}"][:, :P][:, use], axis=1), self.bins.boot_level)
                res.setdefault(name, {})[f"tar_{t}"] = {"value": val, "ci_lo": float(lo), "ci_hi": float(hi)}
        return res

    # --- per video / per object
    def per_video(self):
        ft = self.bins.far_targets[0]
        rows = []
        t3 = self.thresholds["thresholds"][-1]
        for a, c in zip(self.accs, self.cells_per_video):
            pk = reduce_pack(c, self.bins, ())
            m = named(M.compute(pk, False, ft), self.bins)
            mb = named(M.compute(pk, True, ft), self.bins)
            meta = a["meta"]
            rows.append({
                "video_id": meta["video_id"], "site": meta.get("site"), "pov": meta.get("pov"),
                "n_objects": int(a["n_objects"]), "n_crops": int(a["n_crops"]),
                "n_pos": int(m["n_pos"]), "n_neg": int(m["n_neg"]),
                "auc": float(m["auc"]), "eer": float(m["eer"]),
                "tar_t2": float(m["tar_t2"]), "tar_t3": float(m["tar_t3"]),
                "far_t2": float(m["far_t2"]), "far_t3": float(m["far_t3"]),
                "balanced_tar_t3": float(mb["tar_t3"]),
                "confused_object_pairs": int((a["pair_med"] > t3).sum()), "n_object_pairs": int(len(a["pair_med"])),
            })
        return pd.DataFrame(rows)

    def per_object(self):
        t3 = self.thresholds["thresholds"][-1]
        P1 = self.bins.cell_shape[0]
        pos_labels = self.bins.labels("delta_position")
        rows = []
        for a in self.accs:
            T = int(a["n_objects"])
            hi_neg = np.full(T, -np.inf)
            nfalse = np.zeros(T, np.int64)
            for x, y, mx in zip(a["pair_a"], a["pair_b"], a["pair_max"]):
                hi_neg[x] = max(hi_neg[x], mx); hi_neg[y] = max(hi_neg[y], mx)
                if mx >= t3:
                    nfalse[x] += 1; nfalse[y] += 1
            tar = np.divide(a["obj_tar_num"][-1], a["obj_tar_den"], out=np.full((T, P1), np.nan), where=a["obj_tar_den"] > 0)
            for t in range(T):
                low = a["obj_low_sim"][t]
                r = {"video_id": a["meta"]["video_id"], "tracklet_id": str(a["tracklet_ids"][t]), "n_crops": int(a["obj_n"][t]),
                     "lowest_positive_sim": None if np.isnan(low) else float(low),
                     "lowest_positive_crop_a": str(a["obj_low_uid"][t, 0]), "lowest_positive_crop_b": str(a["obj_low_uid"][t, 1]),
                     "n_other_objects_falsely_matched_t3": int(nfalse[t]),
                     "highest_negative_sim": None if np.isinf(hi_neg[t]) else float(hi_neg[t]),
                     "margin": None if (np.isnan(low) or np.isinf(hi_neg[t])) else float(low - hi_neg[t])}
                for k, lab in enumerate(pos_labels):
                    r[f"tar_t3_dpos{lab}"] = None if np.isnan(tar[t, k]) else float(tar[t, k])
                rows.append(r)
        return pd.DataFrame(rows)

    def kpmin(self):
        return kpmin_table(self.bins, self.accs)
