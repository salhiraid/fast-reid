"""Orchestration of the verification evaluation: validation thresholds -> per-video accumulators -> results files."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from reid_data import group_by_video, load_dataset_with_info
from . import metrics as M
from .accumulate import VideoMeta, accumulate_video, fine_histograms
from .aggregate import AXES, Aggregation, HEATMAPS
from .bins import N_FINE, Bins
from .common import read_json, sha256_file, write_json_atomic
from .failures import collect, contact_sheets
from .split import check_fingerprints
from .templates import SETS, check_alignment, load_video_npz, video_file


def load_video(tdir, set_name, vid, records):
    """Template rows joined with the dataset records; fails if any record is missing/extra/reordered."""
    t = load_video_npz(video_file(tdir, set_name, vid))
    check_alignment(t, records, f"{set_name}/{vid}")
    if not np.isfinite(t["emb"]).all():
        raise ValueError(f"{set_name}/{vid}: NaN/Inf in embeddings")
    return t["emb"], VideoMeta.from_records(records)


def compute_thresholds(tdir, split, by_video, bins: Bins, device=None):
    """Thresholds t(FAR target) from the pooled VALIDATION negative pairs. Test videos are never touched."""
    fine = np.zeros((2, N_FINE), np.int64)
    n_obj_pairs = 0
    for vid in split["validation"]:
        emb, meta = load_video(tdir, "validation", vid, by_video[vid])
        fine += fine_histograms(emb, meta, device)
        T = len(meta.tracklet_ids)
        n_obj_pairs += T * (T - 1) // 2
    neg, pos = fine[1], fine[0]
    thr = [M.threshold_for_far(neg, t) for t in bins.far_targets]
    if neg.sum() * min(bins.far_targets) < 10:
        print(f"WARNING: only {int(neg.sum())} validation negatives for FAR target {min(bins.far_targets):g}", file=sys.stderr)
    return {
        "source": "validation", "far_targets": bins.far_targets, "names": bins.thr_names, "thresholds": thr,
        "n_negative_pairs": int(neg.sum()), "n_positive_pairs": int(pos.sum()), "n_distinct_object_pairs": int(n_obj_pairs),
        "validation_auc": float(M.auc(pos, neg)),
        "validation_far_at_thresholds": [float(neg[int((t + 1) * N_FINE / 2):].sum() / neg.sum()) for t in thr],
        "note": "threshold interpolated inside a 0.001-wide bin of the pooled validation negatives; "
                "the FAR actually measured on validation/test is in summary.json",
    }


def select_mode(template_dirs, split, by_video, device=None):
    """Pick the preprocessing mode with the higher pooled AUC on validation videos."""
    res = {}
    for d in template_dirs:
        manifest = read_json(Path(d) / "manifest.json")
        fine = np.zeros((2, N_FINE), np.int64)
        for vid in split["validation"]:
            emb, meta = load_video(d, "validation", vid, by_video[vid])
            fine += fine_histograms(emb, meta, device)
        res[manifest["preproc_mode"]] = float(M.auc(fine[0], fine[1]))
    best = max(res, key=res.get)
    return {"validation_pooled_auc": res, "chosen": best, "split_version": split.get("version"), "chosen_on": "validation only"}


def _strings(a):
    return np.asarray(a, dtype=str) if len(a) else np.zeros((0,), dtype=str)


def accumulate_set(tdir, split, set_name, by_video, infos, bins, thresholds, acc_dir, device=None, verbose=True):
    paths = []
    for k, vid in enumerate(split[set_name]):
        recs = by_video[vid]
        emb, meta = load_video(tdir, set_name, vid, recs)
        acc = accumulate_video(emb, meta, bins, thresholds, device)
        uid = lambda idx: np.array([[recs[int(i)].crop_uid, recs[int(j)].crop_uid] for i, j in idx], dtype=str).reshape(-1, 2)
        for kind in ("neg", "pos"):
            f = acc[f"fail_{kind}"]
            ii, jj = f[:, 1].astype(int), f[:, 2].astype(int)
            acc[f"fail_{kind}"] = f[:, :1]
            acc[f"fail_{kind}_uid"] = uid(zip(ii, jj))
            acc[f"fail_{kind}_trk"] = np.array([[recs[i].tracklet_id, recs[j].tracklet_id] for i, j in zip(ii, jj)], dtype=str).reshape(-1, 2)
            acc[f"fail_{kind}_frame"] = np.array([[recs[i].frame, recs[j].frame] for i, j in zip(ii, jj)], np.int64).reshape(-1, 2)
            acc[f"fail_{kind}_dpos"] = np.array([np.nan if not (meta.pos_ok[i] and meta.pos_ok[j]) else
                                                 float(np.linalg.norm(meta.pos[i] - meta.pos[j])) for i, j in zip(ii, jj)])
        lo = np.stack([acc["obj_low_i"], acc["obj_low_j"]], 1)
        acc["obj_low_uid"] = np.array([[recs[i].crop_uid, recs[j].crop_uid] if i >= 0 else ["", ""] for i, j in lo], dtype=str).reshape(-1, 2)
        acc["tracklet_ids"] = _strings(meta.tracklet_ids)
        info = infos[vid]
        acc["meta"] = np.asarray(json.dumps({"video_id": vid, "site": info.site, "pov": info.pov, "set": set_name,
                                             "keypoints_present": bool(meta.kp is not None)}))
        path = Path(acc_dir) / set_name / f"{vid}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **acc)
        tmp.replace(path)
        paths.append(path)
        if verbose:
            print(f"[acc {set_name}] {k + 1}/{len(split[set_name])} {vid}: {meta.n} crops, {int(acc['hist'].sum())} pairs", file=sys.stderr)
    return paths


def run(template_dir, split_path, data_root, bins_path, out_base, device=None, verbose=True):
    """Full evaluation of one template folder. Returns the results folder."""
    tdir = Path(template_dir)
    manifest = read_json(tdir / "manifest.json")
    split = read_json(split_path)
    if manifest["split_sha256"] != sha256_file(split_path):
        raise ValueError(f"templates were extracted with a different split file than {split_path}")
    bins = Bins.load(bins_path)
    ids = sorted(set(split["validation"]) | set(split["test"]))
    records, infos = load_dataset_with_info(data_root, ids, verbose=verbose)
    by_video = group_by_video(records)
    check_fingerprints(split, by_video, SETS)

    out = Path(out_base) / f"{manifest['model_name']}__{manifest['preproc_mode']}" / split["version"]
    out.mkdir(parents=True, exist_ok=True)
    thresholds = compute_thresholds(tdir, split, by_video, bins, device)
    thresholds.update({"split_version": split["version"], "split_sha256": manifest["split_sha256"], "bins_version": bins.version,
                       "bins_sha256": bins.sha256, "model_name": manifest["model_name"], "preproc_mode": manifest["preproc_mode"]})
    write_json_atomic(out / "thresholds.json", thresholds)
    if verbose:
        print(f"thresholds {dict(zip(thresholds['names'], thresholds['thresholds']))} from validation", file=sys.stderr)
    thr = thresholds["thresholds"]
    acc_dir = out / "acc"
    val_paths = accumulate_set(tdir, split, "validation", by_video, infos, bins, thr, acc_dir, device, verbose)
    test_paths = accumulate_set(tdir, split, "test", by_video, infos, bins, thr, acc_dir, device, verbose)
    write_json_atomic(out / "run.json", {"template_dir": str(tdir), "manifest": manifest, "split_file": str(split_path),
                                         "data_root": str(data_root), "bins_file": str(bins_path), "bins_sha256": bins.sha256})
    write_results(out, bins, thresholds, val_paths, test_paths, data_root)
    return out


def write_results(out, bins: Bins, thresholds, val_paths, test_paths, data_root=None, verbose=True):
    """Everything except the encoding: reads only accumulators, thresholds and bins (reproducible from saved files)."""
    out = Path(out)
    agg = Aggregation(test_paths, bins, thresholds)
    val = Aggregation(val_paths, bins, thresholds) if val_paths else None
    summary = {"model_name": thresholds.get("model_name"), "preproc_mode": thresholds.get("preproc_mode"),
               "split_version": thresholds.get("split_version"), "bins_version": bins.version,
               "n_videos": agg.V, "thresholds": dict(zip(thresholds["names"], thresholds["thresholds"])),
               "headline": agg.headline(), "bootstrap": {"resamples": bins.boot_resamples, "level": bins.boot_level,
                                                         "unit": "videos", "seed": bins.boot_seed}}
    if val is not None:
        vh = val.table((), ci_=False).iloc[0]
        summary["validation"] = {k: float(vh[k]) for k in ("pooled_auc", "pooled_eer", "pooled_tar_t2", "pooled_tar_t3",
                                                           "pooled_far_t2", "pooled_far_t3", "balanced_tar_t2", "balanced_tar_t3")}
    write_json_atomic(out / "summary.json", summary)
    for axis, df in agg.axis_tables().items():
        df.to_csv(out / f"bins_{axis}.csv", index=False)
    if any(a["meta"].get("keypoints_present") for a in agg.accs):
        agg.kpmin().to_csv(out / "bins_min_visible_keypoints.csv", index=False)
    for (a, b), df in agg.heatmap_tables().items():
        df.to_csv(out / f"heatmap_{a}_x_{b}.csv", index=False)
    agg.cells_table().to_csv(out / "cells.csv", index=False)
    agg.per_video().to_csv(out / "per_video.csv", index=False)
    agg.per_object().to_csv(out / "per_object.csv", index=False)
    fdir = out / "failures"
    fdir.mkdir(exist_ok=True)
    for kind, name in (("neg", "negatives_highest_similarity"), ("pos", "positives_lowest_similarity_dpos_lt_1m")):
        df = collect(agg.accs, kind)
        df.to_csv(fdir / f"{name}.csv", index=False)
        if data_root is not None:
            contact_sheets(df, data_root, fdir / name)
    if verbose:
        h = summary["headline"]["object_balanced"]
        print(f"object-balanced: AUC {h['auc']['value']:.4f}  TAR@t(1e-3) {h['tar_t3']['value']:.4f}  FAR@t(1e-3) {h['far_t3']['value']:.5f}",
              file=sys.stderr)
    return summary
