"""Orchestration of the verification evaluation: validation thresholds -> per-video accumulators -> results files."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from reid_data import group_by_video, load_dataset_with_info
from . import metrics as M
from .accumulate import VideoMeta, accumulate_video, fine_histograms
from .aggregate import AXES, Aggregation, HEATMAPS, load_summaries
from .bins import N_FINE, Bins
from .common import read_json, sha256_file, write_json_atomic
from .failures import collect, contact_sheets
from .matches import object_sheets
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
        "validation_auc": float(M.auc(pos, neg)), "strict": bins.strict_name,
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


def accumulate_set(tdir, split, set_name, by_video, infos, bins, thresholds, acc_dir, device=None, verbose=True, plain=False,
                   matches_dir=None, match_topk=10):
    paths = []
    for k, vid in enumerate(split[set_name]):
        recs = by_video[vid]
        emb, meta = load_video(tdir, set_name, vid, recs)
        if matches_dir is not None:  # one image per object with its top positive / negative matches (needs the raw embeddings)
            j = int(np.argmin([abs(np.log10(t) - np.log10(0.01)) for t in bins.far_targets]))  # flags use the threshold nearest FAR 1 %
            object_sheets(emb, recs, Path(matches_dir) / _slug(vid, set()), thresholds[j], f"t(FAR {bins.thr_names[j].replace('pct', '%')})",
                          match_topk, device=device)
        if plain:  # no pose / occlusion / keypoint information at all: one global result per video
            meta = meta.neutral()
        acc = accumulate_video(emb, meta, bins, thresholds, device, fail_pos_all=plain)
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


def run(template_dir, split_path, data_root, bins_path, out_base, device=None, verbose=True, plain=False, per_subset=True,
        match_sheets=None, match_topk=10):
    """Full evaluation of one template folder. `plain=True`: no difficulty criteria (pose/occlusion/keypoints), global metrics only.
    `match_sheets` (default: on for the full variant, off for plain): per-object top-k positive/negative match images of the test
    videos, in <results>/matches/<video>/. Returns the results folder."""
    match_sheets = (not plain) if match_sheets is None else match_sheets
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

    kind = "plain" if plain else "full"
    out = Path(out_base) / f"{manifest['model_name']}__{manifest['preproc_mode']}" / (split["version"] + ("__plain" if plain else ""))
    out.mkdir(parents=True, exist_ok=True)
    thresholds = compute_thresholds(tdir, split, by_video, bins, device)
    thresholds.update({"split_version": split["version"], "split_sha256": manifest["split_sha256"], "bins_version": bins.version,
                       "bins_sha256": bins.sha256, "model_name": manifest["model_name"], "preproc_mode": manifest["preproc_mode"],
                       "kind": kind})
    write_json_atomic(out / "thresholds.json", thresholds)
    if verbose:
        print(f"[{kind}] thresholds {dict(zip(thresholds['names'], thresholds['thresholds']))} from validation", file=sys.stderr)
    thr = thresholds["thresholds"]
    acc_dir = out / "acc"
    val_paths = accumulate_set(tdir, split, "validation", by_video, infos, bins, thr, acc_dir, device, verbose, plain)
    test_paths = accumulate_set(tdir, split, "test", by_video, infos, bins, thr, acc_dir, device, verbose, plain,
                                matches_dir=(out / "matches") if match_sheets else None, match_topk=match_topk)
    write_json_atomic(out / "run.json", {"template_dir": str(tdir), "manifest": manifest, "split_file": str(split_path), "kind": kind,
                                         "data_root": str(data_root), "bins_file": str(bins_path), "bins_sha256": bins.sha256})
    write_results(out, bins, thresholds, val_paths, test_paths, data_root, kind=kind, per_subset=per_subset)
    return out


def _slug(name, used):
    """Folder-safe unique name."""
    base = re.sub(r"[^\w.\-]+", "_", str(name)).strip("_") or "unnamed"
    slug, k = base, 2
    while slug in used:
        slug, k = f"{base}_{k}", k + 1
    used.add(slug)
    return slug


def write_subset(agg: Aggregation, out, bins: Bins, thresholds, kind, title, extra=None):
    """Tables for one group of videos (all test videos, one site, or one video). The report is built from these files."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    names = bins.thr_names
    summary = {"title": title, "short_title": extra.get("short_title", "") if extra else "", "kind": kind,
               "model_name": thresholds.get("model_name"), "preproc_mode": thresholds.get("preproc_mode"),
               "split_version": thresholds.get("split_version"), "bins_version": bins.version, "n_videos": agg.V,
               "videos": [x.video_id for x in agg.s] if agg.V <= 50 else None, "far_names": names, "strict_name": bins.strict_name,
               "heat_names": bins.heat_names, "thresholds": dict(zip(names, thresholds["thresholds"])), "has_ci": agg.use_ci,
               "headline": agg.headline(), "bootstrap": {"resamples": bins.boot_resamples, "level": bins.boot_level,
                                                         "unit": "videos", "seed": bins.boot_seed}}
    summary.update(extra or {})
    write_json_atomic(out / "summary.json", summary)
    agg.roc().to_csv(out / "roc.csv", index=False)
    if kind == "full":
        for axis, df in agg.axis_tables().items():
            df.to_csv(out / f"bins_{axis}.csv", index=False)
        if agg.has_keypoints():
            agg.kpmin().to_csv(out / "bins_min_visible_keypoints.csv", index=False)
        for (a, b), df in agg.heatmap_tables().items():
            df.to_csv(out / f"heatmap_{a}_x_{b}.csv", index=False)
    return summary


def write_results(out, bins: Bins, thresholds, val_paths, test_paths, data_root=None, kind="full", per_subset=True, verbose=True):
    """Everything except the encoding: reads only accumulators, thresholds and bins (reproducible from saved files)."""
    from .report import build_report_from_dir
    out = Path(out)
    test_s, pooled_cells = load_summaries(test_paths, bins, thresholds)
    val_s, _ = load_summaries(val_paths, bins, thresholds) if val_paths else ([], None)
    agg = Aggregation(test_s, bins, thresholds, pooled_cells)
    names = bins.thr_names
    model = f"{thresholds.get('model_name')} / {thresholds.get('preproc_mode')}"
    summary = write_subset(agg, out, bins, thresholds, kind, f"Verification report ({kind}): {model}",
                           {"short_title": ""})
    if val_s:
        vh = Aggregation(val_s, bins, thresholds).table(()).iloc[0]
        summary["validation"] = {"pooled_auc": float(vh["pooled_auc"]), "pooled_eer": float(vh["pooled_eer"]),
                                 **{f"pooled_{b}_at_{n}": float(vh[f"pooled_{b}_at_{n}"]) for n in names for b in ("tar", "far")}}
        write_json_atomic(out / "summary.json", summary)
    if kind == "full":
        agg.cells_table().to_csv(out / "cells.csv", index=False)
    pv, ps = agg.per_video(), agg.per_site()
    agg.per_object().to_csv(out / "per_object.csv", index=False)

    # ---- the same figures/tables for every site and every video
    if per_subset:
        used = set()
        site_dirs = {site: _slug(site, used) for site in ps["site"]}
        ps["report"] = [f"per_site/{site_dirs[s]}/report.md" for s in ps["site"]]
        used = set()
        vid_dirs = {v: _slug(v, used) for v in pv["video_id"]}
        pv["report"] = [f"per_video/{vid_dirs[v]}/report.md" for v in pv["video_id"]]
    pv.to_csv(out / "per_video.csv", index=False)
    ps.to_csv(out / "per_site.csv", index=False)
    if per_subset:
        for site, d in site_dirs.items():
            sub = [x for x in test_s if x.site == site]
            sd = out / "per_site" / d
            write_subset(Aggregation(sub, bins, thresholds), sd, bins, thresholds, kind, f"Site {site}: {model} ({kind})",
                         {"site": site, "short_title": f"site {site}"})
            build_report_from_dir(sd, data_root)
        for x in test_s:
            vd = out / "per_video" / vid_dirs[x.video_id]
            mdir = out / "matches" / _slug(x.video_id, set())
            write_subset(Aggregation([x], bins, thresholds), vd, bins, thresholds, kind, f"Video {x.video_id} (site {x.site}): {model} ({kind})",
                         {"video_id": x.video_id, "site": x.site, "short_title": x.video_id,
                          "matches": f"../../matches/{mdir.name}/index.md" if (mdir / "index.md").exists() else None})
            build_report_from_dir(vd, data_root)
            if verbose:
                print(f"[report] {x.video_id}", file=sys.stderr)

    fdir = out / "failures"
    fdir.mkdir(exist_ok=True)
    fnames = (("neg", "negatives_highest_similarity"),
              ("pos", "positives_lowest_similarity" if kind == "plain" else "positives_lowest_similarity_dpos_lt_1m"))
    for k, name in fnames:
        df = collect(test_s, k)
        df.to_csv(fdir / f"{name}.csv", index=False)
        if data_root is not None:
            contact_sheets(df, data_root, fdir / name)
    if verbose:
        h = summary["headline"]["object_balanced"]
        print(f"[{kind}] object-balanced: AUC {h['auc']['value']:.4f}  " +
              "  ".join(f"TAR@{n} {h[f'tar_at_{n}']['value']:.4f}" for n in sorted(names, key=lambda x: float(x[:-3]))), file=sys.stderr)
    build_report_from_dir(out, data_root)
    return summary
