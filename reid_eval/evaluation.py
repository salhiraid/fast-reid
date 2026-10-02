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
from .accumulate import VideoMeta, accumulate_video, assign_folds, fine_histograms, fine_histograms_subset
from .aggregate import AXES, Aggregation, HEATMAPS, load_summaries
from .bins import N_FINE, Bins
from .common import read_json, sha256_file, write_json_atomic
from .failures import collect, contact_sheets
from .matches import object_sheets
from .site_matches import site_visuals
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
    thr = [M.threshold_for_far(neg, t) for t in bins.far_targets] + list(bins.fixed_thresholds)   # FAR thresholds, then fixed ones
    if neg.sum() * min(bins.far_targets) < 10:
        print(f"WARNING: only {int(neg.sum())} validation negatives for FAR target {min(bins.far_targets):g}", file=sys.stderr)
    return {
        "source": "validation", "far_targets": bins.far_targets, "fixed_thresholds": bins.fixed_thresholds,
        "names": bins.op_names, "thresholds": thr,
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


def _finish_acc(acc, recs, meta, info, vid, set_name):
    """Add the human-readable parts (crop uids, tracklet ids, site...) to a raw accumulator."""
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
    acc["meta"] = np.asarray(json.dumps({"video_id": vid, "site": info.site, "pov": info.pov, "set": set_name,
                                         "keypoints_present": bool(meta.kp is not None)}))
    return acc


def _save_acc(acc, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **acc)
    tmp.replace(path)
    return path


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
        acc = _finish_acc(acc, recs, meta, infos[vid], vid, set_name)
        paths.append(_save_acc(acc, Path(acc_dir) / set_name / f"{vid}.npz"))
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
    _refresh_variant_comparison(out.parent, split["version"])
    return out


def _refresh_variant_comparison(model_dir, split_version):
    """Rebuild <split>__threshold_comparison.* from every variant folder that exists NOW: called at the end of every evaluation function, so
    the table is complete whatever the order in which the variants were run (separately or by run_full_eval)."""
    from .report import write_variant_comparison
    write_variant_comparison(model_dir, split_version)


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
    names = bins.op_names
    summary = {"title": title, "short_title": extra.get("short_title", "") if extra else "", "kind": kind,
               "model_name": thresholds.get("model_name"), "preproc_mode": thresholds.get("preproc_mode"),
               "split_version": thresholds.get("split_version"), "bins_version": bins.version, "n_videos": agg.V,
               "videos": [x.video_id for x in agg.s] if agg.V <= 50 else None, "far_names": bins.thr_names, "fixed_names": bins.fixed_names, "strict_name": bins.strict_name,
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


def _spread(df):
    """Mean, std, median, min, max over the rows (videos or sites) of every numeric metric column (AUC, EER, TAR/FAR/accuracy...)."""
    cols = [c for c in df.columns if c not in ("n_videos", "n_objects", "n_crops", "n_pos", "n_neg", "n_object_pairs", "confused_object_pairs")
            and df[c].dtype.kind == "f" and not c.endswith(("_lo", "_hi"))]
    st = df[cols].agg(["count", "mean", "std", "median", "min", "max"]).T
    st.index.name = "metric"
    return st


def write_results(out, bins: Bins, thresholds, val_paths, test_paths, data_root=None, kind="full", per_subset=True, verbose=True,
                  check_thresholds=True, extra=None, title=None, subsets=("site", "video")):
    """Everything except the encoding: reads only accumulators, thresholds and bins (reproducible from saved files)."""
    from .report import build_report_from_dir
    out = Path(out)
    test_s, pooled_cells = load_summaries(test_paths, bins, thresholds, check_thresholds)
    val_s, _ = load_summaries(val_paths, bins, thresholds) if val_paths else ([], None)
    agg = Aggregation(test_s, bins, thresholds, pooled_cells)
    names = bins.op_names
    model = f"{thresholds.get('model_name')} / {thresholds.get('preproc_mode')}"
    summary = write_subset(agg, out, bins, thresholds, kind, title or f"Verification report ({kind}): {model}",
                           {"short_title": "", **(extra or {})})
    if val_s:
        vh = Aggregation(val_s, bins, thresholds).table(()).iloc[0]
        summary["validation"] = {"pooled_auc": float(vh["pooled_auc"]), "pooled_eer": float(vh["pooled_eer"]),
                                 **{f"pooled_{b}_at_{n}": float(vh[f"pooled_{b}_at_{n}"]) for n in names for b in ("tar", "far", "acc", "bacc", "prec")}}
        write_json_atomic(out / "summary.json", summary)
    if kind == "full":
        agg.cells_table().to_csv(out / "cells.csv", index=False)
    pv, ps = agg.per_video(), agg.per_site()
    agg.per_object().to_csv(out / "per_object.csv", index=False)

    # ---- the same figures/tables for every site and every video
    do_site, do_video = per_subset and "site" in subsets, per_subset and "video" in subsets
    if do_site:
        used = set()
        site_dirs = {site: _slug(site, used) for site in ps["site"]}
        ps["report"] = [f"per_site/{site_dirs[s]}/report.md" for s in ps["site"]]
    if do_video:
        used = set()
        vid_dirs = {v: _slug(v, used) for v in pv["video_id"]}
        pv["report"] = [f"per_video/{vid_dirs[v]}/report.md" for v in pv["video_id"]]
    pv.to_csv(out / "per_video.csv", index=False)
    ps.to_csv(out / "per_site.csv", index=False)
    _spread(pv).to_csv(out / "per_video_stats.csv")
    if len(ps) >= 2:
        _spread(ps).to_csv(out / "per_site_stats.csv")
    if do_site:
        for site, d in site_dirs.items():
            sub = [x for x in test_s if x.site == site]
            sd = out / "per_site" / d
            sm = out / "site_matches" / _slug(site, set()) / "index.md"
            write_subset(Aggregation(sub, bins, thresholds), sd, bins, thresholds, kind, f"Site {site}: {model} ({kind})",
                         {"site": site, "short_title": f"site {site}", **(extra or {}),
                          "site_matches": f"../../site_matches/{sm.parent.name}/index.md" if sm.exists() else None})
            build_report_from_dir(sd, data_root)
    if do_video:
        for x in test_s:
            vd = out / "per_video" / vid_dirs[x.video_id]
            mdir = out / "matches" / _slug(x.video_id, set())
            write_subset(Aggregation([x], bins, thresholds), vd, bins, thresholds, kind, f"Video {x.video_id} (site {x.site}): {model} ({kind})",
                         {"video_id": x.video_id, "site": x.site, "short_title": x.video_id, **(extra or {}),
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
              "  ".join(f"TAR@{n} {h[f'tar_at_{n}']['value']:.4f}" for n in bins.thr_names) +
              "".join(f"  acc@{n} {h[f'acc_at_{n}']['value']:.4f}" for n in bins.fixed_names), file=sys.stderr)
    build_report_from_dir(out, data_root)
    return summary


# ----------------------------------------------------------------------------- per-video / per-site thresholds
THR_PROTOCOLS = {
    "global-oracle": ("ORACLE GLOBAL threshold (the same for every video): set on the negative pairs of ALL test videos pooled, including the "
                      "evaluated ones, then used unchanged on every video (matching within each video). The static counterpart of the per-video / "
                      "per-site thresholds; OPTIMISTIC because it is tuned on the data it is measured on. The honest global threshold (set on the "
                      "validation videos) is the `global` evaluation."),
    "video": ("HELD-OUT per-video thresholds: the objects of each video are split in two folds; each fold is evaluated with the "
              "threshold set on the negative pairs of the OTHER fold (pairs across the two folds are not evaluated). "
              "Videos with fewer than 4 objects cannot be split and are skipped."),
    "site": ("HELD-OUT per-site thresholds (leave-one-video-out): each test video is evaluated with the threshold set on the negative "
             "pairs of the OTHER test videos of its site. A site with a single test video cannot be evaluated and is skipped."),
    "video-oracle": ("ORACLE per-video thresholds: each video's threshold is set on that video's own negative pairs, so FAR is forced "
                     "to the target and the numbers are OPTIMISTIC (tuned on the data they are measured on). Reference only."),
    "site-oracle": ("ORACLE per-site thresholds: each site's threshold is set on all negative pairs of the site's test videos, "
                    "including the evaluated video. OPTIMISTIC (tuned on the data it is measured on). Reference only."),
}


def _thr_row(neg_hist, bins: Bins):
    """FAR thresholds from a negative histogram, then the fixed thresholds."""
    return [M.threshold_for_far(neg_hist, t) for t in bins.far_targets] + list(bins.fixed_thresholds)


def _write_site_visuals(tdir, by_video, site_videos, own, bins, out_dir, n_queries, topk, seed, device, verbose):
    """For every site: its objects against the gallery of all crops of all the site's test videos (see site_matches.py)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    j = int(np.argmin([abs(np.log10(t) - np.log10(0.01)) for t in bins.far_targets]))      # the FAR target nearest 1 %
    label = f"t(FAR {bins.thr_names[j].replace('pct', '%')})"
    rows = []
    for site, vlist in sorted(site_videos.items()):
        emb_by = {v: load_video(tdir, "test", v, by_video[v])[0] for v in vlist}
        thr = M.threshold_for_far(sum(own[v] for v in vlist), bins.far_targets[j])
        rows.append(site_visuals(site, vlist, emb_by, {v: by_video[v] for v in vlist}, thr, label, out_dir / _slug(site, set()),
                                 n_queries, topk, seed, device=device))
        if verbose:
            print(f"[site matches] {site}: {rows[-1]['objects']} objects, {rows[-1]['candidates_above_threshold']} cross-video candidates "
                  f"above the site threshold", file=sys.stderr)
    pd.DataFrame(rows).to_csv(out_dir / "index.csv", index=False)
    md = ["# Site-level matching", "",
          "Each site's objects against the gallery of all crops of all the site's test videos. Same-video matches are labelled; "
          "matches between videos have no ground truth (identity unknown). One folder per site.", "",
          "| site | videos | objects | crops | sampled queries | cross-video candidates above the site threshold |", "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| [{r['site']}]({_slug(r['site'], set())}/index.md) | {r['videos']} | {r['objects']} | {r['crops']} | {r['queries']} | "
                  f"{r['candidates_above_threshold']} |")
    (out_dir / "index.md").write_text("\n".join(md), encoding="utf-8")
    return rows


def run_threshold_modes(template_dir, split_path, data_root, bins_path, out_base, modes, device=None, verbose=True, per_subset=True,
                        seed=0, kind="full", site_queries=10, match_topk=10):
    """Evaluations with a threshold per video and/or per site (`modes`: video, site, video-oracle, site-oracle).

    Global-threshold evaluations (full / plain) use thresholds from the VALIDATION videos. These use the test videos' own
    data to calibrate, so the leak-free protocols hold the calibrated data out of the measurement (see THR_PROTOCOLS); the
    '-oracle' modes do not and are labelled as optimistic. The validation videos are not used. Plain kind (no pose criteria).
    Returns {mode: results folder}.
    """
    for m in modes:
        if m not in THR_PROTOCOLS:
            raise ValueError(f"unknown threshold mode {m!r}; one of {sorted(THR_PROTOCOLS)}")
    tdir = Path(template_dir)
    manifest = read_json(tdir / "manifest.json")
    split = read_json(split_path)
    if manifest["split_sha256"] != sha256_file(split_path):
        raise ValueError(f"templates were extracted with a different split file than {split_path}")
    bins = Bins.load(bins_path)
    test_ids = list(split["test"])
    records, infos = load_dataset_with_info(data_root, sorted(test_ids), verbose=verbose)
    by_video = group_by_video(records)
    check_fingerprints(split, by_video, ("test",))
    site_of = {v: (infos[v].site or "unknown") for v in test_ids}

    # ---- pass 1: negative histograms (all pairs of a video; and of each fold for the held-out per-video mode)
    own, fold_neg, folds = {}, {}, {}
    for vid in test_ids:
        emb, meta = load_video(tdir, "test", vid, by_video[vid])
        meta = meta.neutral()
        own[vid] = fine_histograms(emb, meta, device)[1]
        T = len(meta.tracklet_ids)
        if "video" in modes and T >= 4:
            folds[vid] = assign_folds(vid, T, seed)
            fold_neg[vid] = [fine_histograms_subset(emb, meta, folds[vid] == k, device)[1] for k in (0, 1)]
    site_videos = {}
    for v in test_ids:
        site_videos.setdefault(site_of[v], []).append(v)

    out_dirs = {}
    for mode in modes:
        units, rows = {}, []   # units: vid -> (threshold table (F, Q), folds or None); rows: per-unit report
        for vid in test_ids:
            skip, table, n_cal = None, None, []
            if mode == "video":
                if vid not in folds:
                    skip = "fewer than 4 objects: cannot split into two folds"
                else:   # fold f is evaluated with the threshold from the OTHER fold's negatives
                    hists = [fold_neg[vid][1 - f] for f in (0, 1)]
                    if min(h.sum() for h in hists) == 0:
                        skip = "a fold has no negative pairs"
                    else:
                        table, n_cal = [_thr_row(h, bins) for h in hists], [int(h.sum()) for h in hists]
            elif mode == "video-oracle":
                table, n_cal = [_thr_row(own[vid], bins)], [int(own[vid].sum())]
            elif mode == "global-oracle":
                allneg = sum(own.values())
                table, n_cal = [_thr_row(allneg, bins)], [int(allneg.sum())]
            else:
                others = [u for u in site_videos[site_of[vid]] if mode == "site-oracle" or u != vid]
                if not others:
                    skip = "only test video of its site: no other video to calibrate on"
                else:
                    h = sum(own[u] for u in others)
                    table, n_cal = [_thr_row(h, bins)], [int(h.sum())]
            if skip:
                rows.append({"unit": vid, "site": site_of[vid], "fold": "", "status": f"skipped: {skip}", "n_calibration_negatives": 0,
                             **{f"thr_{n}": np.nan for n in bins.op_names}})
                continue
            units[vid] = (np.asarray(table), folds[vid] if mode == "video" else None)
            for f, (row, n) in enumerate(zip(table, n_cal)):
                rows.append({"unit": vid, "site": site_of[vid], "fold": f if mode == "video" else "", "status": "used",
                             "n_calibration_negatives": n, **{f"thr_{nm}": v for nm, v in zip(bins.op_names, row)}})
        used = [r for r in rows if r["status"] == "used"]
        if not units:   # e.g. no site has two test videos: skip this variant, do not abort the other ones
            print(f"WARNING: [thr-{mode}] no test video can be evaluated with this protocol "
                  f"({rows[0]['status'] if rows else 'no videos'}); variant skipped", file=sys.stderr)
            continue
        out = Path(out_base) / f"{manifest['model_name']}__{manifest['preproc_mode']}" / (f"{split['version']}__thr-{mode}" + ("__plain" if kind == "plain" else ""))
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out / "thresholds_per_unit.csv", index=False)
        med = {nm: float(np.median([r[f"thr_{nm}"] for r in used])) for nm in bins.op_names}
        n_skip = len(rows) - len(used)
        thresholds = {"source": f"test videos (held out)" if "oracle" not in mode else "test videos (ORACLE)", "mode": mode,
                      "protocol": THR_PROTOCOLS[mode], "far_targets": bins.far_targets, "fixed_thresholds": bins.fixed_thresholds,
                      "names": bins.op_names, "thresholds": [med[nm] for nm in bins.op_names],
                      "thresholds_note": "median over units; every unit has its own thresholds (thresholds_per_unit.csv)",
                      "n_units_used": len({r['unit'] for r in used}), "units_skipped": [r for r in rows if r["status"] != "used"],
                      "n_calibration_negatives_median": float(np.median([r["n_calibration_negatives"] for r in used])),
                      "split_version": split["version"], "split_sha256": manifest["split_sha256"], "bins_version": bins.version,
                      "bins_sha256": bins.sha256, "model_name": manifest["model_name"], "preproc_mode": manifest["preproc_mode"],
                      "kind": kind}
        write_json_atomic(out / "thresholds.json", thresholds)
        # ---- pass 2: count at each unit's own thresholds
        test_paths = []
        for k, (vid, (table, fold)) in enumerate(units.items()):
            recs = by_video[vid]
            emb, meta = load_video(tdir, "test", vid, recs)
            if kind == "plain":
                meta = meta.neutral()
            if fold is not None:
                meta = meta.with_folds(fold)
            acc = accumulate_video(emb, meta, bins, table, device, fail_pos_all=(kind == "plain"))
            acc = _finish_acc(acc, recs, meta, infos[vid], vid, "test")
            test_paths.append(_save_acc(acc, out / "acc" / "test" / f"{vid}.npz"))
            if verbose:
                print(f"[acc thr-{mode}] {k + 1}/{len(units)} {vid}", file=sys.stderr)
        site_summary = None
        if mode == "site" and site_queries:      # site-level matching visuals (sampled queries, object matrix, cross-video candidates)
            site_summary = _write_site_visuals(tdir, by_video, site_videos, own, bins, out / "site_matches", site_queries, match_topk, seed,
                                               device, verbose)
        write_json_atomic(out / "run.json", {"template_dir": str(tdir), "manifest": manifest, "split_file": str(split_path), "kind": kind,
                                             "threshold_mode": mode, "data_root": str(data_root), "bins_file": str(bins_path),
                                             "bins_sha256": bins.sha256})
        extra = {"threshold_protocol": THR_PROTOCOLS[mode], "threshold_mode": mode, "threshold_units_used": thresholds["n_units_used"],
                 "threshold_units_skipped": thresholds["units_skipped"], "oracle": "oracle" in mode,
                 "n_calibration_negatives_median": thresholds["n_calibration_negatives_median"],
                 "site_matches": "site_matches/index.md" if site_summary is not None else None}
        model = f"{manifest['model_name']} / {manifest['preproc_mode']}"
        write_results(out, bins, thresholds, [], test_paths, data_root, kind=kind, per_subset=per_subset, verbose=verbose,
                      check_thresholds=False, extra=extra,
                      title=f"Verification report, {'global threshold on all test videos' if mode == 'global-oracle' else 'threshold per ' + ('video' if 'video' in mode else 'site')}"
                            f"{' (ORACLE, optimistic)' if 'oracle' in mode else ' (held out)'}: {model}")
        out_dirs[mode] = out
    _refresh_variant_comparison(Path(out_base) / f"{manifest['model_name']}__{manifest['preproc_mode']}", split["version"])
    return out_dirs


# ----------------------------------------------------------------------------- site matching (gallery = the whole site)
GALLERY_PROTOCOLS = {
    "site-gallery": ("SITE MATCHING with a HELD-OUT threshold per site: every crop of every test video of a site is matched with every other "
                     "crop of the site (the gallery is the whole site, so an object is also matched with the objects of the site's OTHER videos). "
                     "Positives = same tracklet. Every other pair is a negative, INCLUDING cross-video pairs, whose identity is unknown: this assumes "
                     "that no vehicle reappears in two videos of a site (if one does, FAR is overstated). The site's objects are split in two folds; "
                     "each fold is evaluated with the threshold set on the negative pairs of the OTHER fold (pairs across folds are not evaluated). "
                     "Sites with fewer than 4 objects are skipped. The units of this evaluation are SITES."),
    "site-gallery-oracle": ("SITE MATCHING (gallery = the whole site, cross-video pairs assumed negative) with an ORACLE threshold per site, set on "
                            "all negative pairs of the site including the evaluated ones: FAR is forced to the target and the numbers are OPTIMISTIC. "
                            "Reference only. The units of this evaluation are SITES."),
}
THR_PROTOCOLS.update(GALLERY_PROTOCOLS)


def run_site_gallery(template_dir, split_path, data_root, bins_path, out_base, modes=("site-gallery",), device=None, verbose=True,
                     per_subset=True, seed=0):
    """Evaluation where, for each site, ALL crops of ALL the site's test videos form one gallery (see GALLERY_PROTOCOLS).

    Plain kind (no pose criteria: positions / azimuths of different videos are not comparable). The site is handled as one big
    pseudo-video, so the whole accumulation machinery is reused. Returns {mode: results folder}.
    """
    from types import SimpleNamespace
    for m in modes:
        if m not in GALLERY_PROTOCOLS:
            raise ValueError(f"unknown site-gallery mode {m!r}; one of {sorted(GALLERY_PROTOCOLS)}")
    tdir = Path(template_dir)
    manifest = read_json(tdir / "manifest.json")
    split = read_json(split_path)
    if manifest["split_sha256"] != sha256_file(split_path):
        raise ValueError(f"templates were extracted with a different split file than {split_path}")
    bins = Bins.load(bins_path)
    test_ids = list(split["test"])
    records, infos = load_dataset_with_info(data_root, sorted(test_ids), verbose=verbose)
    by_video = group_by_video(records)
    check_fingerprints(split, by_video, ("test",))
    sites = {}
    for v in test_ids:
        sites.setdefault(infos[v].site or "unknown", []).append(v)

    def load_site(vids):
        vids = sorted(vids)
        embs = [load_video(tdir, "test", v, by_video[v])[0] for v in vids]
        recs = [r for v in vids for r in by_video[v]]
        return np.concatenate(embs), recs, VideoMeta.from_records(recs).neutral()

    # ---- pass 1: negative histograms of each site (all pairs; and of each fold of its objects)
    own, folds, fold_neg = {}, {}, {}
    for site, vids in sorted(sites.items()):
        emb, recs, meta = load_site(vids)
        own[site] = fine_histograms(emb, meta, device)[1]
        T = len(meta.tracklet_ids)
        if "site-gallery" in modes and T >= 4:
            folds[site] = assign_folds(f"site::{site}", T, seed)
            fold_neg[site] = [fine_histograms_subset(emb, meta, folds[site] == k, device)[1] for k in (0, 1)]
        if verbose:
            print(f"[site gallery] {site}: {len(vids)} videos, {T} objects, {len(recs)} crops, {int(own[site].sum()):,} negative pairs",
                  file=sys.stderr)

    out_dirs = {}
    for mode in modes:
        units, rows = {}, []
        for site in sorted(sites):
            skip = table = None
            n_cal = []
            if mode == "site-gallery":
                if site not in folds:
                    skip = "fewer than 4 objects: cannot split into two folds"
                else:
                    hists = [fold_neg[site][1 - f] for f in (0, 1)]
                    if min(h.sum() for h in hists) == 0:
                        skip = "a fold has no negative pairs"
                    else:
                        table, n_cal = [_thr_row(h, bins) for h in hists], [int(h.sum()) for h in hists]
            else:
                if own[site].sum() == 0:
                    skip = "no negative pairs (a single object)"
                else:
                    table, n_cal = [_thr_row(own[site], bins)], [int(own[site].sum())]
            if skip:
                rows.append({"unit": site, "site": site, "fold": "", "status": f"skipped: {skip}", "n_calibration_negatives": 0,
                             **{f"thr_{n}": np.nan for n in bins.op_names}})
                continue
            units[site] = (np.asarray(table), folds[site] if mode == "site-gallery" else None)
            for f, (row, n) in enumerate(zip(table, n_cal)):
                rows.append({"unit": site, "site": site, "fold": f if mode == "site-gallery" else "", "status": "used",
                             "n_calibration_negatives": n, **{f"thr_{nm}": v for nm, v in zip(bins.op_names, row)}})
        used = [r for r in rows if r["status"] == "used"]
        if not units:
            print(f"WARNING: [{mode}] no site can be evaluated with this protocol; variant skipped", file=sys.stderr)
            continue
        out = Path(out_base) / f"{manifest['model_name']}__{manifest['preproc_mode']}" / f"{split['version']}__{mode}"
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out / "thresholds_per_unit.csv", index=False)
        med = {nm: float(np.median([r[f"thr_{nm}"] for r in used])) for nm in bins.op_names}
        thresholds = {"source": "test sites (held out)" if mode == "site-gallery" else "test sites (ORACLE)", "mode": mode,
                      "protocol": GALLERY_PROTOCOLS[mode], "far_targets": bins.far_targets, "fixed_thresholds": bins.fixed_thresholds,
                      "names": bins.op_names, "thresholds": [med[nm] for nm in bins.op_names],
                      "thresholds_note": "median over sites; every site has its own thresholds (thresholds_per_unit.csv)",
                      "n_units_used": len({r["unit"] for r in used}), "units_skipped": [r for r in rows if r["status"] != "used"],
                      "n_calibration_negatives_median": float(np.median([r["n_calibration_negatives"] for r in used])),
                      "split_version": split["version"], "split_sha256": manifest["split_sha256"], "bins_version": bins.version,
                      "bins_sha256": bins.sha256, "model_name": manifest["model_name"], "preproc_mode": manifest["preproc_mode"],
                      "kind": "plain", "unit": "site"}
        write_json_atomic(out / "thresholds.json", thresholds)
        # ---- pass 2: count at each site's thresholds
        test_paths = []
        for k, (site, (table, fold)) in enumerate(units.items()):
            emb, recs, meta = load_site(sites[site])
            if fold is not None:
                meta = meta.with_folds(fold)
            acc = accumulate_video(emb, meta, bins, table, device, fail_pos_all=True)
            acc = _finish_acc(acc, recs, meta, SimpleNamespace(site=site, pov=None), site, "test")
            test_paths.append(_save_acc(acc, out / "acc" / "test" / f"{_slug(site, set())}.npz"))
            if verbose:
                print(f"[acc {mode}] {k + 1}/{len(units)} {site}: {meta.n} crops, {int(acc['hist'].sum()):,} pairs", file=sys.stderr)
        write_json_atomic(out / "run.json", {"template_dir": str(tdir), "manifest": manifest, "split_file": str(split_path), "kind": "plain",
                                             "threshold_mode": mode, "data_root": str(data_root), "bins_file": str(bins_path),
                                             "bins_sha256": bins.sha256})
        extra = {"threshold_protocol": GALLERY_PROTOCOLS[mode], "threshold_mode": mode, "threshold_units_used": thresholds["n_units_used"],
                 "threshold_units_skipped": thresholds["units_skipped"], "oracle": "oracle" in mode, "unit": "site",
                 "n_calibration_negatives_median": thresholds["n_calibration_negatives_median"]}
        model = f"{manifest['model_name']} / {manifest['preproc_mode']}"
        write_results(out, bins, thresholds, [], test_paths, data_root, kind="plain", per_subset=per_subset, verbose=verbose,
                      check_thresholds=False, extra=extra, subsets=("site",),
                      title=f"Site matching (gallery = whole site), threshold per site {'(ORACLE, optimistic)' if 'oracle' in mode else '(held out)'}: {model}")
        out_dirs[mode] = out
    _refresh_variant_comparison(Path(out_base) / f"{manifest['model_name']}__{manifest['preproc_mode']}", split["version"])
    return out_dirs
