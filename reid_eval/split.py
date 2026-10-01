"""Frozen validation/test split (spec section 5)."""
from __future__ import annotations

import collections
import datetime as dt
import hashlib
import random
from pathlib import Path

from reid_data import load_dataset_with_info, group_by_video
from .common import sha256_file

DEFAULT_CRITERIA = {"min_tracklets": 2, "min_crops_per_tracklet": 2, "min_crops": 20, "require_calibration": True}


def crops_fingerprint(by_video, video_ids) -> str:
    """sha256 over (video, tracklet, crop_uid) of the kept crops of the given videos, in canonical order.

    This, not summary.json, decides whether the dataset still matches a split: build_reid_crops.py rewrites
    summary.json on every run (even a resume), while the kept crops of a finished video do not change.
    """
    h = hashlib.sha256()
    for vid in sorted(video_ids):
        for r in by_video[vid]:
            h.update(f"{vid}\t{r.tracklet_id}\t{r.crop_uid}\n".encode())
    return h.hexdigest()


def check_fingerprints(split, by_video, sets):
    """Raise if the kept crops of the split's videos changed since the split was made (legacy splits: skipped)."""
    saved = split.get("crops_fingerprint_sha256")
    if not saved:
        return
    for s in sets:
        if all(v in by_video for v in split[s]) and crops_fingerprint(by_video, split[s]) != saved[s]:
            raise ValueError(f"the kept crops of the '{s}' videos changed since split {split['version']} was made "
                             f"(quality filter / rebuild?). Create a new split version; never reuse or edit this one.")


def eligible_videos(records, infos, criteria):
    """Return (eligible ids sorted, {video_id: reason} for the rest)."""
    by_video = group_by_video(records)
    ok, why = [], {}
    for vid in sorted(infos):
        recs = by_video.get(vid, [])
        per_trk = collections.Counter(r.tracklet_id for r in recs)
        n_good = sum(c >= criteria["min_crops_per_tracklet"] for c in per_trk.values())
        if criteria["require_calibration"] and not infos[vid].calibration_available:
            why[vid] = "no calibration"
        elif n_good < criteria["min_tracklets"]:
            why[vid] = f"only {n_good} tracklets with >= {criteria['min_crops_per_tracklet']} crops"
        elif len(recs) < criteria["min_crops"]:
            why[vid] = f"only {len(recs)} crops"
        else:
            ok.append(vid)
    return ok, why


def choose(eligible, sites, n_val, n_test, seed, max_overshoot=0.5):
    """Shuffle with a fixed seed; site-disjoint validation if possible. Returns (val, test, site_disjoint, warnings)."""
    warnings = []
    rng = random.Random(seed)
    shuffled = list(eligible)
    rng.shuffle(shuffled)

    by_site = collections.OrderedDict()
    for v in shuffled:
        by_site.setdefault(sites[v], []).append(v)
    site_order = list(by_site)
    random.Random(seed + 1).shuffle(site_order)

    cap = int(n_val * (1 + max_overshoot))
    val, site_disjoint = [], False
    if len(by_site) >= 2:
        for s in site_order:
            if len(val) >= n_val:
                break
            if len(val) + len(by_site[s]) <= cap:  # whole sites only; skip a site that would overshoot too much
                val.extend(by_site[s])
        site_disjoint = len(val) >= n_val and len(val) < len(eligible)
    if not site_disjoint:
        val = shuffled[:n_val]
        warnings.append("SITE-DISJOINT validation NOT possible (too few sites or sites too large): the threshold "
                        "is tuned on cameras that also appear in the test set")
    val_set = set(val)
    rest = [v for v in shuffled if v not in val_set]
    if site_disjoint:
        val_sites = {sites[v] for v in val}
        assert not any(sites[v] in val_sites for v in rest)
    test = rest[:n_test]
    if len(val) < n_val:
        warnings.append(f"only {len(val)} eligible videos available for validation (wanted {n_val})")
    if len(test) < n_test:
        warnings.append(f"only {len(test)} eligible videos left for test (wanted {n_test}); took all of them")
    return val, test, site_disjoint, warnings


def build_split(data_root, version="v1", seed=0, n_val=30, n_test=150, criteria=None, created=None, verbose=False):
    criteria = {**DEFAULT_CRITERIA, **(criteria or {})}
    data_root = Path(data_root)
    records, infos = load_dataset_with_info(data_root, verbose=verbose, check_files=False, sample_images=0)
    eligible, why = eligible_videos(records, infos, criteria)
    sites = {v: infos[v].site or "?" for v in infos}
    val, test, site_disjoint, warnings = choose(eligible, sites, n_val, n_test, seed)
    by_video = group_by_video(records)

    def counts(ids):
        return {"videos": len(ids), "tracklets": len({r.tracklet_id for v in ids for r in by_video[v]}),
                "crops": sum(len(by_video[v]) for v in ids)}

    summary = data_root / "summary.json"
    return {
        "version": version,
        "created": created or dt.datetime.now().replace(microsecond=0).isoformat(),
        "dataset_root": str(data_root),
        "dataset_summary_sha256": sha256_file(summary) if summary.is_file() else None,  # informational only
        "crops_fingerprint_sha256": {"validation": crops_fingerprint(by_video, val), "test": crops_fingerprint(by_video, test)},
        "seed": seed,
        "criteria": criteria,
        "site_disjoint": site_disjoint,
        "validation": val,
        "test": test,
        "counts": {"validation": counts(val), "test": counts(test)},
        "n_videos_total": len(infos),
        "n_videos_eligible": len(eligible),
        "sites": {"validation": sorted({sites[v] for v in val}), "test": sorted({sites[v] for v in test})},
        "warnings": warnings,
    }
