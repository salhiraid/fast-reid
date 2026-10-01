"""Loader for the vehicle crop dataset (spec sections 2-4).

Everything downstream (encoding, evaluation) works on the flat list of `CropRecord`
returned here. The loader reads `meta.json` only (never walks the crop folders), uses
`tracks[*].crops` only (never `rejected_crops`), and never writes under the dataset root.
"""
from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

CROP_SIZE = 224
OCCLUSION_OVERLAP_THRESHOLD = 0.3


@dataclass(frozen=True)
class CropRecord:
    crop_uid: str  # "<video_id>/<crop_file>", unique key joining embeddings and metadata
    video_id: str
    tracklet_id: str  # object id for evaluation
    identity_id: str
    label: str
    frame: int
    timestamp_s: float
    path: Path  # absolute path to the JPEG
    pad_x: int
    pad_y: int
    bbox_wh: tuple
    position_xy: Optional[tuple]  # position_road_m[:2] if position_reliable else None
    distance_m: Optional[float]  # only if position_reliable
    azimuth_deg: Optional[float]
    view_bin: Optional[str]
    occluded: bool  # occluded_annot or overlap_by_closer_box > 0.3
    occlusion_source: str  # "annotated" | "overlap" | "none"
    keypoints_visible: Optional[np.ndarray] = field(default=None, compare=False)  # None until keypoints exist
    quality: dict = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class VideoInfo:
    video_id: str
    site: Optional[str]
    country: Optional[str]
    orient: Optional[str]
    pov: Optional[str]
    time: Optional[str]
    calibration_available: bool
    n_crops_meta: int
    n_tracks_meta: int
    effective_fps: Optional[float]
    quality_params: Optional[dict]
    folder: Path


def _num(x):
    return None if x is None else float(x)


def parse_keypoints(crop: dict) -> Optional[np.ndarray]:
    """Return a boolean visibility array, or None if the crop has no keypoint information.

    A missing `keypoints` field means "unknown", never "nothing visible". This is the only
    place that knows the keypoint format; adapt it if the final format differs.
    """
    kp = crop.get("keypoints")
    if not kp:
        return None
    vis = kp.get("visible")
    if vis is not None:
        return np.asarray(vis, dtype=bool)
    pts = kp.get("points")
    if pts is None:
        return None
    thr = kp.get("visibility_threshold", 0.5)
    return np.asarray([p[2] >= thr for p in pts], dtype=bool)


def _norm_rel(p: str) -> str:
    """Crop paths may contain backslashes when built on Windows."""
    return p.replace("\\", "/").lstrip("/")


def _discover_videos(root: Path) -> list:
    vroot = root / "videos"
    if not vroot.is_dir():
        raise FileNotFoundError(f"{vroot} does not exist; is --data the dataset root?")
    return sorted(d.name for d in vroot.iterdir() if (d / "meta.json").is_file())


def _read_video(root: Path, video_id: str, labels, check_files: bool):
    folder = root / "videos" / video_id
    with open(folder / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("video_id", video_id) != video_id:
        raise ValueError(f"{folder}/meta.json video_id={meta.get('video_id')!r} does not match folder {video_id!r}")

    records, n_expected = [], 0
    for tr in meta.get("tracks", []):  # kept tracks only; `rejected_crops` is never read
        if labels is not None and tr.get("label") not in labels:
            continue
        track_id = tr["track_id"]
        tracklet_id = tr.get("tracklet_id") or f"{video_id}_{track_id}"
        identity_id = tr.get("identity_id") or tracklet_id
        for c in tr.get("crops", []):
            n_expected += 1
            rel = _norm_rel(c["crop_file"])
            if "rejected" in rel.split("/"):
                raise ValueError(f"{video_id}: kept crop points into a rejected folder: {rel}")
            path = folder / rel
            if check_files and not path.is_file():
                raise FileNotFoundError(f"{video_id}: missing crop file {path}")
            ct = c.get("crop_transform") or {}
            reliable = bool(c.get("position_reliable"))
            pos = c.get("position_road_m")
            occ_annot = bool(c.get("occluded_annot"))
            overlap = c.get("overlap_by_closer_box")
            overlap_occ = overlap is not None and float(overlap) > OCCLUSION_OVERLAP_THRESHOLD
            bw = c.get("bbox_wh") or [float("nan")] * 2
            records.append(CropRecord(
                crop_uid=f"{video_id}/{rel}",
                video_id=video_id,
                tracklet_id=tracklet_id,
                identity_id=identity_id,
                label=tr.get("label"),
                frame=int(c["frame"]),
                timestamp_s=float(c.get("timestamp_s") or 0.0),
                path=path.resolve() if check_files else path,
                pad_x=int(round(ct.get("pad_x") or 0)),
                pad_y=int(round(ct.get("pad_y") or 0)),
                bbox_wh=(float(bw[0]), float(bw[1])),
                position_xy=(float(pos[0]), float(pos[1])) if (reliable and pos is not None) else None,
                distance_m=_num(c.get("distance_m")) if reliable else None,
                azimuth_deg=_num(c.get("view_azimuth_deg")),
                view_bin=c.get("view_bin"),
                occluded=occ_annot or overlap_occ,
                occlusion_source="annotated" if occ_annot else ("overlap" if overlap_occ else "none"),
                keypoints_visible=parse_keypoints(c),
                quality=dict(c.get("quality") or {}),
            ))

    cal = meta.get("calibration") or {}
    cp = meta.get("crop_params") or {}
    info = VideoInfo(
        video_id=video_id, site=meta.get("site"), country=meta.get("country"), orient=meta.get("orient"),
        pov=meta.get("pov"), time=meta.get("time"), calibration_available=bool(cal.get("available")),
        n_crops_meta=int(meta.get("n_crops", n_expected)), n_tracks_meta=int(meta.get("n_tracks", len(meta.get("tracks", [])))),
        effective_fps=_num(cp.get("effective_fps")), quality_params=meta.get("quality_params"), folder=folder)
    if labels is None and info.n_crops_meta != len(records):
        raise ValueError(f"{video_id}: meta.json n_crops={info.n_crops_meta} but tracks list {len(records)} kept crops")
    return records, info


def load_dataset_with_info(root, video_ids: Optional[Iterable[str]] = None, labels: Optional[Sequence[str]] = None,
                           verbose: bool = True, check_files: bool = True, sample_images: int = 100, seed: int = 0):
    """Return (records, {video_id: VideoInfo}); records are in the canonical order."""
    root = Path(root)
    ids = sorted(set(video_ids)) if video_ids is not None else _discover_videos(root)
    labels = set(labels) if labels is not None else None
    records, infos = [], {}
    for vid in ids:
        if not (root / "videos" / vid / "meta.json").is_file():
            raise FileNotFoundError(f"video {vid!r} has no meta.json under {root / 'videos'}")
        recs, info = _read_video(root, vid, labels, check_files)
        records.extend(recs)
        infos[vid] = info
    records.sort(key=lambda r: (r.video_id, r.tracklet_id, r.frame, r.crop_uid))

    uids = [r.crop_uid for r in records]
    if len(set(uids)) != len(uids):
        raise ValueError("crop_uid is not unique")
    if check_files and records and sample_images:
        import cv2  # local import: the loader is otherwise numpy-only
        rng = random.Random(seed)
        for r in rng.sample(records, min(sample_images, len(records))):
            img = cv2.imread(str(r.path), cv2.IMREAD_COLOR)
            if img is None or img.shape != (CROP_SIZE, CROP_SIZE, 3):
                raise ValueError(f"{r.path}: expected {CROP_SIZE}x{CROP_SIZE}x3, got {None if img is None else img.shape}")
    if verbose:
        _print_summary(records, infos)
    return records, infos


def load_dataset(root, video_ids: Optional[Iterable[str]] = None, labels: Optional[Sequence[str]] = None, **kw) -> list:
    """Flat list of kept crops sorted by (video_id, tracklet_id, frame): the canonical order."""
    return load_dataset_with_info(root, video_ids, labels, **kw)[0]


def group_by_video(records: Sequence[CropRecord]) -> dict:
    out: dict = {}
    for r in records:
        out.setdefault(r.video_id, []).append(r)
    return out


def _print_summary(records, infos, file=None):
    file = file or sys.stderr
    print(f"{'video_id':<40} {'tracklets':>9} {'crops':>7} {'pos_ok':>7} {'azimuth':>7} {'keypts':>7}", file=file)
    tot = np.zeros(5, dtype=np.int64)
    for vid, recs in group_by_video(records).items():
        row = np.array([len({r.tracklet_id for r in recs}), len(recs), sum(r.position_xy is not None for r in recs),
                        sum(r.azimuth_deg is not None for r in recs), sum(r.keypoints_visible is not None for r in recs)])
        tot += row
        print(f"{vid[:40]:<40} {row[0]:>9} {row[1]:>7} {row[2]:>7} {row[3]:>7} {row[4]:>7}", file=file)
    print(f"{'TOTAL (' + str(len(infos)) + ' videos)':<40} {tot[0]:>9} {tot[1]:>7} {tot[2]:>7} {tot[3]:>7} {tot[4]:>7}", file=file)
