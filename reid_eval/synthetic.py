"""Tiny fake dataset in the spec's layout (section 2/3), for tests and demos only."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def make_fake_dataset(root, n_videos=6, n_sites=3, tracklets=(3, 5), crops=(4, 9), seed=0, windows_paths=True,
                      calibration_every=0, keypoints=0):
    """Build `root/videos/<id>/{meta.json,crops/,rejected/}` + summary.json. Returns {video_id: [tracklet_ids]}.

    Each tracklet is a coloured box on black, letterboxed into 224x224. `calibration_every=k` makes every
    k-th video uncalibrated (ineligible for the split). Some crops are rejected and written to rejected/.
    """
    rng = np.random.RandomState(seed)
    root = Path(root)
    out = {}
    summary = {"videos": {}}
    for v in range(n_videos):
        vid = f"vid{v:03d}"
        folder = root / "videos" / vid
        (folder / "crops").mkdir(parents=True, exist_ok=True)
        n_tr = int(rng.randint(tracklets[0], tracklets[1] + 1))
        tracks, rejected = [], []
        for t in range(1, n_tr + 1):
            colour = rng.randint(30, 255, size=3)
            n_c = int(rng.randint(crops[0], crops[1] + 1))
            (folder / "crops" / f"{t:05d}").mkdir(parents=True, exist_ok=True)
            x0, y0 = rng.uniform(-30, 30, size=2)
            vx, vy = rng.uniform(-2, 2, size=2)
            clist = []
            for k in range(n_c):
                frame = 15 * (k + 1)
                w = int(rng.randint(120, 220)); h = int(rng.randint(80, 150))
                img = np.zeros((224, 224, 3), np.uint8)
                px, py = (224 - w) // 2, (224 - h) // 2
                body = np.clip(colour + rng.randint(-15, 16, size=(h, w, 3)), 0, 255).astype(np.uint8)
                img[py:py + h, px:px + w] = body
                rel = f"crops/{t:05d}/{frame:06d}.jpg"
                cv2.imwrite(str(folder / rel), img)
                reliable = bool(rng.rand() > 0.1)
                heading = rng.uniform(-180, 180)
                clist.append({
                    "crop_file": rel.replace("/", "\\") if windows_paths and k % 2 else rel,
                    "frame": frame, "timestamp_s": frame / 30.0,
                    "bbox_xyxy": [0, 0, w, h], "bbox_wh": [w, h],
                    "crop_transform": {"scale_x": 1.0, "scale_y": 1.0, "pad_x": px, "pad_y": py, "crop_box_px": [0, 0, w, h]},
                    "position_road_m": [x0 + vx * k, y0 + vy * k, 0.0] if reliable else None,
                    "position_cam_m": [0.0, 0.0, 10.0], "distance_m": 10.0 if reliable else None,
                    "position_reliable": reliable,
                    "view_azimuth_deg": float(heading) if rng.rand() > 0.05 else None,
                    "view_bin": "front", "occluded_annot": bool(rng.rand() < 0.05),
                    "overlap_by_closer_box": float(rng.rand() * 0.5),
                    "quality": {"brightness": 100.0, "flags": []}, "blur_laplacian_var": 50.0,
                })
                if keypoints:  # fake keypoints in the planned format (coordinates in the 224 crop)
                    vis = (rng.rand(keypoints) < 0.6).tolist()
                    clist[-1]["keypoints"] = {"schema": "fake", "points": rng.uniform(0, 224, size=(keypoints, 3)).tolist(),
                                              "visible": vis, "visibility_threshold": 0.5}
            tracks.append({"track_id": t, "tracklet_id": f"{vid}_{t}", "identity_id": f"{vid}_{t}", "label": "car",
                           "first_frame": 15, "last_frame": 15 * n_c, "n_crops": n_c, "crops": clist})
            # a rejected crop that must never be read
            (folder / "rejected" / f"{t:05d}").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(folder / "rejected" / f"{t:05d}" / "000001.jpg"), np.full((224, 224, 3), 255, np.uint8))
            rejected.append({"crop_file": f"rejected/{t:05d}/000001.jpg", "reasons": ["blur"]})
        n_crops = sum(len(tr["crops"]) for tr in tracks)
        calibrated = not (calibration_every and v % calibration_every == calibration_every - 1)
        meta = {"video_id": vid, "site": f"site{v % n_sites}", "country": "XX", "orient": "A", "pov": "front" if v % 2 else "rear",
                "time": "day", "fps": 30.0, "calibration": {"available": calibrated},
                "crop_params": {"size": 224, "resize_mode": "letterbox", "effective_fps": 5.0},
                "quality_params": None, "n_tracks": n_tr, "n_crops": n_crops, "tracks": tracks, "rejected_crops": rejected}
        (folder / "meta.json").write_text(json.dumps(meta))
        out[vid] = [tr["tracklet_id"] for tr in tracks]
        summary["videos"][vid] = {"status": "ok", "n_crops": n_crops}
    (root / "summary.json").write_text(json.dumps(summary))
    return out


def make_fake_templates(tdir, split, by_video, split_sha256, noise=0.0, dim=16, seed=0, model_name="fake", mode="letterbox"):
    """Templates whose similarities are known: embedding = one-hot(tracklet) + Gaussian noise (std `noise`)."""
    from .templates import save_video_npz, video_file
    from .common import write_json_atomic
    rng = np.random.RandomState(seed)
    tdir = Path(tdir)
    for set_name in ("validation", "test"):
        for vid in split[set_name]:
            recs = by_video[vid]
            tids = sorted({r.tracklet_id for r in recs})
            assert len(tids) <= dim
            emb = np.stack([np.eye(dim)[tids.index(r.tracklet_id)] for r in recs]) + noise * rng.randn(len(recs), dim)
            save_video_npz(video_file(tdir, set_name, vid), emb.astype(np.float32), [r.crop_uid for r in recs],
                           [r.tracklet_id for r in recs], [r.frame for r in recs])
    write_json_atomic(tdir / "manifest.json", {"model_name": model_name, "preproc_mode": mode, "split_sha256": split_sha256})
