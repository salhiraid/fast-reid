"""Template files: one .npz per video, rows in the canonical record order (spec section 6)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

SETS = ("validation", "test")


def template_dir(out, model_name, mode) -> Path:
    return Path(out) / f"{model_name}__{mode}"


def save_video_npz(path, emb, crop_uid, tracklet_id, frame):
    """Atomic: write a temp file in the same folder, then rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, emb=np.asarray(emb, np.float32), crop_uid=np.asarray(crop_uid, dtype=str),
                 tracklet_id=np.asarray(tracklet_id, dtype=str), frame=np.asarray(frame, np.int32))
    os.replace(tmp, path)


def load_video_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in ("emb", "crop_uid", "tracklet_id", "frame")}


def check_alignment(tpl, records, where=""):
    """Fail if the file's crop_uid differ in any way from the loader records (missing, extra, or reordered)."""
    uids = [r.crop_uid for r in records]
    got = tpl["crop_uid"].tolist()
    if got != uids:
        missing = set(uids) - set(got)
        extra = set(got) - set(uids)
        raise ValueError(f"{where}: templates do not match the dataset records "
                         f"({len(missing)} missing, {len(extra)} extra, order_ok={not missing and not extra})")
    if tpl["tracklet_id"].tolist() != [r.tracklet_id for r in records]:
        raise ValueError(f"{where}: tracklet_id in the template differs from the dataset")
    if len(tpl["emb"]) != len(records):
        raise ValueError(f"{where}: {len(tpl['emb'])} embeddings for {len(records)} records")


def video_file(tdir, split_set, video_id) -> Path:
    return Path(tdir) / split_set / f"{video_id}.npz"
