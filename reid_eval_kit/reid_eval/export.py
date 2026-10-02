"""Generic template exporter: turn ANY encoder (one Python function) into templates this evaluation can read.

    def encode(images):            # list of BGR uint8 images (H, W, 3), already cut according to --preproc, NOT resized
        ...                        # resize / normalise / run your model here
        return features            # array-like (len(images), D), raw features (no L2 normalisation needed)

    python export_templates.py --data DATASET --split splits/eval_split_v1.json --out templates \
        --model-name mymodel --preproc unpad_stretch --encoder mypackage.mymodule:encode
"""
from __future__ import annotations

import datetime as dt
import importlib
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from reid_data import group_by_video, load_dataset
from .common import read_json, sha256_file, write_json_atomic
from .preprocess import MODES, crop_for_mode
from .templates import SETS, check_alignment, load_video_npz, save_video_npz, template_dir, video_file


def load_callable(spec):
    """'package.module:function' (or 'path/to/file.py:function')."""
    mod, _, fn = spec.partition(":")
    if not fn:
        raise ValueError(f"--encoder must be 'module:function', got {spec!r}")
    if mod.endswith(".py"):
        sys.path.insert(0, str(Path(mod).resolve().parent))
        mod = Path(mod).stem
    return getattr(importlib.import_module(mod), fn)


def export_templates(data, split_path, out, model_name, encode, preproc="unpad_stretch", batch_size=64, sets=SETS, overwrite=False,
                     notes=None, verbose=True):
    """Encode every kept crop of the split's videos with `encode` and write one .npz per video + manifest.json. Resumable."""
    if preproc not in MODES:
        raise ValueError(f"preproc must be one of {MODES}")
    split = read_json(split_path)
    ids = sorted({v for s in sets for v in split[s]})
    by_video = group_by_video(load_dataset(data, video_ids=ids, verbose=False))
    tdir = template_dir(out, model_name, preproc)
    n_img, t0, dim = 0, time.time(), None
    for s in sets:
        for k, vid in enumerate(split[s]):
            recs, path = by_video[vid], video_file(tdir, s, vid)
            if path.is_file() and not overwrite:
                check_alignment(load_video_npz(path), recs, f"{s}/{vid}")
                continue
            feats = []
            for i in range(0, len(recs), batch_size):
                imgs = [crop_for_mode(cv2.imread(str(r.path), cv2.IMREAD_COLOR), r.pad_x, r.pad_y, preproc) for r in recs[i:i + batch_size]]
                f = np.asarray(encode(imgs), dtype=np.float32)
                if f.ndim != 2 or len(f) != len(imgs):
                    raise ValueError(f"encode() must return (len(images), D); got {f.shape} for {len(imgs)} images")
                feats.append(f)
            emb = np.concatenate(feats)
            if not np.isfinite(emb).all():
                raise ValueError(f"{s}/{vid}: NaN/Inf in the features")
            dim = emb.shape[1]
            save_video_npz(path, emb, [r.crop_uid for r in recs], [r.tracklet_id for r in recs], [r.frame for r in recs])
            n_img += len(recs)
            if verbose:
                print(f"[{model_name}/{preproc}] {s} {k + 1}/{len(split[s])} {vid}: {len(recs)} crops, dim {dim}", file=sys.stderr)
    write_json_atomic(tdir / "manifest.json", {
        "model_name": model_name, "preproc_mode": preproc, "split_file": str(split_path), "split_sha256": sha256_file(split_path),
        "split_version": split.get("version"), "dataset_root": str(data), "embedding_dim": dim, "created": dt.datetime.now().replace(microsecond=0).isoformat(),
        "throughput_img_per_s": round(n_img / max(time.time() - t0, 1e-9), 2) if n_img else None, "exporter": "reid_eval.export (generic)",
        "notes": notes or [], "counts": {s: {"videos": len(split[s]), "crops": sum(len(by_video[v]) for v in split[s])} for s in sets}})
    return tdir
