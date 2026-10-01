#!/usr/bin/env python
"""Encode every kept crop of the validation and test videos once per model (spec section 6).

  python extract_templates.py --model fastreid_veriwild_r50ibn --weights veriwild_bot_R50-ibn.pth \
      --split splits/eval_split_v1.json --data D:/data/Re-ID_safe_test --out templates/ --preproc unpad_stretch
"""
import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from reid_data import load_dataset, group_by_video
from reid_eval.common import read_json, sha256_file, write_json_atomic
from reid_eval.encoders import available_encoders, build_encoder
from reid_eval.preprocess import MODES
from reid_eval.templates import SETS, check_alignment, load_video_npz, save_video_npz, template_dir, video_file

COMPAT_KEYS = ("model_name", "checkpoint_sha256", "preproc_mode", "input_size", "flip_tta", "split_sha256", "batch_size", "embedding_dim")


class CropDataset(torch.utils.data.Dataset):
    def __init__(self, records, encoder, mode):
        self.records, self.encoder, self.mode = records, encoder, mode

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        img = cv2.imread(str(r.path), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"cannot read {r.path}")
        return self.encoder.preprocess(img, r.pad_x, r.pad_y, self.mode)


def encode_records(encoder, records, mode, batch_size, workers, flip):
    dl = torch.utils.data.DataLoader(CropDataset(records, encoder, mode), batch_size=batch_size, shuffle=False,
                                     num_workers=workers, drop_last=False)
    out = []
    with torch.no_grad():
        for batch in dl:
            f = encoder.encode(batch)
            if flip:
                f = (f + encoder.encode(batch.flip(-1))) / 2
            out.append(f.float().cpu())
    return torch.cat(out).numpy().astype(np.float32)


def verify(tdir, records_by_video, split, encoder, mode, batch_size, flip, seed=0, n_recheck=20):
    """Post-extraction checks. Raises on failure."""
    norms_all, rows = [], []
    for s in SETS:
        for vid in split[s]:
            p = video_file(tdir, s, vid)
            if not p.is_file():
                continue
            t = load_video_npz(p)
            check_alignment(t, records_by_video[vid], f"{s}/{vid}")
            if not np.isfinite(t["emb"]).all():
                raise ValueError(f"{s}/{vid}: NaN or Inf in embeddings")
            norms_all.append(np.linalg.norm(t["emb"], axis=1))
            rows.extend((p, i, vid) for i in range(len(t["emb"])))
    if not rows:
        return {"n_rows": 0}
    norms = np.concatenate(norms_all)
    if len(norms) > 1 and np.ptp(norms) < 1e-6 * max(1.0, float(norms.max())):
        raise ValueError("all embedding norms are equal: the model output is constant (broken checkpoint load?)")
    rng = np.random.RandomState(seed)
    pick = [rows[i] for i in rng.choice(len(rows), size=min(n_recheck, len(rows)), replace=False)]
    recs = [records_by_video[vid][i] for _, i, vid in pick]
    new = encode_records(encoder, recs, mode, batch_size=min(batch_size, len(recs)), workers=0, flip=flip)
    old = np.stack([load_video_npz(p)["emb"][i] for p, i, _ in pick])
    cos = (new * old).sum(1) / (np.linalg.norm(new, axis=1) * np.linalg.norm(old, axis=1) + 1e-12)
    if cos.min() <= 0.9999:
        raise ValueError(f"re-encoding {len(pick)} crops gave min cosine {cos.min():.6f} <= 0.9999 (non-deterministic encoder)")
    return {"n_rows": len(rows), "norm_min": float(norms.min()), "norm_max": float(norms.max()), "recheck_min_cos": float(cos.min())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help=f"one of {available_encoders()}")
    ap.add_argument("--split", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="templates/")
    ap.add_argument("--preproc", nargs="+", default=["unpad_stretch"], choices=MODES)
    ap.add_argument("--sets", nargs="+", default=list(SETS), choices=SETS)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--config", default=None, help="override the encoder's default config file")
    ap.add_argument("--clipreid-repo", default=None)
    ap.add_argument("--num-classes", type=int, default=None, help="CLIP-ReID: classes of the checkpoint")
    ap.add_argument("--trust-checkpoint", action="store_true", help="allow pickled (non-tensor) checkpoint contents")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--flip-tta", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")

    split = read_json(a.split)
    split_sha = sha256_file(a.split)
    summary = Path(a.data) / "summary.json"
    if split.get("dataset_summary_sha256") and summary.is_file() and sha256_file(summary) != split["dataset_summary_sha256"]:
        sys.exit(f"{summary} changed since the split was made. Create eval_split_v2.json for the new dataset; never reuse v1.")
    if str(Path(split["dataset_root"])) != str(Path(a.data)):
        print(f"note: split was made on {split['dataset_root']}, extracting from {a.data}", file=sys.stderr)

    kw = dict(weights=a.weights, device=a.device, config=a.config, trust_checkpoint=a.trust_checkpoint,
              clipreid_repo=a.clipreid_repo)
    if a.num_classes is not None:
        kw["num_classes"] = a.num_classes
    encoder = build_encoder(a.model, **{k: v for k, v in kw.items() if v is not None})
    desc = encoder.describe()

    ids = sorted({v for s in a.sets for v in split[s]})
    records = load_dataset(a.data, video_ids=ids)
    by_video = group_by_video(records)

    for mode in a.preproc:
        tdir = template_dir(a.out, a.model, mode)
        mpath = tdir / "manifest.json"
        manifest = {**desc, "model_name": a.model, "preproc_mode": mode, "flip_tta": a.flip_tta,
                    "split_file": a.split, "split_sha256": split_sha, "split_version": split.get("version"),
                    "dataset_root": str(a.data), "batch_size": a.batch_size, "seed": a.seed,
                    "num_workers_note": "affects speed only", "torch": torch.__version__}
        if mpath.is_file() and not a.overwrite:
            old = read_json(mpath)
            bad = [k for k in COMPAT_KEYS if old.get(k) != manifest.get(k)]
            if bad:
                sys.exit(f"{mpath} exists with different {bad}; use --overwrite to recompute everything or another --out")
            manifest["throughput_img_per_s"] = old.get("throughput_img_per_s", 0.0)
            manifest["created"] = old.get("created")
        else:
            manifest["throughput_img_per_s"] = 0.0
            manifest["created"] = dt.datetime.now().replace(microsecond=0).isoformat()

        n_img, t_enc = 0, 0.0
        for s in a.sets:
            for k, vid in enumerate(split[s]):
                path = video_file(tdir, s, vid)
                recs = by_video[vid]
                if path.is_file() and not a.overwrite:
                    check_alignment(load_video_npz(path), recs, f"{s}/{vid}")  # a stale file must not be silently reused
                    continue
                t0 = time.time()
                emb = encode_records(encoder, recs, mode, a.batch_size, a.num_workers, a.flip_tta)
                t_enc += time.time() - t0; n_img += len(recs)
                save_video_npz(path, emb, [r.crop_uid for r in recs], [r.tracklet_id for r in recs], [r.frame for r in recs])
                print(f"[{mode}] {s} {k + 1}/{len(split[s])} {vid}: {len(recs)} crops, dim {emb.shape[1]}", file=sys.stderr)
        if n_img:
            manifest["throughput_img_per_s"] = round(n_img / max(t_enc, 1e-9), 2)
        manifest["embedding_dim"] = desc["embedding_dim"]
        manifest["counts"] = {s: {"videos": len(split[s]), "crops": sum(len(by_video[v]) for v in split[s])} for s in a.sets}
        manifest["checks"] = verify(tdir, by_video, {s: split[s] for s in a.sets}, encoder, mode, a.batch_size, a.flip_tta, a.seed)
        write_json_atomic(mpath, manifest)
        print(f"[{mode}] done: {tdir} (encoded {n_img} new crops; checks {manifest['checks']})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
