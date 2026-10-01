#!/usr/bin/env python
"""Task 3 check: validate an encoder wrapper before using it on our data.

  # 1) reproduce the published benchmark (VeRi-776 layout: image_query/, image_test/, files named 0002_c002_00030600_0.jpg)
  python validate_encoder.py --model fastreid_veri_sbs_r50ibn --weights veri_sbs_R50-ibn.pth --veri-root /data/VeRi
  #    published mAP must be reproduced within about 1 point (no re-ranking, no camera/view ids, same size & normalisation)

  # 2) fallback without benchmark data: same-tracklet pairs must clearly beat different-vehicle pairs
  python validate_encoder.py --model ... --weights ... --data <dataset root> --split splits/eval_split_v1.json
"""
import argparse
import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from reid_data import group_by_video, load_dataset
from reid_eval.encoders import available_encoders, build_encoder
from reid_eval.common import read_json


def map_cmc(sim, q_ids, q_cams, g_ids, g_cams, max_rank=20):
    """VeRi/Market protocol: gallery images with the same id AND camera as the query are ignored."""
    aps, cmc = [], np.zeros(max_rank)
    n_valid = 0
    for i in range(len(q_ids)):
        order = np.argsort(-sim[i], kind="stable")
        keep = ~((g_ids[order] == q_ids[i]) & (g_cams[order] == q_cams[i]))
        match = (g_ids[order] == q_ids[i])[keep]
        if not match.any():
            continue
        n_valid += 1
        hits = np.cumsum(match)
        aps.append(((hits / (np.arange(len(match)) + 1)) * match).sum() / match.sum())
        first = int(np.argmax(match))
        if first < max_rank:
            cmc[first:] += 1
    return float(np.mean(aps)), cmc / max(n_valid, 1)


def encode_files(enc, files, batch=128):
    out = []
    for k in range(0, len(files), batch):
        x = torch.stack([enc.preprocess(cv2.imread(str(f), cv2.IMREAD_COLOR), 0, 0, "letterbox") for f in files[k:k + batch]])
        out.append(enc.encode(x))
    f = torch.cat(out)
    return torch.nn.functional.normalize(f, dim=1).numpy()


def veri(enc, root):
    pat = re.compile(r"(\d+)_c(\d+)_")
    parse = lambda fs: (np.array([int(pat.match(f.name).group(1)) for f in fs]), np.array([int(pat.match(f.name).group(2)) for f in fs]))
    q = sorted((Path(root) / "image_query").glob("*.jpg")); g = sorted((Path(root) / "image_test").glob("*.jpg"))
    fq, fg = encode_files(enc, q), encode_files(enc, g)
    (qi, qc), (gi, gc) = parse(q), parse(g)
    mAP, cmc = map_cmc(fq @ fg.T, qi, qc, gi, gc)
    print(f"VeRi-776: mAP {mAP * 100:.2f}  Rank-1 {cmc[0] * 100:.2f}  Rank-5 {cmc[4] * 100:.2f}   (compare with the published number, tolerance ~1 mAP)")


def own_pairs(enc, data, split, mode, n=20, seed=0):
    ids = read_json(split)["validation"]
    recs = load_dataset(data, ids, verbose=False)
    rng = random.Random(seed)
    pos, neg = [], []
    for vid, rs in group_by_video(recs).items():
        by = {}
        for r in rs:
            by.setdefault(r.tracklet_id, []).append(r)
        multi = [t for t, v in by.items() if len(v) >= 2]
        if len(multi) < 2:
            continue
        a, b = rng.sample(multi, 2)
        x, y = rng.sample(by[a], 2); z = rng.choice(by[b])
        pos.append((x, y)); neg.append((x, z))
    pos, neg = pos[:n], neg[:n]
    emb = lambda r: torch.nn.functional.normalize(enc.encode(enc.preprocess(cv2.imread(str(r.path)), r.pad_x, r.pad_y, mode)[None]), dim=1)[0]
    sp = np.array([float(emb(a) @ emb(b)) for a, b in pos]); sn = np.array([float(emb(a) @ emb(b)) for a, b in neg])
    print(f"{len(pos)} same-tracklet pairs: mean cos {sp.mean():.3f};  {len(neg)} different-vehicle pairs: mean cos {sn.mean():.3f};  "
          f"positive > negative in {(sp > sn).mean() * 100:.0f}% of the matched pairs")
    return sp, sn


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help=f"one of {available_encoders()}")
    ap.add_argument("--weights"); ap.add_argument("--config"); ap.add_argument("--clipreid-repo"); ap.add_argument("--device")
    ap.add_argument("--veri-root"); ap.add_argument("--data"); ap.add_argument("--split")
    ap.add_argument("--preproc", default="unpad_stretch")
    a = ap.parse_args(argv)
    kw = {k: v for k, v in dict(weights=a.weights, config=a.config, clipreid_repo=a.clipreid_repo, device=a.device).items() if v}
    enc = build_encoder(a.model, **kw)
    print("embedding dim:", enc.describe()["embedding_dim"], " input size:", enc.input_size)
    if a.veri_root:
        veri(enc, a.veri_root)
    if a.data and a.split:
        own_pairs(enc, a.data, a.split, a.preproc)
    if not (a.veri_root or (a.data and a.split)):
        ap.error("give --veri-root and/or --data with --split")
    return 0


if __name__ == "__main__":
    sys.exit(main())
