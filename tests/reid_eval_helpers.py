"""Shared test helpers: random per-video metadata and a slow brute-force reference for the accumulator."""
import numpy as np

from reid_eval.accumulate import VideoMeta
from reid_eval.bins import N_FINE, N_HIST


def random_meta(rng, n_obj=7, crops=(2, 14), with_kp=False, K=6):
    counts = rng.randint(crops[0], crops[1] + 1, size=n_obj)
    trk = np.repeat(np.arange(n_obj), counts)
    N = len(trk)
    pos = rng.uniform(0, 60, size=(N, 2))
    pos_ok = rng.rand(N) > 0.15
    az = rng.uniform(-180, 180, size=N); az[rng.rand(N) < 0.1] = np.nan
    occ = rng.rand(N) < 0.2
    kp = kp_ok = None
    if with_kp:
        kp = rng.rand(N, K) < 0.6
        kp_ok = rng.rand(N) > 0.1
    return VideoMeta([f"v_{t}" for t in range(n_obj)], trk, pos, pos_ok, az, occ, kp, kp_ok)


def bucket(v, edges):
    i = int(np.searchsorted(edges[1:-1], v, side="right"))
    return min(i, len(edges) - 2)


def brute_force(emb, meta, bins, thr):
    E = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    S = E @ E.T
    P1, A1, O3, K1 = bins.cell_shape
    N = len(E); T = len(meta.tracklet_ids)
    cn = np.bincount(meta.trk, minlength=T).astype(float)
    NC = bins.n_cells
    hist = np.zeros((NC, 2, N_HIST)); histb = np.zeros((NC, 2, N_HIST)); cnt = np.zeros((NC, 2, len(thr)))
    cntb = np.zeros_like(cnt); fine = np.zeros((2, N_FINE))
    pos_keys, neg_keys = set(), set()
    pairs = {}
    for i in range(N):
        for j in range(i + 1, N):
            s = float(np.clip(S[i, j], -1, 1)); ti, tj = meta.trk[i], meta.trk[j]
            ty = int(ti != tj)
            pb = bucket(np.linalg.norm(meta.pos[i] - meta.pos[j]), bins.pos_edges) if meta.pos_ok[i] and meta.pos_ok[j] else P1 - 1
            if np.isnan(meta.az[i]) or np.isnan(meta.az[j]):
                ab = A1 - 1
            else:
                d = abs(meta.az[i] - meta.az[j]); d = 360 - d if d > 180 else d
                ab = bucket(d, bins.az_edges)
            ob = int(meta.occ[i]) + int(meta.occ[j])
            kb = K1 - 1
            if meta.kp is not None and meta.kp_ok[i] and meta.kp_ok[j]:
                u = (meta.kp[i] | meta.kp[j]).sum()
                if u > 0:
                    kb = bucket((meta.kp[i] & meta.kp[j]).sum() / u, bins.kp_edges)
            c = ((pb * A1 + ab) * O3 + ob) * K1 + kb
            w = 1 / (cn[ti] * cn[tj]) if ty else 1 / (cn[ti] * (cn[ti] - 1) / 2)
            b = min(int((s + 1) * N_HIST / 2), N_HIST - 1)
            hist[c, ty, b] += 1; histb[c, ty, b] += w
            fine[ty, min(int((s + 1) * N_FINE / 2), N_FINE - 1)] += 1
            for q, t in enumerate(thr):
                if s >= t:
                    cnt[c, ty, q] += 1; cntb[c, ty, q] += w
            (neg_keys if ty else pos_keys).add((c, ti, tj) if ty else (c, ti))
            if ty:
                pairs.setdefault((ti, tj), []).append(s)
    return dict(hist=hist, histb=histb, cnt=cnt, cntb=cntb, fine=fine, pos_keys=pos_keys, neg_keys=neg_keys, pairs=pairs)
