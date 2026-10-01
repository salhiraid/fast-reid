"""Per-video pair accumulation (spec section 8).

Every unordered pair (i<j) of crops of ONE video is compared; the pair is assigned to a joint cell
(delta position, delta azimuth, occlusion, shared keypoints) and only histograms / counts are kept.
Pairs are never stored (a 2,000-crop video has ~2M pairs): row groups of whole tracklets are multiplied
against all later crops on the GPU in blocks, so every tracklet's positives and every tracklet-pair's
negatives are complete inside one group (needed for exact per-object-pair medians).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .bins import Bins, N_FINE, N_HIST

CHUNK = 1 << 22  # pairs processed at once (bounds temporary memory)


@dataclass
class VideoMeta:
    """Per-crop descriptors, aligned with the template rows (canonical record order)."""
    tracklet_ids: list          # (T,) object ids in order of first appearance
    trk: np.ndarray             # (N,) dense object index, non-decreasing
    pos: np.ndarray             # (N, 2) road position (metres); garbage where pos_ok is False
    pos_ok: np.ndarray          # (N,) bool: position_reliable
    az: np.ndarray              # (N,) azimuth degrees, NaN if unknown
    occ: np.ndarray             # (N,) bool
    kp: Optional[np.ndarray] = None     # (N, K) bool visibility, None if no crop has keypoints
    kp_ok: Optional[np.ndarray] = None  # (N,) bool: this crop has keypoint information

    @property
    def n(self):
        return len(self.trk)

    @classmethod
    def from_records(cls, records):
        tids, trk = [], np.empty(len(records), np.int64)
        index = {}
        for n, r in enumerate(records):
            if r.tracklet_id not in index:
                index[r.tracklet_id] = len(tids)
                tids.append(r.tracklet_id)
            trk[n] = index[r.tracklet_id]
        if np.any(np.diff(trk) < 0):
            raise ValueError("records must be grouped by tracklet (canonical order)")
        pos = np.array([r.position_xy if r.position_xy is not None else (0.0, 0.0) for r in records], np.float64).reshape(-1, 2)
        pos_ok = np.array([r.position_xy is not None for r in records], bool)
        az = np.array([np.nan if r.azimuth_deg is None else r.azimuth_deg for r in records], np.float64)
        occ = np.array([r.occluded for r in records], bool)
        kp = kp_ok = None
        have = [r.keypoints_visible for r in records if r.keypoints_visible is not None]
        if have:
            K = {len(h) for h in have}
            if len(K) != 1:
                raise ValueError(f"keypoint sets of different sizes in one video: {sorted(K)}")
            K = K.pop()
            kp = np.zeros((len(records), K), bool)
            kp_ok = np.zeros(len(records), bool)
            for n, r in enumerate(records):
                if r.keypoints_visible is not None:
                    kp[n] = r.keypoints_visible
                    kp_ok[n] = True
        return cls(tids, trk, pos, pos_ok, az, occ, kp, kp_ok)


def _bucket(values, edges):
    """Index of the bin [e_i, e_{i+1}) of each value; the last bin includes its upper edge."""
    inner = torch.as_tensor(edges[1:-1], dtype=values.dtype, device=values.device)
    return torch.bucketize(values, inner, right=True)


def _device(device):
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def _prep(emb, meta, device):
    dev = _device(device)
    E = F.normalize(torch.as_tensor(np.asarray(emb), dtype=torch.float32, device=dev), dim=1)
    trk = torch.as_tensor(meta.trk, device=dev)
    T = len(meta.tracklet_ids)
    counts = torch.bincount(trk, minlength=T)
    starts = np.concatenate([[0], np.cumsum(counts.cpu().numpy())])
    return dev, E, trk, T, counts, starts


def _groups(starts, N, block_pairs):
    T = len(starts) - 1
    t0 = 0
    while t0 < T:
        t1 = t0 + 1
        while t1 < T and (starts[t1 + 1] - starts[t0]) * (N - starts[t0]) <= block_pairs:
            t1 += 1
        yield t0, t1
        t0 = t1


def _upper(S):
    """Strictly upper-triangular mask of the (R, C) block of rows r0.. x columns r0.. (column > row)."""
    R, C = S.shape
    return torch.arange(C, device=S.device)[None, :] > torch.arange(R, device=S.device)[:, None]


def _fine_index(s):
    return ((s.clamp(-1, 1) + 1) * (N_FINE / 2)).long().clamp_(0, N_FINE - 1)


def fine_histograms(emb, meta, device=None, block_pairs=1 << 24):
    """Light first pass: global fine histogram (2, 2000) of positives (row 0) and negatives (row 1). Used for thresholds."""
    dev, E, trk, T, counts, starts = _prep(emb, meta, device)
    N = len(E)
    out = torch.zeros(2 * N_FINE, dtype=torch.int64, device=dev)
    for t0, t1 in _groups(starts, N, block_pairs):
        r0, r1 = int(starts[t0]), int(starts[t1])
        S = E[r0:r1] @ E[r0:].T
        mask = _upper(S)
        same = trk[r0:r1, None] == trk[None, r0:]
        ty = (~same).long()
        idx = ty * N_FINE + _fine_index(S)
        out += torch.bincount(idx[mask], minlength=2 * N_FINE)
    return out.view(2, N_FINE).cpu().numpy()


def accumulate_video(emb, meta: VideoMeta, bins: Bins, thresholds, device=None, block_pairs=1 << 24, n_fail=200):
    """Return the per-video accumulator dict (see module docstring and README_reid_eval.md for the layout)."""
    dev, E, trk, T, counts, starts = _prep(emb, meta, device)
    N = len(E)
    P1, A1, O3, K1 = bins.cell_shape
    NC = bins.n_cells
    nthr = len(thresholds)
    thr = [float(t) for t in thresholds]
    f32 = lambda a: torch.as_tensor(a, dtype=torch.float32, device=dev)
    pos, pos_ok, az = f32(meta.pos), torch.as_tensor(meta.pos_ok, device=dev), f32(meta.az)
    occ = torch.as_tensor(meta.occ, device=dev).long()
    has_kp = meta.kp is not None
    if has_kp:
        kp = torch.as_tensor(meta.kp, device=dev)
        kp_ok = torch.as_tensor(meta.kp_ok, device=dev)
        kp_n = kp.sum(1)
    cn = counts.double()
    w_pos_obj = torch.where(cn >= 2, 1.0 / (cn * (cn - 1) / 2).clamp(min=1), torch.zeros_like(cn))
    KM1 = bins.n_kpmin

    i64 = lambda n: torch.zeros(n, dtype=torch.int64, device=dev)
    f64 = lambda n: torch.zeros(n, dtype=torch.float64, device=dev)
    H, HB = i64(NC * 2 * N_HIST), f64(NC * 2 * N_HIST)
    FI, FB = i64(2 * N_FINE), f64(2 * N_FINE)
    CNT = [i64(NC * 2) for _ in range(nthr)]
    CNTB = [f64(NC * 2) for _ in range(nthr)]
    M1, M2, M1B, M2B, NB = f64(NC * 2), f64(NC * 2), f64(NC * 2), f64(NC * 2), f64(NC * 2)
    KH = i64(KM1 * 2 * N_HIST)
    KCNT = [i64(KM1 * 2) for _ in range(nthr)]
    TARN = [i64(T * P1) for _ in range(nthr)]
    TARD = i64(T * P1)
    low_s = np.full(T, np.inf, np.float32)
    low_i = np.full(T, -1, np.int64)
    low_j = np.full(T, -1, np.int64)
    pos_keys, neg_keys = [], []
    fneg = np.zeros((0, 3)); fpos = np.zeros((0, 3))
    pa, pb_, pmed, pmax, pn = [], [], [], [], []

    def merge_top(cur, sim, i, j, largest):
        new = np.stack([sim.cpu().numpy().astype(np.float64), i.cpu().numpy(), j.cpu().numpy()], 1)
        allr = np.concatenate([cur, new])
        order = np.argsort(-allr[:, 0] if largest else allr[:, 0], kind="stable")[:n_fail]
        return allr[order]

    for t0, t1 in _groups(starts, N, block_pairs):
        r0, r1 = int(starts[t0]), int(starts[t1])
        S = E[r0:r1] @ E[r0:].T
        mask = _upper(S)
        ri, cj = torch.nonzero(mask, as_tuple=True)
        s_all = S[ri, cj]
        del S, mask
        grp_neg_s, grp_neg_a, grp_neg_b = [], [], []
        for c0 in range(0, len(s_all), CHUNK):
            s = s_all[c0:c0 + CHUNK]
            i = ri[c0:c0 + CHUNK] + r0
            j = cj[c0:c0 + CHUNK] + r0
            ti, tj = trk[i], trk[j]
            is_neg = ti != tj
            ty = is_neg.long()
            # ---- joint cell
            dp = (pos[i] - pos[j]).norm(dim=1)
            pb = torch.where(pos_ok[i] & pos_ok[j], _bucket(dp, bins.pos_edges), torch.full_like(ti, P1 - 1))
            d = (az[i] - az[j]).abs()
            d = torch.where(d > 180, 360 - d, d)
            ab = torch.where(torch.isnan(d), torch.full_like(ti, A1 - 1), _bucket(torch.nan_to_num(d), bins.az_edges))
            ob = occ[i] + occ[j]
            if has_kp:
                inter = (kp[i] & kp[j]).sum(1)
                union = (kp[i] | kp[j]).sum(1)
                known = kp_ok[i] & kp_ok[j] & (union > 0)
                iou = inter.float() / union.clamp(min=1).float()
                kb = torch.where(known, _bucket(iou, bins.kp_edges), torch.full_like(ti, K1 - 1))
                kmin = torch.minimum(kp_n[i], kp_n[j]).float()
                kmb = torch.where(kp_ok[i] & kp_ok[j], _bucket(kmin, bins.kpmin_edges), torch.full_like(ti, KM1 - 1))
            else:
                kb = torch.full_like(ti, K1 - 1)
                kmb = torch.full_like(ti, KM1 - 1)
            cell = ((pb * A1 + ab) * O3 + ob) * K1 + kb
            ct = cell * 2 + ty
            # ---- weights: object-balanced
            w = torch.where(is_neg, 1.0 / (cn[ti] * cn[tj]), w_pos_obj[ti])
            sc = s.clamp(-1, 1)
            cb = ((sc + 1) * (N_HIST / 2)).long().clamp_(0, N_HIST - 1)
            H += torch.bincount(ct * N_HIST + cb, minlength=NC * 2 * N_HIST)
            HB += torch.bincount(ct * N_HIST + cb, weights=w, minlength=NC * 2 * N_HIST)
            fi = ty * N_FINE + _fine_index(s)
            FI += torch.bincount(fi, minlength=2 * N_FINE)
            FB += torch.bincount(fi, weights=w, minlength=2 * N_FINE)
            s64 = sc.double()
            M1 += torch.bincount(ct, weights=s64, minlength=NC * 2)
            M2 += torch.bincount(ct, weights=s64 * s64, minlength=NC * 2)
            M1B += torch.bincount(ct, weights=w * s64, minlength=NC * 2)
            M2B += torch.bincount(ct, weights=w * s64 * s64, minlength=NC * 2)
            NB += torch.bincount(ct, weights=w, minlength=NC * 2)
            KH += torch.bincount((kmb * 2 + ty) * N_HIST + cb, minlength=KM1 * 2 * N_HIST)
            for q, t in enumerate(thr):
                ge = s >= t
                CNT[q] += torch.bincount(ct[ge], minlength=NC * 2)
                CNTB[q] += torch.bincount(ct[ge], weights=w[ge], minlength=NC * 2)
                KCNT[q] += torch.bincount((kmb * 2 + ty)[ge], minlength=KM1 * 2)
            # ---- distinct objects / object pairs per cell
            isp = ~is_neg
            pos_keys.append(torch.unique(cell[isp] * T + ti[isp]).cpu().numpy())
            neg_keys.append(torch.unique((cell[is_neg] * T + ti[is_neg]) * T + tj[is_neg]).cpu().numpy())
            # ---- per object (positives)
            sp, tip, pbp = s[isp], ti[isp], pb[isp]
            if len(sp):
                TARD += torch.bincount(tip * P1 + pbp, minlength=T * P1)
                for q, t in enumerate(thr):
                    TARN[q] += torch.bincount((tip * P1 + pbp)[sp >= t], minlength=T * P1)
                # lowest positive of each object in this chunk: sort by (object, similarity), take the first of each object
                order = torch.argsort(tip.double() + (sp.double().clamp(-1, 1) + 1) / 4)
                ts = tip[order]
                first = torch.ones_like(ts, dtype=torch.bool)
                first[1:] = ts[1:] != ts[:-1]
                sel = order[first]
                objs, vals = ts[first].cpu().numpy(), sp[sel].cpu().numpy()
                ic, jc = i[isp][sel].cpu().numpy(), j[isp][sel].cpu().numpy()
                better = vals < low_s[objs]
                low_s[objs[better]], low_i[objs[better]], low_j[objs[better]] = vals[better], ic[better], jc[better]
                # failure candidates: lowest positives with delta position < 1 m (first known bin)
                near = pbp == 0
                if near.any():
                    fpos = merge_top(fpos, sp[near], i[isp][near], j[isp][near], False)
            if is_neg.any():
                fneg = merge_top(fneg, s[is_neg], i[is_neg], j[is_neg], True)
                grp_neg_s.append(s[is_neg]); grp_neg_a.append(ti[is_neg]); grp_neg_b.append(tj[is_neg])
        # ---- exact median / max per object pair (all of a pair's crop pairs are inside this group)
        if grp_neg_s:
            ns, na, nb = torch.cat(grp_neg_s), torch.cat(grp_neg_a), torch.cat(grp_neg_b)
            key = na * T + nb
            order = torch.argsort(key.double() + (ns.double().clamp(-1, 1) + 1) / 4)  # by pair, then similarity
            skey, ss = key[order], ns[order]
            uk, cnts = torch.unique_consecutive(skey, return_counts=True)
            st = torch.cumsum(cnts, 0) - cnts
            med = (ss[st + (cnts - 1) // 2] + ss[st + cnts // 2]) / 2
            pa.append((uk // T).cpu().numpy()); pb_.append((uk % T).cpu().numpy())
            pmed.append(med.cpu().numpy()); pmax.append(ss[st + cnts - 1].cpu().numpy()); pn.append(cnts.cpu().numpy())
        del s_all, ri, cj

    # ---------------- finalise: sparse over non-empty cells
    cells_n = (H.view(NC, 2, N_HIST).sum(2)).cpu().numpy()
    keep = np.nonzero(cells_n.sum(1) > 0)[0]
    kt = torch.as_tensor(keep, device=dev)
    cpu = lambda t: t.cpu().numpy()
    cat = lambda xs, dt, shape=(0,): np.concatenate(xs) if xs else np.zeros(shape, dt)
    pk = np.unique(cat(pos_keys, np.int64)); nk = np.unique(cat(neg_keys, np.int64))
    acc = {
        "cell_ids": keep.astype(np.int32),
        "hist": cpu(H.view(NC, 2, N_HIST)[kt]), "hist_bal": cpu(HB.view(NC, 2, N_HIST)[kt]),
        "cnt": np.stack([cpu(c.view(NC, 2)[kt]) for c in CNT], -1), "cnt_bal": np.stack([cpu(c.view(NC, 2)[kt]) for c in CNTB], -1),
        "m1": cpu(M1.view(NC, 2)[kt]), "m2": cpu(M2.view(NC, 2)[kt]), "m1_bal": cpu(M1B.view(NC, 2)[kt]),
        "m2_bal": cpu(M2B.view(NC, 2)[kt]), "n_bal": cpu(NB.view(NC, 2)[kt]),
        "fine": cpu(FI.view(2, N_FINE)), "fine_bal": cpu(FB.view(2, N_FINE)),
        "pos_keys": np.stack([pk // T, pk % T], 1), "neg_keys": np.stack([nk // (T * T), (nk // T) % T, nk % T], 1),
        "obj_n": cpu(counts), "obj_tar_num": np.stack([cpu(x).reshape(T, P1) for x in TARN]), "obj_tar_den": cpu(TARD).reshape(T, P1),
        "obj_low_sim": np.where(np.isinf(low_s), np.nan, low_s), "obj_low_i": low_i, "obj_low_j": low_j,
        "pair_a": cat(pa, np.int64), "pair_b": cat(pb_, np.int64), "pair_med": cat(pmed, np.float32),
        "pair_max": cat(pmax, np.float32), "pair_n": cat(pn, np.int64),
        "fail_neg": fneg, "fail_pos": fpos,
        "kpmin_hist": cpu(KH.view(KM1, 2, N_HIST)), "kpmin_cnt": np.stack([cpu(c.view(KM1, 2)) for c in KCNT], -1),
        "thresholds": np.asarray(thr, np.float64),
        "n_crops": np.int64(N), "n_objects": np.int64(T), "bins_sha256": np.asarray(bins.sha256),
    }
    return acc
