from pathlib import Path

import numpy as np
import pytest

from reid_eval import accumulate
from reid_eval.accumulate import accumulate_video, fine_histograms
from reid_eval.bins import Bins
from reid_eval_helpers import brute_force, random_meta

BINS = Bins.load(Path(__file__).resolve().parent.parent / "configs/bins_v1.yaml")


@pytest.mark.parametrize("with_kp", [False, True])
@pytest.mark.parametrize("block_pairs,chunk", [(1 << 24, 1 << 22), (900, 257)])  # single block vs many groups/chunks
def test_matches_brute_force(monkeypatch, with_kp, block_pairs, chunk):
    monkeypatch.setattr(accumulate, "CHUNK", chunk)
    rng = np.random.RandomState(4)
    meta = random_meta(rng, n_obj=8, with_kp=with_kp)
    emb = rng.randn(meta.n, 5).astype(np.float32)
    # make some objects tighter so positives and negatives differ
    emb += 2.5 * np.eye(8)[meta.trk][:, :5].astype(np.float32)
    thr = [0.3, 0.6]
    acc = accumulate_video(emb, meta, BINS, thr, device="cpu", block_pairs=block_pairs)
    ref = brute_force(emb.astype(np.float64), meta, BINS, thr)
    NC = BINS.n_cells

    def dense(a):
        out = np.zeros((NC,) + a.shape[1:], a.dtype); out[acc["cell_ids"]] = a; return out
    n_pairs = meta.n * (meta.n - 1) // 2
    assert acc["hist"].sum() == n_pairs == acc["fine"].sum()
    # float32 (GPU/CPU) vs float64 reference can flip a pair lying exactly on a bin edge: allow a handful
    assert np.abs(dense(acc["hist"]) - ref["hist"]).sum() <= 4
    assert np.abs(dense(acc["cnt"]) - ref["cnt"]).sum() <= 4
    assert np.allclose(dense(acc["hist_bal"]).sum(), ref["histb"].sum())
    assert np.abs(dense(acc["cnt_bal"]) - ref["cntb"]).sum() < 0.05
    assert np.abs(acc["fine"] - ref["fine"]).sum() <= 4
    # distinct objects / object pairs per cell
    assert {tuple(k) for k in acc["pos_keys"].tolist()} == set(ref["pos_keys"])
    assert {tuple(k) for k in acc["neg_keys"].tolist()} == set(ref["neg_keys"])
    # exact per-object-pair median / max
    got = {(a, b): (m, mx, n) for a, b, m, mx, n in zip(acc["pair_a"], acc["pair_b"], acc["pair_med"], acc["pair_max"], acc["pair_n"])}
    assert set(got) == set(ref["pairs"])
    for k, v in ref["pairs"].items():
        assert got[k][2] == len(v) and abs(got[k][0] - np.median(v)) < 1e-5 and abs(got[k][1] - max(v)) < 1e-5
    # fine-only pass agrees with the full pass
    assert np.array_equal(fine_histograms(emb, meta, device="cpu", block_pairs=block_pairs), acc["fine"])
    # object-balanced: every object has total positive weight 1, every object pair total negative weight 1
    T = len(meta.tracklet_ids)
    cn = np.bincount(meta.trk)
    assert np.isclose(acc["hist_bal"][:, 0].sum(), (cn >= 2).sum())
    assert np.isclose(acc["hist_bal"][:, 1].sum(), T * (T - 1) / 2)
    # lowest positive per object
    for t in range(T):
        idx = np.nonzero(meta.trk == t)[0]
        if len(idx) < 2:
            continue
        E = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        lows = [(E[i] @ E[j], i, j) for a, i in enumerate(idx) for j in idx[a + 1:]]
        m = min(lows)
        assert abs(acc["obj_low_sim"][t] - m[0]) < 1e-5
        i, j = acc["obj_low_i"][t], acc["obj_low_j"][t]
        assert meta.trk[i] == meta.trk[j] == t and abs(E[i] @ E[j] - m[0]) < 1e-5
