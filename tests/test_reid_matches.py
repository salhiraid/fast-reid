import cv2
import numpy as np
import pandas as pd

from reid_data import load_dataset
from reid_eval.matches import object_sheets
from reid_eval.synthetic import make_fake_dataset


def test_object_sheets_content_and_flags(tmp_path):
    make_fake_dataset(tmp_path / "d", n_videos=1, tracklets=(4, 4), crops=(14, 14), seed=3)
    recs = load_dataset(tmp_path / "d", verbose=False)
    trk = np.array([r.tracklet_id for r in recs])
    ids = sorted(set(trk))
    rng = np.random.RandomState(0)
    emb = np.eye(8)[[ids.index(t) for t in trk]] + 0.3 * rng.randn(len(recs), 8)
    thr = 0.5
    rows = object_sheets(emb, recs, tmp_path / "m", thr, "t(FAR 1%)", topk=10, tile=64, device="cpu")

    assert len(rows) == 4 and sorted(r["tracklet_id"] for r in rows) == ids
    for r in rows:
        im = cv2.imread(str(tmp_path / "m" / r["sheet"]))
        assert im is not None
        assert im.shape[1] == 6 + 11 * (64 + 6)                       # query + 10 matches per row
        assert im.shape[0] == 30 + 3 * (64 + 46) + 6                   # 14 crops -> 13 positives > 10 -> three rows
    df = pd.read_csv(tmp_path / "m" / "index.csv")
    assert len(df) == 4 and (tmp_path / "m" / "index.md").exists()
    E = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    S = E @ E.T
    for r in rows:
        idx = np.nonzero(trk == r["tracklet_id"])[0]
        q = next(i for i, c in enumerate(recs) if c.crop_uid == r["query_crop"])
        assert q in idx
        assert q == idx[np.argmax([(S[i, idx].sum() - 1) / (len(idx) - 1) for i in idx])]       # medoid
        pos = np.setdiff1d(idx, [q])
        assert np.isclose(r["best_positive_sim"], S[q, pos].max()) and np.isclose(r["worst_positive_sim"], S[q, pos].min())
        assert r["false_rejects"] == int((S[q, pos] < thr).sum())
        neg = np.nonzero(trk != r["tracklet_id"])[0]
        assert np.isclose(r["top_negative_sim"], S[q, neg].max()) and r["false_accepts"] == int((S[q, neg] >= thr).sum())
        assert recs[neg[np.argmax(S[q, neg])]].tracklet_id == r["top_negative_object"]
        assert np.isclose(r["separation"], S[q, pos].min() - S[q, neg].max())


def test_object_with_few_crops_has_two_rows_and_single_crop_object_works(tmp_path):
    make_fake_dataset(tmp_path / "d", n_videos=1, tracklets=(3, 3), crops=(5, 5), seed=4)
    recs = load_dataset(tmp_path / "d", verbose=False)
    keep = [r for r in recs if not (r.tracklet_id.endswith("_3") and r.frame > 15)]   # object 3 keeps one crop
    emb = np.random.RandomState(1).randn(len(keep), 6)
    rows = object_sheets(emb, keep, tmp_path / "m", 0.0, "t", topk=10, tile=48, device="cpu")
    one = next(r for r in rows if r["tracklet_id"].endswith("_3"))
    assert one["n_positives"] == 0 and np.isnan(one["separation"])
    im = cv2.imread(str(tmp_path / "m" / rows[0]["sheet"]))
    assert im.shape[0] == 30 + 2 * (48 + 46) + 6                                       # 4 positives <= 10: no 'hardest' row
