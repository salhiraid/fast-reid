"""Accuracy at a fixed threshold (0.5) and at every FAR threshold, checked against a brute-force computation from the templates."""
import numpy as np
import pandas as pd
import pytest

from reid_eval import evaluation
from reid_eval.common import read_json
from reid_eval.templates import load_video_npz
from test_reid_evaluation import build

BINS = "configs/bins_v3.yaml"


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("acc")
    ds, split_p, split, tdir = build(tmp, equal_crops=False, noise=0.25, crops=(15, 22), keypoints=0)
    out = evaluation.run(tdir, split_p, ds, BINS, tmp / "res", device="cpu", verbose=False, per_subset=False, match_sheets=False)
    return split, tdir, out


def brute(split, tdir, thr):
    """TP, FN, FP, TN plus object-balanced weights, over all within-video pairs of the test videos."""
    tot = {"tp": 0, "p": 0, "fp": 0, "n": 0, "tpw": 0.0, "pw": 0.0, "fpw": 0.0, "nw": 0.0}
    for vid in split["test"]:
        t = load_video_npz(tdir / "test" / f"{vid}.npz")
        E = t["emb"].astype(np.float64)
        E /= np.linalg.norm(E, axis=1, keepdims=True)
        S = E @ E.T
        trk = t["tracklet_id"]
        cnt = {x: (trk == x).sum() for x in set(trk)}
        iu, ju = np.triu_indices(len(E), 1)
        s, same = S[iu, ju], trk[iu] == trk[ju]
        w = np.where(same, 1 / (np.array([cnt[x] for x in trk[iu]]) * (np.array([cnt[x] for x in trk[iu]]) - 1) / 2),
                     1 / (np.array([cnt[x] for x in trk[iu]]) * np.array([cnt[x] for x in trk[ju]])))
        acc = s >= thr
        tot["p"] += same.sum(); tot["n"] += (~same).sum()
        tot["tp"] += (acc & same).sum(); tot["fp"] += (acc & ~same).sum()
        tot["pw"] += w[same].sum(); tot["nw"] += w[~same].sum()
        tot["tpw"] += w[acc & same].sum(); tot["fpw"] += w[acc & ~same].sum()
    return tot


def test_accuracy_matches_brute_force_at_every_operating_point(run):
    split, tdir, out = run
    t = read_json(out / "thresholds.json")
    s = read_json(out / "summary.json")
    assert t["names"] == ["0.1pct", "1pct", "2pct", "5pct", "10pct", "th0.5"] and t["thresholds"][-1] == 0.5
    assert s["fixed_names"] == ["th0.5"] and s["far_names"] == t["names"][:-1]
    total = None
    for name, thr in zip(t["names"], t["thresholds"]):
        b = brute(split, tdir, thr)
        total = b["p"] + b["n"]
        acc = (b["tp"] + b["n"] - b["fp"]) / total
        bacc = (b["tp"] / b["p"] + 1 - b["fp"] / b["n"]) / 2
        accw = (b["tpw"] + b["nw"] - b["fpw"]) / (b["pw"] + b["nw"])
        baccw = (b["tpw"] / b["pw"] + 1 - b["fpw"] / b["nw"]) / 2
        pooled, bal = s["headline"]["pooled"], s["headline"]["object_balanced"]
        assert pooled[f"acc_at_{name}"]["value"] == pytest.approx(acc, abs=3 / total), name         # exact counts (<= a float-tie flip)
        assert pooled[f"bacc_at_{name}"]["value"] == pytest.approx(bacc, abs=3 / b["p"]), name
        assert bal[f"acc_at_{name}"]["value"] == pytest.approx(accw, abs=1e-3), name
        assert bal[f"bacc_at_{name}"]["value"] == pytest.approx(baccw, abs=1e-3), name
        # internal consistency of the definitions
        tar, far = pooled[f"tar_at_{name}"]["value"], pooled[f"far_at_{name}"]["value"]
        assert pooled[f"bacc_at_{name}"]["value"] == pytest.approx((tar + 1 - far) / 2, abs=1e-12)
        assert pooled[f"acc_at_{name}"]["value"] == pytest.approx((tar * b["p"] + (1 - far) * b["n"]) / total, abs=1e-9)
        assert "ci_lo" in pooled[f"acc_at_{name}"]                                                  # bootstrap CI over videos
    # the plain accuracy is dominated by the negatives (that is why balanced accuracy is reported too)
    b = brute(split, tdir, 0.5)
    assert b["n"] / (b["p"] + b["n"]) > 0.7


def test_accuracy_curve_and_tables(run):
    split, tdir, out = run
    roc = pd.read_csv(out / "roc.csv")
    assert {"acc_pooled", "bacc_pooled", "acc_balanced", "bacc_balanced"} <= set(roc.columns)
    row = roc.iloc[1500]                                       # threshold exactly 0.5
    assert row.threshold == pytest.approx(0.5)
    s = read_json(out / "summary.json")["headline"]
    assert row.acc_pooled == pytest.approx(s["pooled"]["acc_at_th0.5"]["value"], abs=1e-4)
    assert row.bacc_balanced == pytest.approx(s["object_balanced"]["bacc_at_th0.5"]["value"], abs=1e-3)
    assert roc.acc_pooled.iloc[0] < roc.acc_pooled.max() and roc.bacc_pooled.max() > 0.8   # a real curve: 0 threshold rejects nothing
    for csv in ("per_video.csv", "per_site.csv"):
        df = pd.read_csv(out / csv)
        assert {"acc_at_th0.5", "bacc_at_th0.5", "acc_at_1pct", "bacc_at_0.1pct"} <= set(df.columns)
    df = pd.read_csv(out / "bins_delta_position.csv")
    assert {"pooled_acc_at_th0.5", "balanced_bacc_at_th0.5", "pooled_acc_at_5pct", "pooled_acc_at_th0.5_lo"} <= set(df.columns)
    text = (out / "report.md").read_text()
    assert "## Accuracy" in text and "threshold 0.5" in text and (out / "figures" / "accuracy_vs_threshold.png").exists()
