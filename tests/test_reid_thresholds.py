"""Threshold per video / per site (held-out and oracle), checked against brute force on the raw template embeddings."""
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from make_split import main as make_split
from reid_data import group_by_video, load_dataset_with_info
from reid_eval import evaluation
from reid_eval import metrics as M
from reid_eval.accumulate import assign_folds
from reid_eval.bins import Bins
from reid_eval.common import read_json, sha256_file
from reid_eval.synthetic import make_fake_dataset, make_fake_templates
from reid_eval.templates import load_video_npz

BINS = "configs/bins_v3.yaml"
bins = Bins.load(BINS)
Q1 = bins.op_names.index("1pct")


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("thr")
    ds = tmp / "d"
    make_fake_dataset(ds, n_videos=30, n_sites=4, tracklets=(6, 8), crops=(14, 20), seed=21)
    split_p = tmp / "split.json"
    make_split(["--data", str(ds), "--n-val", "6", "--n-test", "18", "--out", str(split_p)])
    split = read_json(split_p)
    lonely = split["test"][0]                                   # the only test video of its own site (crops unchanged: fingerprint ok)
    m = ds / "videos" / lonely / "meta.json"
    meta = json.loads(m.read_text()); meta["site"] = "lonely_site"; m.write_text(json.dumps(meta))
    recs, _ = load_dataset_with_info(ds, split["validation"] + split["test"], verbose=False)
    tdir = tmp / "t" / "fake__letterbox"
    make_fake_templates(tdir, split, group_by_video(recs), sha256_file(split_p), noise=0.25, dim=16, seed=5)
    return ds, split_p, split, tdir, tmp, lonely


def neg_hist(sims):
    h = np.bincount(np.clip(((sims + 1) * 1000).astype(int), 0, 1999), minlength=2000)
    return h


def video_pairs(tdir, vid):
    t = load_video_npz(tdir / "test" / f"{vid}.npz")
    E = t["emb"].astype(np.float64)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    trk_names = sorted(set(t["tracklet_id"].tolist()), key=lambda x: list(dict.fromkeys(t["tracklet_id"].tolist())).index(x))
    trk = np.array([trk_names.index(x) for x in t["tracklet_id"]])
    S = E @ E.T
    iu, ju = np.triu_indices(len(E), 1)
    return S[iu, ju], trk[iu], trk[ju], len(trk_names)


def thr_for(hist, target):
    return M.threshold_for_far(hist, target)


def counts(s, same, thr):
    return int(((s >= thr) & same).sum()), int(same.sum()), int(((s >= thr) & ~same).sum()), int((~same).sum())


@pytest.fixture(scope="module")
def held_out(setup):
    ds, split_p, split, tdir, tmp, lonely = setup
    return evaluation.run_threshold_modes(tdir, split_p, ds, BINS, tmp / "res", ["video", "site"], device="cpu", verbose=False)


def test_per_video_held_out_matches_brute_force(setup, held_out):
    ds, split_p, split, tdir, tmp, lonely = setup
    out = held_out["video"]
    assert out.name == "v1__thr-video"
    units = pd.read_csv(out / "thresholds_per_unit.csv")
    tp = p = fp = n = 0
    n_pairs = 0
    for vid in split["test"]:
        s, ta, tb, T = video_pairs(tdir, vid)
        if T < 4:
            assert units[units.unit == vid].status.iloc[0].startswith("skipped")
            continue
        fold = assign_folds(vid, T, 0)
        for f in (0, 1):
            other = (fold[ta] == 1 - f) & (fold[tb] == 1 - f)           # calibration: negatives among the OTHER fold's objects
            thr = thr_for(neg_hist(s[other & (ta != tb)]), 0.01)
            row = units[(units.unit == vid) & (units.fold == f)].iloc[0]
            assert row["thr_1pct"] == pytest.approx(thr, abs=1e-9)
            assert row.n_calibration_negatives == int((other & (ta != tb)).sum())
            mine = (fold[ta] == f) & (fold[tb] == f)                    # evaluation: pairs inside this fold only
            a, b, c, d = counts(s[mine], (ta == tb)[mine], row["thr_1pct"])
            tp += a; p += b; fp += c; n += d
            n_pairs += int(mine.sum())
    h = read_json(out / "summary.json")["headline"]["pooled"]
    sup = read_json(out / "summary.json")["headline"]["support"]
    assert sup["n_pos"] + sup["n_neg"] == n_pairs                        # no cross-fold pair was evaluated
    assert h["tar_at_1pct"]["value"] == pytest.approx(tp / p, abs=3 / p)
    assert h["far_at_1pct"]["value"] == pytest.approx(fp / n, abs=3 / n)
    assert 0.5 < h["tar_at_1pct"]["value"] < 1 and 0.003 < h["far_at_1pct"]["value"] < 0.03     # measured, not forced
    assert h["acc_at_th0.5"]["value"] > 0.5 and "th0.5" in bins.op_names


def test_per_site_leave_one_video_out_matches_brute_force(setup, held_out):
    ds, split_p, split, tdir, tmp, lonely = setup
    out = held_out["site"]
    infos = load_dataset_with_info(ds, split["test"], verbose=False)[1]
    site = {v: infos[v].site for v in split["test"]}
    pairs = {v: video_pairs(tdir, v) for v in split["test"]}
    own = {v: neg_hist(pairs[v][0][pairs[v][1] != pairs[v][2]]) for v in split["test"]}
    units = pd.read_csv(out / "thresholds_per_unit.csv")
    tp = p = fp = n = 0
    for v in split["test"]:
        others = [u for u in split["test"] if site[u] == site[v] and u != v]
        row = units[units.unit == v].iloc[0]
        if not others:
            assert row.status.startswith("skipped") and site[v] == "lonely_site"
            continue
        thr = thr_for(sum(own[u] for u in others), 0.01)
        assert row["thr_1pct"] == pytest.approx(thr, abs=1e-9)
        s, ta, tb, _ = pairs[v]
        a, b, c, d = counts(s, ta == tb, thr)
        tp += a; p += b; fp += c; n += d
    summ = read_json(out / "summary.json")
    h = summ["headline"]["pooled"]
    assert h["tar_at_1pct"]["value"] == pytest.approx(tp / p, abs=3 / p)
    assert h["far_at_1pct"]["value"] == pytest.approx(fp / n, abs=3 / n)
    assert summ["n_videos"] == len(split["test"]) - 1 and summ["threshold_units_skipped"][0]["unit"] == split["test"][0]
    pv = pd.read_csv(out / "per_video.csv")
    assert split["test"][0] not in set(pv.video_id) and "lonely_site" not in set(pd.read_csv(out / "per_site.csv").site)


def test_variant_comparison_table(setup, held_out):
    ds, split_p, split, tdir, tmp, lonely = setup
    cmp = tmp / "res" / "fake__letterbox" / "v1__threshold_comparison.csv"
    df = pd.read_csv(cmp)
    assert len(df) == 2 and "TAR @ FAR 1%" in df.columns and "measured FAR @ FAR 1%" in df.columns
    text = (tmp / "res" / "fake__letterbox" / "v1__threshold_comparison.md").read_text()
    assert "do not evaluate exactly the same pairs" in text and "held out" in text


def test_reports_label_the_protocol(setup, held_out):
    for mode, out in held_out.items():
        text = (out / "report.md").read_text()
        assert "threshold protocol" in text and "HELD-OUT" in text and "ORACLE" not in text
        assert (out / "per_site").is_dir() and (out / "per_video").is_dir() and (out / "figures" / "roc.png").exists()
        if mode == "site":                                                  # the single-video site is listed as skipped, with the reason
            assert "Videos that could not be evaluated" in text and "only test video of its site" in text


def test_oracle_modes_force_far_and_are_labelled(setup):
    ds, split_p, split, tdir, tmp, lonely = setup
    outs = evaluation.run_threshold_modes(tdir, split_p, ds, BINS, tmp / "res_o", ["video-oracle", "site-oracle"], device="cpu",
                                          verbose=False, per_subset=False)
    pv = pd.read_csv(outs["video-oracle"] / "per_video.csv")
    assert len(pv) == len(split["test"])                                     # nothing is skipped: own negatives always exist
    assert np.allclose(pv["far_at_1pct"], 0.01, rtol=0.1)                    # FAR is forced to the target on each video
    ps = pd.read_csv(outs["site-oracle"] / "per_site.csv")
    assert np.allclose(ps["far_at_1pct"], 0.01, rtol=0.1) and len(ps) == len(set(pd.read_csv(outs["site-oracle"] / "per_video.csv").site))
    for mode, d in outs.items():
        text = (d / "report.md").read_text()
        assert "ORACLE evaluation" in text and read_json(d / "summary.json")["oracle"] is True
    held = pd.read_csv(tmp / "res" / "fake__letterbox" / "v1__thr-video" / "per_video.csv")
    assert not np.allclose(held["far_at_1pct"], 0.01, rtol=0.1)              # the held-out FAR is a measurement, not forced


def test_validation_videos_are_not_used(setup, tmp_path):
    ds, split_p, split, tdir, tmp, lonely = setup
    t2 = tmp_path / tdir.name
    shutil.copytree(tdir, t2)
    shutil.rmtree(t2 / "validation")                                         # thresholds here never touch validation data
    evaluation.run_threshold_modes(t2, split_p, ds, BINS, tmp_path / "r", ["video"], device="cpu", verbose=False, per_subset=False)


def test_variant_without_evaluable_units_is_skipped_not_fatal(setup, tmp_path, capsys):
    ds, split_p, split, tdir, tmp, lonely = setup
    infos = load_dataset_with_info(ds, split["test"], verbose=False)[1]
    copy = tmp_path / "d"
    shutil.copytree(ds, copy)
    for i, v in enumerate(split["test"]):                                    # every test video gets its own site
        m = copy / "videos" / v / "meta.json"
        meta = json.loads(m.read_text()); meta["site"] = f"solo{i}"; m.write_text(json.dumps(meta))
    outs = evaluation.run_threshold_modes(tdir, split_p, copy, BINS, tmp_path / "r", ["site", "video"], device="cpu", verbose=False,
                                          per_subset=False)
    assert "site" not in outs and "video" in outs                            # per-site impossible, per-video still runs
    assert "variant skipped" in capsys.readouterr().err
