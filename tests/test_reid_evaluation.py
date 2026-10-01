"""Task 5: synthetic evaluation with known similarities (one-hot tracklet + noise)."""
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from make_split import main as make_split
from reid_data import group_by_video, load_dataset_with_info
from reid_eval import evaluation
from reid_eval.aggregate import AXES
from reid_eval.bins import Bins
from reid_eval.common import read_json, sha256_file
from reid_eval.synthetic import make_fake_dataset, make_fake_templates

BINS = "configs/bins_v2.yaml"


def build(tmp_path, equal_crops, keypoints=0, n_videos=36, noise=0.0, dim=16, crops=(6, 16)):
    ds = tmp_path / "d"
    make_fake_dataset(ds, n_videos=n_videos, n_sites=9, tracklets=(6, 8), crops=(10, 10) if equal_crops else crops,
                      seed=11, keypoints=keypoints)
    split_p = tmp_path / "split.json"
    make_split(["--data", str(ds), "--n-val", "10", "--n-test", "12", "--out", str(split_p)])
    split = read_json(split_p)
    recs, infos = load_dataset_with_info(ds, split["validation"] + split["test"], verbose=False)
    tdir = tmp_path / "t" / "fake__letterbox"
    make_fake_templates(tdir, split, group_by_video(recs), sha256_file(split_p), noise=noise, dim=dim, seed=5)
    return ds, split_p, split, tdir


@pytest.fixture(scope="module")
def noiseless(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("noiseless")
    ds, split_p, split, tdir = build(tmp, equal_crops=True, keypoints=6)
    out = evaluation.run(tdir, split_p, ds, BINS, tmp / "res", device="cpu", verbose=False)
    return ds, split, out


@pytest.fixture(scope="module")
def noisy(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("noisy")
    ds, split_p, split, tdir = build(tmp, equal_crops=False, noise=0.25, crops=(25, 40))
    out = evaluation.run(tdir, split_p, ds, BINS, tmp / "res", device="cpu", verbose=False)
    return ds, split, out, tdir, tmp


def test_noise_free_is_perfect(noiseless):
    _, _, out = noiseless
    s = read_json(out / "summary.json")
    for version in ("object_balanced", "pooled"):
        h = s["headline"][version]
        assert h["auc"]["value"] == pytest.approx(1.0, abs=1e-9)
        assert h["tar_at_1pct"]["value"] == pytest.approx(1.0) and h["tar_at_0.1pct"]["value"] == pytest.approx(1.0)
        assert h["far_at_0.1pct"]["value"] == 0.0 and h["eer"]["value"] == pytest.approx(0.0, abs=1e-6)
    assert s["headline"]["video_averaged"]["auc"]["value"] == pytest.approx(1.0)
    assert "ci_lo" in s["headline"]["object_balanced"]["tar_at_0.1pct"]  # confidence intervals present


def test_bin_counts_sum_to_total_pairs(noiseless):
    _, split, out = noiseless
    per_video = pd.read_csv(out / "per_video.csv")
    total = (per_video.n_pos + per_video.n_neg).sum()
    assert total == sum(n * (n - 1) // 2 for n in per_video.n_crops)
    for axis in AXES:
        df = pd.read_csv(out / f"bins_{axis}.csv")
        assert (df.n_pos + df.n_neg).sum() == total, axis
    for f in out.glob("heatmap_*.csv"):
        assert (pd.read_csv(f).eval("n_pos + n_neg")).sum() == total, f.name
    assert (pd.read_csv(out / "cells.csv").eval("n_pos + n_neg")).sum() == total
    # positives are all in `same tracklet` pairs: P = sum C(10,2) per tracklet
    assert per_video.n_pos.sum() == 45 * per_video.n_objects.sum()


def test_keypoint_axis_known_and_unknown(noiseless):
    df = pd.read_csv(noiseless[2] / "bins_keypoint_iou.csv")
    unknown = df[df.bin_keypoint_iou == "unknown"]
    assert unknown.n_pos.iloc[0] + unknown.n_neg.iloc[0] < (df.n_pos + df.n_neg).sum()  # fake keypoints: mostly known
    assert df[df.bin_keypoint_iou != "unknown"][["n_pos", "n_neg"]].to_numpy().sum() > 0
    assert (noiseless[2] / "bins_min_visible_keypoints.csv").exists()


def test_no_keypoints_everything_unknown(noisy):
    df = pd.read_csv(noisy[2] / "bins_keypoint_iou.csv")
    known = df[df.bin_keypoint_iou != "unknown"]
    assert known.n_pos.sum() == 0 and known.n_neg.sum() == 0
    assert df[df.bin_keypoint_iou == "unknown"].n_pos.iloc[0] > 0


def test_pooled_equals_balanced_with_equal_crops(noiseless):
    df = pd.read_csv(noiseless[2] / "bins_delta_position.csv")
    ok = df.n_pos > 0
    for k in ("tar_at_0.1pct", "far_at_0.1pct", "auc"):
        assert np.allclose(df.loc[ok, f"pooled_{k}"], df.loc[ok, f"balanced_{k}"], atol=1e-9, equal_nan=True), k


def test_far_on_validation_matches_target(noisy):
    _, _, out, _, _ = noisy
    s = read_json(out / "summary.json")
    t = read_json(out / "thresholds.json")
    assert t["n_negative_pairs"] > 100_000 and t["n_distinct_object_pairs"] > 0
    assert s["validation"]["pooled_far_at_0.1pct"] == pytest.approx(1e-3, rel=0.1)
    assert s["validation"]["pooled_far_at_1pct"] == pytest.approx(1e-2, rel=0.05)
    # on test the FAR is measured, not forced: close to the target because the cameras are statistically identical
    h = s["headline"]["pooled"]
    assert 2e-4 < h["far_at_0.1pct"]["value"] < 3e-3 and 0.007 < h["far_at_1pct"]["value"] < 0.013
    assert 0.2 < h["tar_at_0.1pct"]["value"] < 1.0 and h["auc"]["value"] < 1.0   # noise makes the problem non-trivial


def test_pooled_vs_balanced_differ_with_unequal_crops(noisy):
    s = read_json(noisy[2] / "summary.json")["headline"]
    assert s["pooled"]["tar_at_0.1pct"]["value"] != s["object_balanced"]["tar_at_0.1pct"]["value"]


def test_thresholds_use_validation_only(noisy, tmp_path):
    ds, split, out, tdir0, tmp = noisy
    tdir = tmp / "copy_for_threshold_test" / tdir0.name       # never destroy the shared fixture
    shutil.copytree(tdir0, tdir)
    bins = Bins.load(BINS)
    recs, _ = load_dataset_with_info(ds, split["validation"] + split["test"], verbose=False)
    by = group_by_video(recs)
    before = evaluation.compute_thresholds(tdir, split, by, bins, "cpu")["thresholds"]
    for vid in split["test"]:  # destroy every test template: thresholds must not care
        (tdir / "test" / f"{vid}.npz").unlink()
    after = evaluation.compute_thresholds(tdir, split, by, bins, "cpu")["thresholds"]
    assert before == after


def test_per_object_per_video_failures_written(noisy):
    out = noisy[2]
    po = pd.read_csv(out / "per_object.csv")
    assert {"lowest_positive_sim", "margin", "n_other_objects_falsely_matched_at_0.1pct"} <= set(po.columns)
    assert len(po) == pd.read_csv(out / "per_video.csv").n_objects.sum()
    neg = pd.read_csv(out / "failures" / "negatives_highest_similarity.csv")
    assert len(neg) == 200 and neg.similarity.is_monotonic_decreasing
    pos = pd.read_csv(out / "failures" / "positives_lowest_similarity_dpos_lt_1m.csv")
    assert len(pos) > 0 and pos.similarity.is_monotonic_increasing and (pos.delta_position_m < 1).all()
    assert list((out / "failures").glob("negatives_highest_similarity_*.png"))
    assert all("/rejected/" not in c for c in neg.crop_a)
    # greyed-out flag and joint-cell file exist
    assert "supported" in pd.read_csv(out / "bins_delta_position.csv").columns


def test_template_mismatch_fails(noisy, tmp_path):
    ds, split, out, tdir0, tmp = noisy
    tdir = tmp_path / tdir0.name
    shutil.copytree(tdir0, tdir)
    from reid_eval.templates import load_video_npz, save_video_npz
    vid = split["validation"][0]
    p = tdir / "validation" / f"{vid}.npz"
    t = load_video_npz(p)
    save_video_npz(p, t["emb"][:-1], t["crop_uid"][:-1], t["tracklet_id"][:-1], t["frame"][:-1])  # one crop missing
    recs, _ = load_dataset_with_info(ds, [vid], verbose=False)
    with pytest.raises(ValueError, match="do not match"):
        evaluation.load_video(tdir, "validation", vid, recs)


# ----------------------------------------------------------------------------- multi-FAR, per site / video, plain variant
FAR_NAMES = ["0.1pct", "1pct", "2pct", "5pct", "10pct"]


def test_multiple_far_operating_points(noisy):
    out = noisy[2]
    t = read_json(out / "thresholds.json")
    s = read_json(out / "summary.json")
    assert t["names"] == FAR_NAMES and s["far_names"] == FAR_NAMES
    assert t["thresholds"] == sorted(t["thresholds"], reverse=True)       # stricter FAR -> higher similarity threshold
    pooled, bal = s["headline"]["pooled"], s["headline"]["object_balanced"]
    tars = [bal[f"tar_at_{n}"]["value"] for n in FAR_NAMES]
    assert tars == sorted(tars) and tars[0] < tars[-1]                    # looser FAR -> higher TAR
    for n, target in zip(FAR_NAMES, (1e-3, 1e-2, 2e-2, 5e-2, 1e-1)):
        assert s["validation"][f"pooled_far_at_{n}"] == pytest.approx(target, rel=0.1)       # set on validation
        assert pooled[f"far_at_{n}"]["value"] == pytest.approx(target, rel=0.35)              # measured on test (not forced)
        assert "ci_lo" in bal[f"tar_at_{n}"]
    df = pd.read_csv(out / "bins_delta_position.csv")
    assert all(f"balanced_tar_at_{n}" in df.columns and f"pooled_far_at_{n}_lo" in df.columns for n in FAR_NAMES)
    for n in ("0.1pct", "1pct"):
        assert (out / "figures" / f"heatmap_delta_position_x_delta_azimuth__tar_{n}.png").exists()


def test_per_site_and_per_video_have_the_same_outputs(noisy):
    out = noisy[2]
    g = read_json(out / "summary.json")
    ps, pv = pd.read_csv(out / "per_site.csv"), pd.read_csv(out / "per_video.csv")
    assert ps.n_videos.sum() == len(pv) == g["n_videos"]
    assert ps.n_pos.sum() == pv.n_pos.sum() == g["headline"]["support"]["n_pos"]
    assert ps.n_crops.sum() == pv.n_crops.sum()
    for report in list(ps.report) + list(pv.report):
        d = (out / report).parent
        assert (d / "report.md").exists() and (d / "roc.csv").exists() and (d / "figures" / "roc.png").exists()
        for axis in ("delta_position", "delta_azimuth", "occlusion", "keypoint_iou"):
            assert (d / f"bins_{axis}.csv").exists() and (d / "figures" / f"curve_{axis}.png").exists()
        assert (d / "figures" / "heatmap_delta_position_x_keypoint_iou__tar_1pct.png").exists()
    assert (out / "figures" / "per_site_tar.png").exists()
    # a site's pooled numbers are exactly the pooled numbers of its videos
    site = ps.iloc[0]
    sub = pv[pv.site == site.site]
    assert site.n_pos == sub.n_pos.sum() and site.n_videos == len(sub)
    one = read_json(out / pv.report.iloc[0].replace("report.md", "summary.json"))
    assert one["n_videos"] == 1 and one["has_ci"] is False and "ci_lo" not in one["headline"]["pooled"]["auc"]
    assert "**not computed here" in (out / pv.report.iloc[0]).read_text()
    assert "per_site/" in (out / "report.md").read_text() and "per_video/" in (out / "report.md").read_text()
    # one top-10 match image per object of every test video, linked from the per-video report
    po = pd.read_csv(out / "per_object.csv")
    for vid, grp in po.groupby("video_id"):
        idx = pd.read_csv(out / "matches" / vid / "index.csv")
        assert sorted(idx.tracklet_id) == sorted(grp.tracklet_id)
        assert all((out / "matches" / vid / f).exists() for f in idx.sheet)
    first = pv.report.iloc[0]
    assert "matches/" in (out / first).read_text()


def test_plain_equals_full_globally_and_ignores_pose_metadata(noisy):
    ds0, split, out, tdir, tmp = noisy
    ds = tmp / "dataset_copy"
    shutil.copytree(ds0, ds)                                   # this test edits meta.json files
    plain = evaluation.run(tdir, tmp / "split.json", ds, BINS, tmp / "res", device="cpu", verbose=False, plain=True)
    assert plain.name == "v1__plain" and out.name == "v1"
    f, p = read_json(out / "summary.json"), read_json(plain / "summary.json")
    assert p["kind"] == "plain" and f["kind"] == "full"
    for version in ("pooled", "object_balanced"):
        for k in [f"tar_at_{n}" for n in FAR_NAMES] + [f"far_at_{n}" for n in FAR_NAMES] + ["auc", "eer"]:
            assert p["headline"][version][k]["value"] == pytest.approx(f["headline"][version][k]["value"], abs=1e-9), (version, k)
    assert read_json(plain / "thresholds.json")["thresholds"] == read_json(out / "thresholds.json")["thresholds"]
    assert not list(plain.glob("bins_*.csv")) and not list(plain.glob("heatmap_*"))        # no difficulty criteria
    assert not (plain / "matches").exists()                                                 # match images: full variant only (default)
    assert (plain / "per_site.csv").exists() and (plain / "per_video.csv").exists() and (plain / "figures" / "roc.png").exists()
    assert "Difficulty axes" not in (plain / "report.md").read_text()
    pos = pd.read_csv(plain / "failures" / "positives_lowest_similarity.csv")
    assert len(pos) == 200                                                                  # all positives, not only delta position < 1 m
    # destroy every position / azimuth / occlusion field: the plain result must not change
    for meta in (ds / "videos").glob("*/meta.json"):
        m = json.loads(meta.read_text())
        for tr in m["tracks"]:
            for c in tr["crops"]:
                c.update(position_road_m=None, position_reliable=False, view_azimuth_deg=None, occluded_annot=True, overlap_by_closer_box=1.0)
        meta.write_text(json.dumps(m))
    plain2 = evaluation.run(tdir, tmp / "split.json", ds, BINS, tmp / "res2", device="cpu", verbose=False, plain=True, per_subset=False)
    p2 = read_json(plain2 / "summary.json")
    assert p2["headline"]["pooled"]["tar_at_1pct"]["value"] == pytest.approx(p["headline"]["pooled"]["tar_at_1pct"]["value"], abs=1e-12)
    full2 = evaluation.run(tdir, tmp / "split.json", ds, BINS, tmp / "res3", device="cpu", verbose=False, per_subset=False)
    unk = pd.read_csv(full2 / "bins_delta_position.csv")
    assert unk[unk.bin_delta_position != "unknown"].n_pos.sum() == 0                          # the full eval does see the change
