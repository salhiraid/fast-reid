"""The all-methods comparison: every evaluation method side by side, site-averaged tables, curve figures (no heatmaps)."""
import numpy as np
import pandas as pd
import pytest

import evaluate
from reid_eval import evaluation
from reid_eval.all_methods import build_all_methods, discover
from reid_eval.common import read_json
from test_reid_site_gallery import make_setup  # 21 fake videos, 3 sites, one object-unique embedding per tracklet

BINS = "configs/bins_v3.yaml"
FAR = ["0.1pct", "1pct", "2pct", "5pct", "10pct"]


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    return make_setup(tmp_path_factory.mktemp("allm"), (4, 4))         # 4 objects per video: every protocol can split a video in two folds


@pytest.fixture(scope="module")
def built(setup):
    ds, split_p, split, tdir, tmp, solo, infos = setup
    res = tmp / "allres"
    evaluation.run(tdir, split_p, ds, BINS, res, device="cpu", verbose=False, plain=True, per_subset=False)
    evaluation.run_threshold_modes(tdir, split_p, ds, BINS, res, ["video", "site", "video-oracle", "site-oracle", "global-oracle"], device="cpu",
                                   verbose=False, per_subset=False, site_queries=0)
    evaluation.run_site_gallery(tdir, split_p, ds, BINS, res, ["site-gallery", "site-gallery-oracle"], device="cpu", verbose=False,
                                per_subset=False)
    model_dir = res / "fake__letterbox"
    return model_dir, build_all_methods(model_dir, "v1")


def test_all_methods_are_discovered_and_files_written(built):
    model_dir, report = built
    methods = discover(model_dir, "v1")
    assert [m.key for m in methods] == ["global", "global-oracle", "thr-video", "thr-site", "site-gallery", "thr-video-oracle", "thr-site-oracle", "site-gallery-oracle"]
    out = model_dir / "v1__all_methods"
    assert report == out / "report.md" and (out / "all_methods_summary.csv").exists() and (out / "all_methods_per_site.csv").exists()
    figs = {p.name for p in (out / "figures").glob("*.png")}
    assert {"roc_mean_over_sites.png", "tar_vs_far_target.png", "measured_far_vs_target.png", "accuracy_vs_threshold.png",
            "auc_eer_by_method.png", "per_site_tar_01.png"} <= figs
    assert not any("heatmap" in f for f in figs) and "heatmap_" not in report.read_text()      # curves only: no heatmap figure referenced


def test_tables_hold_the_site_averages(built):
    model_dir, report = built
    out = model_dir / "v1__all_methods"
    summ = pd.read_csv(out / "all_methods_summary.csv").set_index("method")
    for m in discover(model_dir, "v1"):
        ps = pd.read_csv(m.dir / "per_site.csv")
        sa = read_json(m.dir / "summary.json")["headline"]["site_averaged"]
        for n in FAR + ["th0.5"]:
            assert summ.loc[m.key, f"tar_at_{n}"] == pytest.approx(sa[f"tar_at_{n}"]["value"], abs=1e-9)
            assert summ.loc[m.key, f"bacc_at_{n}"] == pytest.approx(sa[f"bacc_at_{n}"]["value"], abs=1e-9)
            assert summ.loc[m.key, f"acc_at_{n}"] == pytest.approx(sa[f"acc_at_{n}"]["value"], abs=1e-9)
        assert summ.loc[m.key, "auc"] == pytest.approx(sa["auc"]["value"]) and summ.loc[m.key, "eer"] == pytest.approx(sa["eer"]["value"])
        # the site average is the plain mean of the per-site numbers (all sites have enough pairs here)
        assert summ.loc[m.key, "tar_at_1pct"] == pytest.approx(ps["tar_at_1pct"].mean(), abs=1e-9)
        assert summ.loc[m.key, "tar_at_1pct_std_over_sites"] == pytest.approx(ps["tar_at_1pct"].std(ddof=1), abs=1e-9)
    long = pd.read_csv(out / "all_methods_per_site.csv")
    assert set(long.method) == set(summ.index) and {"tar_at_1pct", "acc_at_th0.5", "auc", "eer"} <= set(long.columns)
    text = report.read_text()
    for needle in ("Site-averaged results: TAR at each FAR, AUC, EER, accuracy", "Balanced accuracy at every operating point", "Accuracy at every operating point",
                   "Measured FAR at each target", "Pooled over all pairs", "Per site: TAR @ FAR 1%", "Per site: AUC", "Per site: EER",
                   "ORACLE", "cross-video pairs", "Read the comparison with care", "## Curves"):
        assert needle in text, needle
    # a site that a method cannot evaluate is reported as n/a, not silently dropped or zero
    assert "n/a" in text and "solo_site" in text


def test_methods_have_the_expected_protocol_properties(built):
    model_dir, _ = built
    by = {m.key: m for m in discover(model_dir, "v1")}
    assert "solo_site" in set(by["global"].per_site.site) and "solo_site" not in set(by["thr-site"].per_site.site)    # single-video site: no other video
    assert "solo_site" in set(by["site-gallery"].per_site.site) and "solo_site" in set(by["thr-site-oracle"].per_site.site)
    # oracle thresholds force FAR to the target, held-out ones measure it
    assert np.allclose(by["thr-video-oracle"].per_site["far_at_1pct"], 0.01, rtol=0.1)
    assert not np.allclose(by["thr-video"].per_site["far_at_1pct"], 0.01, rtol=0.1)
    # the whole-site gallery has far more negatives than within-video matching (cross-video pairs)
    neg_g = by["site-gallery-oracle"].summary["headline"]["support"]["n_neg"]
    neg_w = by["global"].summary["headline"]["support"]["n_neg"]
    assert neg_g > neg_w


def test_cli_all_methods(built, capsys):
    model_dir, _ = built
    assert evaluate.main(["--all-methods", str(model_dir), "--split-version", "v1"]) == 0
    assert "v1__all_methods" in capsys.readouterr().out
    assert evaluate.main(["--all-methods", str(model_dir), "--split-version", "nope"]) == 1
