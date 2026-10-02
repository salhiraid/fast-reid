"""threshold_comparison.csv/md: every method incl. the global-threshold rows, recall / precision / accuracy / threshold at every FAR,
complete whatever the order the variants were run in; precision and global-oracle checked against brute force."""
import numpy as np
import pandas as pd
import pytest

from reid_eval import evaluation
from reid_eval.common import read_json
from test_reid_all_methods import BINS, FAR, built, setup  # noqa: F401  (fixtures: 4-object fake world, every method already run)
from test_reid_site_gallery import counts
from test_reid_thresholds import neg_hist, thr_for, video_pairs

# the header the user asked to keep, in this order
PASTED = ["variant", "folder", "videos evaluated", "pos pairs", "neg pairs", "TAR @ FAR 0.1%", "TAR @ FAR 1%", "TAR @ FAR 2%", "TAR @ FAR 5%",
          "TAR @ FAR 10%", "measured FAR @ FAR 0.1%", "measured FAR @ FAR 1%", "measured FAR @ FAR 2%", "measured FAR @ FAR 5%",
          "measured FAR @ FAR 10%", "balanced acc @ threshold 0.5"]
PCT = ["0.1%", "1%", "2%", "5%", "10%"]


def test_comparison_has_the_pasted_header_then_the_new_columns(built):
    model_dir, _ = built
    df = pd.read_csv(model_dir / "v1__threshold_comparison.csv")
    assert list(df.columns[:len(PASTED)]) == PASTED                           # the original columns, same names, same order
    for kind in ("recall", "precision", "pooled precision", "accuracy", "pooled accuracy", "balanced acc", "threshold"):
        for p in PCT:
            assert f"{kind} @ FAR {p}" in df.columns, (kind, p)
    for c in ("accuracy @ threshold 0.5", "precision @ threshold 0.5", "recall @ threshold 0.5", "measured FAR @ threshold 0.5", "unit", "threshold basis"):
        assert c in df.columns, c
    assert (df["recall @ FAR 1%"] == df["TAR @ FAR 1%"]).all()                # recall is the TAR
    variants = list(df.variant)
    assert variants[0].startswith("global threshold from validation")          # the global-threshold row (matching within the video) is there
    assert any("global threshold ORACLE" in v for v in variants) and any("site matching: gallery" in v for v in variants)
    assert len(df) == 8 and df.folder.is_unique


def test_global_row_is_exact_and_precision_matches_brute_force(setup, built):
    ds, split_p, split, tdir, tmp, solo, infos = setup
    model_dir, _ = built
    df = pd.read_csv(model_dir / "v1__threshold_comparison.csv").set_index("folder")
    row = df.loc["v1__plain"]
    th = read_json(model_dir / "v1__plain" / "thresholds.json")
    assert [row[f"threshold @ FAR {p}"] for p in PCT] == pytest.approx(th["thresholds"][:5])      # exact thresholds of the global evaluation
    tp = fp = p_ = n_ = 0
    pooled = {}
    for name, thr in zip(th["names"], th["thresholds"]):
        tp = fp = p_ = n_ = 0
        for vid in split["test"]:
            s, ta, tb, _ = video_pairs(tdir, vid)
            a, b, c, d = counts(s, ta == tb, thr)
            tp += a; p_ += b; fp += c; n_ += d
        pooled[name] = (tp / (tp + fp) if tp + fp else np.nan, (tp + n_ - fp) / (p_ + n_), tp / p_)
    for name, p in zip(["0.1pct", "1pct", "2pct", "5pct", "10pct"], PCT):
        prec, acc, rec = pooled[name]
        assert row[f"pooled precision @ FAR {p}"] == pytest.approx(prec, abs=5e-4)
        assert row[f"pooled accuracy @ FAR {p}"] == pytest.approx(acc, abs=5e-4)
    assert row["pooled precision @ threshold 0.5"] == pytest.approx(pooled["th0.5"][0], abs=5e-4) or np.isnan(pooled["th0.5"][0])
    assert row["pooled accuracy @ threshold 0.5"] == pytest.approx(pooled["th0.5"][1], abs=5e-4)


def test_per_unit_rows_report_the_median_threshold(built):
    model_dir, _ = built
    df = pd.read_csv(model_dir / "v1__threshold_comparison.csv").set_index("folder")
    units = pd.read_csv(model_dir / "v1__thr-site" / "thresholds_per_unit.csv")
    units = units[units.status == "used"]
    assert df.loc["v1__thr-site", "threshold @ FAR 1%"] == pytest.approx(units["thr_1pct"].median(), abs=1e-9)
    assert "median" in df.loc["v1__thr-site", "threshold basis"] and "exact" in df.loc["v1__plain", "threshold basis"]
    assert df.loc["v1__site-gallery", "unit"] == "site" and df.loc["v1__thr-video", "unit"] == "video"


def test_global_oracle_is_one_threshold_forcing_the_pooled_far(setup, built):
    ds, split_p, split, tdir, tmp, solo, infos = setup
    model_dir, _ = built
    d = model_dir / "v1__thr-global-oracle"
    units = pd.read_csv(d / "thresholds_per_unit.csv")
    assert units["thr_1pct"].nunique() == 1 and len(units) == len(split["test"])               # the same threshold for every video
    allneg = sum(neg_hist(video_pairs(tdir, v)[0][video_pairs(tdir, v)[1] != video_pairs(tdir, v)[2]]) for v in split["test"])
    assert units["thr_1pct"].iloc[0] == pytest.approx(thr_for(allneg, 0.01), abs=1e-9)
    h = read_json(d / "summary.json")["headline"]["pooled"]
    assert h["far_at_1pct"]["value"] == pytest.approx(0.01, rel=0.05) and h["far_at_5pct"]["value"] == pytest.approx(0.05, rel=0.05)
    assert "ORACLE" in (d / "report.md").read_text() and "global threshold on all test videos" in (d / "report.md").read_text()
    # honest global (validation) vs oracle global: the oracle is tuned on the measured pairs
    g = read_json(model_dir / "v1__plain" / "summary.json")["headline"]["pooled"]
    assert abs(h["far_at_1pct"]["value"] - 0.01) <= abs(g["far_at_1pct"]["value"] - 0.01) + 1e-9


def test_operating_points_table_in_every_report(built):
    model_dir, _ = built
    for folder in ("v1__plain", "v1__thr-site", "v1__site-gallery", "v1__thr-global-oracle"):
        text = (model_dir / folder / "report.md").read_text()
        assert "## Operating points" in text and "| operating point | threshold | measured FAR | recall (TAR) | precision | accuracy | balanced accuracy |" in text
        assert "FAR 0.1%" in text and "threshold 0.5" in text
    assert "median [min, max]" in (model_dir / "v1__thr-site" / "report.md").read_text()       # per-unit thresholds: a range, not one value
    allm = (model_dir / "v1__all_methods" / "report.md").read_text()
    assert "Precision at every operating point" in allm and "Threshold (cosine similarity) at each operating point" in allm
    s = pd.read_csv(model_dir / "v1__all_methods" / "all_methods_summary.csv")
    assert {"prec_at_1pct", "threshold_at_1pct", "acc_at_0.1pct", "prec_at_th0.5"} <= set(s.columns)


def test_comparison_is_complete_in_run_full_eval_order(setup, tmp_path):
    """run_full_eval runs the per-unit variants BEFORE the global one: the file must still contain the global row at the end."""
    ds, split_p, split, tdir, tmp, solo, infos = setup
    res = tmp_path / "r"
    evaluation.run_threshold_modes(tdir, split_p, ds, BINS, res, ["video", "site", "global-oracle"], device="cpu", verbose=False, per_subset=False, site_queries=0)
    evaluation.run_site_gallery(tdir, split_p, ds, BINS, res, ["site-gallery"], device="cpu", verbose=False, per_subset=False)
    before = pd.read_csv(res / "fake__letterbox" / "v1__threshold_comparison.csv")
    assert not any(v.startswith("global threshold from validation") for v in before.variant)      # not there yet...
    assert any("site matching" in v for v in before.variant)                                       # ...but site matching IS (was missing before)
    evaluation.run(tdir, split_p, ds, BINS, res, device="cpu", verbose=False, plain=True, per_subset=False)
    after = pd.read_csv(res / "fake__letterbox" / "v1__threshold_comparison.csv")
    assert after.variant.iloc[0].startswith("global threshold from validation") and len(after) == len(before) + 1
    assert list(after.columns[:len(PASTED)]) == PASTED
