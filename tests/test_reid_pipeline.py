"""End to end through the CLIs with the debug encoder: split -> extract -> choose mode -> evaluate -> report -> compare."""
import json

import pandas as pd
import pytest

import compare_models
import evaluate
import extract_templates
import make_split
from reid_eval.common import read_json
from reid_eval.synthetic import make_fake_dataset


def test_full_pipeline_cli(tmp_path):
    ds = tmp_path / "d"
    make_fake_dataset(ds, n_videos=20, n_sites=8, tracklets=(4, 6), crops=(12, 16), seed=7, keypoints=5)
    split = tmp_path / "split.json"
    make_split.main(["--data", str(ds), "--n-val", "5", "--n-test", "10", "--out", str(split)])
    ext = lambda *x: extract_templates.main(["--model", "debug_colorgrid", "--split", str(split), "--data", str(ds),
                                             "--out", str(tmp_path / "t"), "--num-workers", "0", *x])
    assert ext("--preproc", "letterbox", "unpad_stretch") == 0
    dirs = [str(tmp_path / "t" / f"debug_colorgrid__{m}") for m in ("letterbox", "unpad_stretch")]
    assert evaluate.main(["--choose-mode", *dirs, "--split", str(split), "--data", str(ds), "--out", str(tmp_path / "res"),
                          "--device", "cpu"]) == 0
    sel = read_json(tmp_path / "res" / "debug_colorgrid__mode_selection_v1.json")
    assert sel["chosen"] in ("letterbox", "unpad_stretch") and sel["chosen_on"] == "validation only"

    res = []
    for d in dirs:
        assert evaluate.main(["--templates", d, "--split", str(split), "--data", str(ds), "--out", str(tmp_path / "res"),
                              "--device", "cpu", "--no-per-subset"]) == 0
    base = tmp_path / "res"
    res = sorted(base.glob("debug_colorgrid__*/v1"))
    assert len(list(base.glob("debug_colorgrid__*/v1__plain"))) == 0   # only with --plain
    assert len(res) == 2
    for r in res:
        text = (r / "report.md").read_text()
        for needle in ("## TAR at fixed FAR", "### delta position", "## Heatmaps", "## Per site", "## Per video", "## Failures", "## ROC"):
            assert needle in text
        assert (r / "figures" / "curve_delta_azimuth.png").exists() and (r / "figures" / "heatmap_delta_position_x_delta_azimuth__tar_0.1pct.png").exists()
    # the report can be rebuilt from the saved files alone
    (res[0] / "report.md").unlink()
    assert evaluate.main(["--report-only", str(res[0])]) == 0 and (res[0] / "report.md").exists()
    # paired comparison
    out = tmp_path / "cmp"
    assert compare_models.main(["--a", str(res[0]), "--b", str(res[1]), "--out", str(out)]) == 0
    df = pd.read_csv(str(out) + ".csv")
    assert {"diff_b_minus_a", "ci_lo", "ci_hi"} <= set(df.columns)
    assert "paired bootstrap" in (out.with_suffix(".md")).read_text()
    # comparing a model with itself: zero difference everywhere, CI exactly 0
    compare_models.main(["--a", str(res[0]), "--b", str(res[0]), "--out", str(tmp_path / "self")])
    same = pd.read_csv(tmp_path / "self.csv").dropna(subset=["diff_b_minus_a"])
    assert same.diff_b_minus_a.abs().max() == 0 and same.ci_lo.abs().max() == 0 and not same.ci_excludes_0.any()


def test_run_full_eval_one_command(tmp_path, capsys):
    import run_full_eval
    ds = tmp_path / "d"
    make_fake_dataset(ds, n_videos=14, n_sites=7, tracklets=(4, 5), crops=(12, 14), seed=9)
    args = ["--data", str(ds), "--model", "debug_colorgrid", "--split", str(tmp_path / "s.json"), "--templates", str(tmp_path / "t"),
            "--out", str(tmp_path / "r"), "--n-val", "3", "--n-test", "6", "--num-workers", "0", "--device", "cpu"]
    assert run_full_eval.main(args) == 0
    reports = list((tmp_path / "r").glob("debug_colorgrid__*/v1/report.md"))
    assert len(reports) == 1 and "DONE" in capsys.readouterr().out
    assert run_full_eval.main(args) == 0  # re-run: split reused, templates skipped
    assert len(list((tmp_path / "r").glob("debug_colorgrid__*/v1__plain/report.md"))) == 1   # both variants by default
    assert len(list((tmp_path / "r").glob("debug_colorgrid__*/v1__site-gallery/report.md"))) == 1   # site matching runs by default
    allm = list((tmp_path / "r").glob("debug_colorgrid__*/v1__all_methods/report.md"))                # and the final comparison
    assert len(allm) == 1 and "Site-averaged results" in allm[0].read_text() and (allm[0].parent / "figures" / "roc_mean_over_sites.png").exists()
