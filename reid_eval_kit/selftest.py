#!/usr/bin/env python
"""Self-test of the evaluation kit on synthetic data (no dataset, no model needed): builds a fake dataset, a split and the templates of
three fake models, runs the whole evaluation and the model comparison, and checks the outputs.   python selftest.py [--workdir DIR]"""
import argparse
import shutil
import sys
import tempfile
import zlib
from pathlib import Path

from make_split import main as make_split
from reid_data import group_by_video, load_dataset_with_info
from reid_eval.common import read_json, sha256_file
from reid_eval.synthetic import make_fake_dataset, make_fake_templates
from reid_eval.universal import run_all


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--quick", action="store_true", help="skip per-site/per-video reports and match sheets")
    a = ap.parse_args(argv)
    work = Path(a.workdir or tempfile.mkdtemp(prefix="reid_eval_selftest_"))
    work.mkdir(parents=True, exist_ok=True)
    print(f"workdir: {work}")
    ds, split_p, templates, out = work / "dataset", work / "splits" / "eval_split_v1.json", work / "templates", work / "results"
    make_fake_dataset(ds, n_videos=24, n_sites=4, tracklets=(4, 5), crops=(12, 16), seed=3)
    make_split(["--data", str(ds), "--n-val", "6", "--n-test", "14", "--out", str(split_p)])
    split = read_json(split_p)
    recs, _ = load_dataset_with_info(ds, split["validation"] + split["test"], verbose=False)
    by = group_by_video(recs)
    sha = sha256_file(split_p)
    for name, mode, noise in (("model_a", "letterbox", 0.20), ("model_a", "unpad_stretch", 0.14), ("model_b", "letterbox", 0.26), ("model_c", "unpad_stretch", 0.32)):
        make_fake_templates(templates / f"{name}__{mode}", split, by, sha, noise=noise, seed=zlib.crc32((name + mode).encode()) % 1000, model_name=name, mode=mode, unique=True)
    res = run_all(ds, templates, out, split_p, per_subset=not a.quick, match_sheets=not a.quick, verbose=False, device="cpu")
    assert set(res["results"]) == {"model_a__unpad_stretch", "model_b__letterbox", "model_c__unpad_stretch"}, res["results"]   # best mode chosen for model_a
    for label, d in res["results"].items():
        assert (d / "v1__all_methods" / "report.md").exists(), label
    assert res["comparison"] and Path(res["comparison"]).exists()
    print("results:", out)
    print("comparison:", res["comparison"])
    print("SELFTEST OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
