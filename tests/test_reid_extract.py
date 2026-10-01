import json

import numpy as np
import pytest

import extract_templates
from make_split import main as make_split
from reid_data import load_dataset
from reid_eval.synthetic import make_fake_dataset
from reid_eval.templates import load_video_npz


@pytest.fixture()
def setup(tmp_path):
    make_fake_dataset(tmp_path / "d", n_videos=8, n_sites=4, tracklets=(3, 4), crops=(5, 7), seed=2)
    make_split(["--data", str(tmp_path / "d"), "--n-val", "2", "--n-test", "4", "--out", str(tmp_path / "split.json")])
    return tmp_path


def run(setup, *extra):
    return extract_templates.main(["--model", "debug_colorgrid", "--split", str(setup / "split.json"), "--data", str(setup / "d"),
                                   "--out", str(setup / "t"), "--num-workers", "0", "--batch-size", "8", *extra])


def test_extract_resume_and_alignment(setup):
    assert run(setup, "--preproc", "letterbox", "unpad_stretch") == 0
    split = json.loads((setup / "split.json").read_text())
    for mode in ("letterbox", "unpad_stretch"):
        tdir = setup / "t" / f"debug_colorgrid__{mode}"
        m = json.loads((tdir / "manifest.json").read_text())
        assert m["preproc_mode"] == mode and m["split_sha256"] and m["embedding_dim"] == 48
        assert m["counts"]["test"]["videos"] == len(split["test"]) >= 3 and m["checks"]["recheck_min_cos"] > 0.9999
        for s in ("validation", "test"):
            for vid in split[s]:
                t = load_video_npz(tdir / s / f"{vid}.npz")
                recs = load_dataset(setup / "d", [vid], verbose=False)
                assert t["crop_uid"].tolist() == [r.crop_uid for r in recs]
                assert t["emb"].dtype == np.float32 and np.isfinite(t["emb"]).all()
    # the two preprocessing modes really differ (letterbox keeps the black bars)
    a = load_video_npz(setup / "t/debug_colorgrid__letterbox/test" / f"{split['test'][0]}.npz")["emb"]
    b = load_video_npz(setup / "t/debug_colorgrid__unpad_stretch/test" / f"{split['test'][0]}.npz")["emb"]
    assert np.abs(a - b).max() > 0.05

    # a stopped run resumes: delete one file, only that one is recomputed
    tdir = setup / "t/debug_colorgrid__letterbox"
    victim = tdir / "test" / f"{split['test'][1]}.npz"
    keep = tdir / "test" / f"{split['test'][0]}.npz"
    mt = keep.stat().st_mtime_ns
    victim.unlink()
    run(setup, "--preproc", "letterbox")
    assert victim.exists() and keep.stat().st_mtime_ns == mt
    assert not list(tdir.rglob("*.tmp"))


def test_incompatible_manifest_refused(setup):
    run(setup, "--preproc", "letterbox")
    with pytest.raises(SystemExit):
        run(setup, "--preproc", "letterbox", "--flip-tta")
