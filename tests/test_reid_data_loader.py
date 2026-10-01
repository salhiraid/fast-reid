import json
import shutil

import numpy as np
import pytest

from reid_data import load_dataset, load_dataset_with_info
from reid_data.loader import parse_keypoints
from reid_eval.synthetic import make_fake_dataset


@pytest.fixture()
def ds(tmp_path):
    make_fake_dataset(tmp_path, n_videos=4, seed=1)
    return tmp_path


def test_counts_order_and_no_rejected(ds):
    recs, infos = load_dataset_with_info(ds, verbose=False)
    for vid, info in infos.items():
        assert sum(r.video_id == vid for r in recs) == info.n_crops_meta
    assert not any("rejected" in str(r.path.relative_to(ds)).split("/") for r in recs)
    keys = [(r.video_id, r.tracklet_id, r.frame) for r in recs]
    assert keys == sorted(keys)
    assert len({r.crop_uid for r in recs}) == len(recs)
    assert all("\\" not in r.crop_uid for r in recs)           # windows paths normalised
    assert all(r.keypoints_visible is None for r in recs)      # keypoints absent -> None, not "no keypoints visible"


def test_position_and_occlusion_rules(ds):
    for r in load_dataset(ds, verbose=False):
        assert (r.position_xy is None) == (r.distance_m is None)
        assert r.occluded == (r.occlusion_source != "none")


def test_missing_file_and_count_mismatch_fail_loudly(ds):
    recs = load_dataset(ds, verbose=False)
    recs[0].path.unlink()
    with pytest.raises(FileNotFoundError):
        load_dataset(ds, verbose=False)
    shutil.rmtree(ds)  # fresh dataset with a lying n_crops
    make_fake_dataset(ds, n_videos=1)
    p = ds / "videos" / "vid000" / "meta.json"
    m = json.loads(p.read_text()); m["n_crops"] += 1; p.write_text(json.dumps(m))
    with pytest.raises(ValueError):
        load_dataset(ds, verbose=False)


def test_rejected_path_is_refused(ds):
    p = ds / "videos" / "vid000" / "meta.json"
    m = json.loads(p.read_text())
    m["tracks"][0]["crops"][0]["crop_file"] = "rejected/00001/000001.jpg"
    p.write_text(json.dumps(m))
    with pytest.raises(ValueError):
        load_dataset(ds, video_ids=["vid000"], verbose=False)


def test_dataset_is_not_modified(ds):
    before = {p: p.stat().st_mtime_ns for p in ds.rglob("*") if p.is_file()}
    load_dataset(ds, verbose=False)
    assert before == {p: p.stat().st_mtime_ns for p in ds.rglob("*") if p.is_file()}


def test_keypoint_parsing():
    assert parse_keypoints({}) is None
    k = {"keypoints": {"points": [[1, 1, 0.9], [2, 2, 0.1]], "visible": [True, False]}}
    assert parse_keypoints(k).tolist() == [True, False]
    k = {"keypoints": {"points": [[1, 1, 0.9], [2, 2, 0.1]], "visibility_threshold": 0.5}}
    assert parse_keypoints(k).tolist() == [True, False]


def test_layout_written_by_build_reid_crops(tmp_path):
    """All-backslash crop paths (Windows build), null quality flags, no view_* keys without calibration, rejected\\ entries."""
    make_fake_dataset(tmp_path, n_videos=4, seed=5, windows_paths="all", calibration_every=2)
    meta = json.loads((tmp_path / "videos" / "vid001" / "meta.json").read_text())   # uncalibrated video
    c = meta["tracks"][0]["crops"][0]
    assert "\\" in c["crop_file"] and "view_azimuth_deg" not in c and meta["calibration"] == {"available": False}
    assert "\\" in meta["rejected_crops"][0]["crop_file"]
    recs, infos = load_dataset_with_info(tmp_path, verbose=False)
    r = next(r for r in recs if r.video_id == "vid001")
    assert r.azimuth_deg is None and r.position_xy is None and r.view_bin is None   # missing keys -> None, no KeyError
    assert r.crop_uid.startswith("vid001/crops/") and "\\" not in r.crop_uid and r.path.is_file()
    assert infos["vid001"].calibration_available is False and infos["vid000"].calibration_available is True
    assert all(r.quality.get("flags") is None for r in recs)
    assert not any("rejected" in r.crop_uid for r in recs)
