import json

import pytest

from make_split import main
from reid_eval.synthetic import make_fake_dataset


@pytest.fixture()
def ds(tmp_path):
    make_fake_dataset(tmp_path / "d", n_videos=24, n_sites=6, tracklets=(3, 4), crops=(6, 8), seed=3, calibration_every=5)
    return tmp_path / "d"


def test_split_disjoint_deterministic_site_disjoint(ds, tmp_path, capsys):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    args = ["--data", str(ds), "--n-val", "4", "--n-test", "10", "--created", "2026-10-01T10:00:00"]
    assert main(args + ["--out", str(a)]) == 0
    assert main(args + ["--out", str(b)]) == 0
    assert a.read_bytes() == b.read_bytes()                       # same seed -> byte-identical
    s = json.loads(a.read_text())
    assert not set(s["validation"]) & set(s["test"])
    assert s["site_disjoint"] is True
    assert not set(s["sites"]["validation"]) & set(s["sites"]["test"])
    assert len(s["test"]) == 10 and len(s["validation"]) >= 4
    assert s["counts"]["test"]["videos"] == 10 and s["counts"]["test"]["crops"] > 0
    uncal = {f"vid{v:03d}" for v in range(24) if v % 5 == 4}
    assert not uncal & (set(s["validation"]) | set(s["test"]))   # ineligible videos never picked
    assert main(args + ["--out", str(a), "--check"]) == 0


def test_refuses_to_overwrite(ds, tmp_path):
    out = tmp_path / "s.json"
    main(["--data", str(ds), "--n-val", "4", "--n-test", "10", "--out", str(out)])
    with pytest.raises(SystemExit):
        main(["--data", str(ds), "--n-val", "4", "--n-test", "10", "--out", str(out)])


def test_loud_warning_when_too_few(ds, tmp_path, capsys):
    main(["--data", str(ds), "--n-val", "4", "--n-test", "500", "--out", str(tmp_path / "s.json")])
    assert "WARNING" in capsys.readouterr().err
    assert json.loads((tmp_path / "s.json").read_text())["warnings"]


def test_rebuilding_summary_does_not_invalidate_split_but_changed_crops_do(ds, tmp_path):
    """build_reid_crops.py rewrites summary.json on every run; only a change of the KEPT crops must block reuse."""
    import extract_templates
    split = tmp_path / "s.json"
    main(["--data", str(ds), "--n-val", "4", "--n-test", "6", "--out", str(split)])
    args = ["--model", "debug_colorgrid", "--split", str(split), "--data", str(ds), "--out", str(tmp_path / "t"),
            "--num-workers", "0", "--preproc", "letterbox"]
    (ds / "summary.json").write_text(json.dumps({"videos": [{"id": "x", "status": "skipped_exists"}]}))   # a resumed build
    assert extract_templates.main(args) == 0
    s = json.loads(split.read_text())
    assert set(s["crops_fingerprint_sha256"]) == {"validation", "test"}
    vid = s["test"][0]                                   # quality_filter.py drops one kept crop in place
    p = ds / "videos" / vid / "meta.json"
    m = json.loads(p.read_text())
    m["tracks"][0]["crops"].pop(); m["n_crops"] -= 1; m["tracks"][0]["n_crops"] -= 1
    p.write_text(json.dumps(m))
    with pytest.raises(SystemExit, match="changed since split"):
        extract_templates.main(args + ["--overwrite"])
