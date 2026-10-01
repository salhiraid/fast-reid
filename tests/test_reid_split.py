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
