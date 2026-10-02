"""The universal evaluation kit: in sync with the sources, standalone, generic exporter, multi-model run + comparison tables."""
import os
import shutil
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import run_eval
from make_split import main as make_split
from reid_data import group_by_video, load_dataset_with_info
from reid_eval.common import read_json, sha256_file
from reid_eval.synthetic import make_fake_dataset, make_fake_templates
from reid_eval.universal import find_models, resolve_split, run_all
from tools.build_eval_kit import MARKER, build, kit_files

REPO = Path(__file__).resolve().parent.parent


def test_committed_kit_is_in_sync_with_the_sources(tmp_path):
    fresh = build(tmp_path / "kit")
    committed = REPO / "reid_eval_kit"
    assert (committed / MARKER).exists(), "run: python tools/build_eval_kit.py"
    a = {p.relative_to(fresh).as_posix() for p in fresh.rglob("*") if p.is_file()}
    b = {p.relative_to(committed).as_posix() for p in committed.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    assert a == b, (sorted(a ^ b))
    for rel in a:
        assert (fresh / rel).read_bytes() == (committed / rel).read_bytes(), f"{rel} differs: rerun python tools/build_eval_kit.py"


def test_kit_has_no_model_code_and_refuses_to_overwrite_a_foreign_folder(tmp_path):
    files = kit_files()
    assert not any("encoders" in rel or "fastreid" in rel for rel in files)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "important.txt").write_text("x")
    with pytest.raises(SystemExit):
        build(foreign)
    assert (foreign / "important.txt").exists()


def test_kit_runs_standalone_in_a_clean_copy(tmp_path):
    kit = tmp_path / "somewhere_else" / "my_eval"
    shutil.copytree(REPO / "reid_eval_kit", kit)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    work = tmp_path / "work"
    r = subprocess.run([sys.executable, "selftest.py", "--quick", "--workdir", str(work)], cwd=kit, env=env, capture_output=True, text=True, timeout=1500)
    assert r.returncode == 0 and "SELFTEST OK" in r.stdout, r.stdout[-1500:] + r.stderr[-2500:]
    # the kit used its own packages, nothing of the repository
    r2 = subprocess.run([sys.executable, "-c", "import reid_eval, reid_data, sys; print(reid_eval.__file__, reid_data.__file__); "
                         "import importlib.util as u; print(u.find_spec('fastreid') is None)"], cwd=kit, env=env, capture_output=True, text=True)
    assert str(kit) in r2.stdout and "True" in r2.stdout.splitlines()[-1]
    # the CLI on the self-test templates: list, and comparison from the saved results
    r3 = subprocess.run([sys.executable, "run_eval.py", "--list", str(work / "templates")], cwd=kit, env=env, capture_output=True, text=True)
    assert r3.returncode == 0 and "model_a: letterbox" in r3.stdout and "unpad_stretch" in r3.stdout and "model_c" in r3.stdout
    r4 = subprocess.run([sys.executable, "run_eval.py", "--compare", str(work / "results"), "--reference", "model_a__unpad_stretch"], cwd=kit, env=env,
                        capture_output=True, text=True)
    assert r4.returncode == 0 and (work / "results" / "comparison_v1" / "report.md").exists(), r4.stderr[-1500:]


# ----------------------------------------------------------------------------- in-process: three models, exporter, comparison
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("kit")
    ds = tmp / "dataset"
    make_fake_dataset(ds, n_videos=24, n_sites=4, tracklets=(4, 5), crops=(12, 16), seed=3)
    split_p = tmp / "splits" / "eval_split_v1.json"
    make_split(["--data", str(ds), "--n-val", "6", "--n-test", "14", "--out", str(split_p)])
    split = read_json(split_p)
    by = group_by_video(load_dataset_with_info(ds, split["validation"] + split["test"], verbose=False)[0])
    templates = tmp / "templates"
    for name, mode, noise in (("good", "unpad_stretch", 0.14), ("good", "letterbox", 0.22), ("mid", "letterbox", 0.26), ("weak", "unpad_stretch", 0.34)):
        make_fake_templates(templates / f"{name}__{mode}", split, by, sha256_file(split_p), noise=noise, seed=zlib.crc32((name + mode).encode()) % 999,
                            model_name=name, mode=mode, unique=True)
    return ds, split_p, templates, tmp


@pytest.fixture(scope="module")
def ran(world):
    ds, split_p, templates, tmp = world
    return run_all(ds, templates, tmp / "results", split_p, variants=("plain", "thr-site", "site-gallery"), per_subset=False, match_sheets=False,
                   verbose=False, device="cpu")


def test_find_models_and_mode_choice(world, ran):
    ds, split_p, templates, tmp = world
    found = find_models(templates)
    assert {k: sorted(v) for k, v in found.items()} == {"good": ["letterbox", "unpad_stretch"], "mid": ["letterbox"], "weak": ["unpad_stretch"]}
    assert sorted(ran["results"]) == ["good__unpad_stretch", "mid__letterbox", "weak__unpad_stretch"]     # best mode of `good` chosen on validation
    for label, d in ran["results"].items():
        assert (d / "v1__all_methods" / "report.md").exists() and (d / "v1__plain" / "report.md").exists()
        assert (d / "v1__thr-site" / "summary.json").exists() and (d / "v1__site-gallery" / "summary.json").exists()


def test_model_comparison_tables(world, ran):
    ds, split_p, templates, tmp = world
    rep = Path(ran["comparison"])
    text = rep.read_text()
    assert rep.parent.name == "comparison_v1" and "Models: `good__unpad_stretch`, `mid__letterbox`, `weak__unpad_stretch`" in text
    for needle in ("## Global threshold (validation)", "## Threshold per site (held out", "## Site matching", "**Averaged over sites**",
                   "**Pooled over all pairs", "Paired bootstrap vs `good__unpad_stretch`", "## Models × methods", "TAR @ FAR 1%", "EER"):
        assert needle in text, needle
    summ = pd.read_csv(rep.parent / "models_summary.csv")
    assert set(summ.model) == {"good__unpad_stretch", "mid__letterbox", "weak__unpad_stretch"} and {"global", "thr-site", "site-gallery"} <= set(summ.method)
    g = summ[summ.method == "global"].set_index("model")
    assert g.loc["good__unpad_stretch", "tar_at_1pct"] > g.loc["mid__letterbox", "tar_at_1pct"] > g.loc["weak__unpad_stretch", "tar_at_1pct"]
    assert g.loc["good__unpad_stretch", "eer"] < g.loc["weak__unpad_stretch", "eer"]
    # the best model's cell is the bold one in the TAR @ 1% column of the global table
    rows = [l for l in text.split("## Global threshold (validation)")[1].split("**Pooled")[0].splitlines() if l.startswith("| good__")]
    assert rows and "**" in rows[0] and all("**" not in l for l in text.split("## Global threshold (validation)")[1].split("**Pooled")[0].splitlines()
                                            if l.startswith("| weak__"))
    # numbers in the table are the site averages of the per-model results
    ps = pd.read_csv(ran["results"]["mid__letterbox"] / "v1__plain" / "per_site.csv")
    assert g.loc["mid__letterbox", "tar_at_1pct"] == pytest.approx(ps["tar_at_1pct"].mean(), abs=1e-9)
    # a model can be left out, and the paired section moves to the new reference
    only = run_eval.main(["--compare", str(tmp / "results"), "--models", "mid", "weak", "--reference", "weak__unpad_stretch"])
    assert only == 0 and "Models: `mid__letterbox`, `weak__unpad_stretch`" in rep.read_text()
    assert (rep.parent / "figures" / "roc_global.png").exists() and (rep.parent / "figures" / "tar_vs_target_site-gallery.png").exists()


def test_split_resolution_and_errors(world, tmp_path):
    ds, split_p, templates, tmp = world
    manifests = [read_json(d / "manifest.json") for d in templates.iterdir() if d.is_dir()]
    assert resolve_split(split_p, templates, manifests) == split_p
    other = tmp_path / "other_split.json"
    other.write_text(split_p.read_text().replace('"seed": 0', '"seed": 7'))
    with pytest.raises(ValueError, match="not the one the templates were extracted with"):
        resolve_split(other, templates, manifests)
    shutil.copy(split_p, templates / "eval_split_v1.json")                    # a split next to the templates is found automatically
    try:
        assert resolve_split(None, templates, manifests).name == "eval_split_v1.json"
    finally:
        (templates / "eval_split_v1.json").unlink()
    with pytest.raises(FileNotFoundError):
        resolve_split(None, tmp_path, [{"split_sha256": "x"}])
    with pytest.raises(FileNotFoundError, match="no template folder"):
        find_models(tmp_path)
    with pytest.raises(ValueError, match="not in"):
        run_all(ds, templates, tmp_path / "r", split_p, models=["nope"], verbose=False)


def test_generic_exporter_with_a_user_function(world, tmp_path):
    """Any encoder = one function: the exported templates run through the same evaluation."""
    import export_templates
    ds, split_p, templates, tmp = world
    enc = tmp_path / "my_encoder.py"
    enc.write_text("import numpy as np\n"
                   "def encode(images):\n"
                   "    # mean colour on a 3x3 grid: a stand-in for any model\n"
                   "    out = []\n"
                   "    for im in images:\n"
                   "        g = im.astype(np.float32)\n"
                   "        h, w = g.shape[:2]\n"
                   "        out.append(np.concatenate([g[i*h//3:(i+1)*h//3, j*w//3:(j+1)*w//3].reshape(-1, 3).mean(0) for i in range(3) for j in range(3)]))\n"
                   "    return np.stack(out)\n")
    out = tmp_path / "templates"
    assert export_templates.main(["--data", str(ds), "--split", str(split_p), "--out", str(out), "--model-name", "colorgrid",
                                  "--preproc", "letterbox", "unpad_stretch", "--encoder", f"{enc}:encode", "--batch-size", "7"]) == 0
    man = read_json(out / "colorgrid__unpad_stretch" / "manifest.json")
    assert man["embedding_dim"] == 27 and man["split_sha256"] == sha256_file(split_p) and man["model_name"] == "colorgrid"
    split = read_json(split_p)
    assert len(list((out / "colorgrid__letterbox" / "test").glob("*.npz"))) == len(split["test"])
    # resumable: a second run does not recompute (files untouched)
    f = next((out / "colorgrid__letterbox" / "test").glob("*.npz"))
    m = f.stat().st_mtime_ns
    export_templates.main(["--data", str(ds), "--split", str(split_p), "--out", str(out), "--model-name", "colorgrid", "--preproc", "letterbox",
                           "--encoder", f"{enc}:encode"])
    assert f.stat().st_mtime_ns == m
    res = run_all(ds, out, tmp_path / "res", split_p, variants=("plain",), per_subset=False, match_sheets=False, verbose=False, device="cpu")
    assert len(res["results"]) == 1 and list(res["results"])[0].startswith("colorgrid__")
    assert (list(res["results"].values())[0] / "v1__plain" / "summary.json").exists()
    # a broken encoder is rejected loudly
    bad = tmp_path / "bad_encoder.py"
    bad.write_text("import numpy as np\ndef encode(images):\n    return np.zeros((len(images) + 1, 4))\n")
    with pytest.raises(ValueError, match="encode\\(\\) must return"):
        export_templates.main(["--data", str(ds), "--split", str(split_p), "--out", str(tmp_path / "t2"), "--model-name", "bad", "--encoder", f"{bad}:encode"])
