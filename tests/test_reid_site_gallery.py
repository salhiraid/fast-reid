"""Site matching: gallery = all crops of all test videos of a site, threshold per site (held out / oracle), vs brute force."""
import json

import numpy as np
import pandas as pd
import pytest

from make_split import main as make_split
from reid_data import group_by_video, load_dataset_with_info
from reid_eval import evaluation
from reid_eval.accumulate import assign_folds
from reid_eval.common import read_json, sha256_file, write_json_atomic
from reid_eval.synthetic import make_fake_dataset
from reid_eval.templates import load_video_npz, save_video_npz
from test_reid_thresholds import counts, neg_hist, thr_for

BINS = "configs/bins_v3.yaml"


def make_setup(tmp, tracklets, noise=0.12):
    ds = tmp / "d"
    make_fake_dataset(ds, n_videos=21, n_sites=3, tracklets=tracklets, crops=(14, 20), seed=41)
    split_p = tmp / "split.json"
    make_split(["--data", str(ds), "--n-val", "6", "--n-test", "14", "--out", str(split_p)])
    split = read_json(split_p)
    solo = split["test"][0]                                        # a site with one video of 3 objects: too small to split in two folds
    m = ds / "videos" / solo / "meta.json"
    meta = json.loads(m.read_text()); meta["site"] = "solo_site"; m.write_text(json.dumps(meta))
    recs, infos = load_dataset_with_info(ds, split["validation"] + split["test"], verbose=False)
    by = group_by_video(recs)
    tdir = tmp / "t" / "fake__letterbox"
    # every object of the whole dataset gets its OWN direction: different vehicles (also in different videos) look different
    all_tr = sorted({r.tracklet_id for r in recs})
    rng = np.random.RandomState(5)
    for s_ in ("validation", "test"):
        for vid in split[s_]:
            rs = by[vid]
            emb = np.stack([np.eye(len(all_tr))[all_tr.index(r.tracklet_id)] for r in rs]) + noise * rng.randn(len(rs), len(all_tr))
            save_video_npz(tdir / s_ / f"{vid}.npz", emb.astype(np.float32), [r.crop_uid for r in rs], [r.tracklet_id for r in rs], [r.frame for r in rs])
    write_json_atomic(tdir / "manifest.json", {"model_name": "fake", "preproc_mode": "letterbox", "split_sha256": sha256_file(split_p)})
    return ds, split_p, split, tdir, tmp, solo, infos


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    return make_setup(tmp_path_factory.mktemp("gal"), (3, 3))


def site_data(split, infos, tdir, solo):
    sites = {}
    for v in split["test"]:
        sites.setdefault("solo_site" if v == solo else infos[v].site, []).append(v)
    out = {}
    for site, vids in sites.items():
        embs, trk, vid_of = [], [], []
        for v in sorted(vids):
            t = load_video_npz(tdir / "test" / f"{v}.npz")
            e = t["emb"].astype(np.float64); e /= np.linalg.norm(e, axis=1, keepdims=True)
            embs.append(e); trk += t["tracklet_id"].tolist(); vid_of += [v] * len(e)
        E = np.concatenate(embs)
        names = list(dict.fromkeys(trk))
        obj = np.array([names.index(x) for x in trk])
        iu, ju = np.triu_indices(len(E), 1)
        out[site] = dict(s=(E @ E.T)[iu, ju], oa=obj[iu], ob=obj[ju], va=np.array(vid_of)[iu], vb=np.array(vid_of)[ju], T=len(names), n=len(E))
    return out


@pytest.fixture(scope="module")
def runs(setup):
    ds, split_p, split, tdir, tmp, solo, infos = setup
    return evaluation.run_site_gallery(tdir, split_p, ds, BINS, tmp / "res", ["site-gallery", "site-gallery-oracle"], device="cpu",
                                       verbose=False, per_subset=False)


def test_held_out_site_gallery_matches_brute_force(setup, runs):
    ds, split_p, split, tdir, tmp, solo, infos = setup
    out = runs["site-gallery"]
    assert out.name == "v1__site-gallery"
    data = site_data(split, infos, tdir, solo)
    units = pd.read_csv(out / "thresholds_per_unit.csv")
    assert units[units.unit == "solo_site"].status.iloc[0].startswith("skipped") and "fewer than 4 objects" in units[units.unit == "solo_site"].status.iloc[0]
    tp = p = fp = n = n_pairs = n_cross_neg = 0
    for site, d in data.items():
        if site == "solo_site":
            continue
        fold = assign_folds(f"site::{site}", d["T"], 0)
        for f in (0, 1):
            other = (fold[d["oa"]] == 1 - f) & (fold[d["ob"]] == 1 - f)
            thr = thr_for(neg_hist(d["s"][other & (d["oa"] != d["ob"])]), 0.01)       # calibration: negatives of the OTHER fold (incl. cross-video)
            row = units[(units.unit == site) & (units.fold == f)].iloc[0]
            assert row["thr_1pct"] == pytest.approx(thr, abs=1e-9)
            mine = (fold[d["oa"]] == f) & (fold[d["ob"]] == f)
            a, b, c, e = counts(d["s"][mine], (d["oa"] == d["ob"])[mine], row["thr_1pct"])
            tp += a; p += b; fp += c; n += e; n_pairs += int(mine.sum())
            n_cross_neg += int((mine & (d["va"] != d["vb"])).sum())
    s = read_json(out / "summary.json")
    h, sup = s["headline"]["pooled"], s["headline"]["support"]
    assert sup["n_pos"] + sup["n_neg"] == n_pairs and sup["n_neg"] == n
    assert n_cross_neg > 0 and n_cross_neg > 0.3 * n                                # cross-video pairs ARE part of the negatives
    assert h["tar_at_1pct"]["value"] == pytest.approx(tp / p, abs=3 / p) and h["far_at_1pct"]["value"] == pytest.approx(fp / n, abs=3 / n)
    assert s["threshold_units_skipped"][0]["unit"] == "solo_site" and s["n_videos"] == len(data) - 1   # units are sites


def test_oracle_site_gallery_forces_far_and_is_labelled(setup, runs):
    out = runs["site-gallery-oracle"]
    ps = pd.read_csv(out / "per_site.csv")
    assert "solo_site" in set(ps.site) and len(ps) >= 3                             # no fold needed: even the small site is evaluated
    assert np.allclose(ps["far_at_1pct"], 0.01, rtol=0.1) and np.allclose(ps["far_at_5pct"], 0.05, rtol=0.1)
    text = (out / "report.md").read_text()
    assert "ORACLE evaluation" in text and "SITE MATCHING" in text and "Sites that could not be evaluated" not in text
    assert read_json(out / "summary.json")["oracle"] is True


def test_site_gallery_is_harder_than_within_video(setup, runs):
    """Same objects, but a much larger set of negatives (other videos): the held-out per-site FAR must still be near target."""
    out = runs["site-gallery"]
    h = read_json(out / "summary.json")["headline"]["pooled"]
    assert 0.003 < h["far_at_1pct"]["value"] < 0.03 and h["tar_at_1pct"]["value"] > 0.3
    pv = pd.read_csv(out / "per_video.csv")
    assert (pv.video_id.isin(set(pd.read_csv(out / "per_site.csv").site))).all()    # per_video rows are the sites
    assert not (out / "per_video").exists() and (out / "failures").is_dir()          # no duplicate per-video reports


def test_per_site_reports_without_duplicate_per_video_reports(setup, tmp_path):
    ds, split_p, split, tdir, tmp, solo, infos = setup
    out = evaluation.run_site_gallery(tdir, split_p, ds, BINS, tmp_path, ["site-gallery"], device="cpu", verbose=False,
                                      per_subset=True)["site-gallery"]
    assert (out / "per_site").is_dir() and not (out / "per_video").exists()
    d = next((out / "per_site").iterdir())
    assert (d / "report.md").exists() and (d / "figures" / "roc.png").exists() and "SITE MATCHING" in (d / "report.md").read_text()
