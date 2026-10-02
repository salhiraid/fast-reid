"""Site-level matching visuals: objects of a site against the gallery of all crops of all the site's videos."""
import json

import cv2
import numpy as np
import pandas as pd
import pytest

from make_split import main as make_split
from reid_data import group_by_video, load_dataset_with_info
from reid_eval import evaluation
from reid_eval.bins import Bins
from reid_eval.common import read_json, sha256_file
from reid_eval.synthetic import make_fake_dataset, make_fake_templates
from test_reid_thresholds import neg_hist, thr_for, video_pairs

BINS = "configs/bins_v3.yaml"
TILE, PAD, TOPK = 112, 6, 10


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sitem")
    ds = tmp / "d"
    make_fake_dataset(ds, n_videos=15, n_sites=3, tracklets=(5, 6), crops=(12, 14), seed=31)
    split_p = tmp / "split.json"
    make_split(["--data", str(ds), "--n-val", "5", "--n-test", "10", "--out", str(split_p)])
    split = read_json(split_p)
    lonely = split["test"][0]                                    # a site with a single test video (crops unchanged: fingerprint ok)
    m = ds / "videos" / lonely / "meta.json"
    meta = json.loads(m.read_text()); meta["site"] = "lonely_site"; m.write_text(json.dumps(meta))
    recs, infos = load_dataset_with_info(ds, split["validation"] + split["test"], verbose=False)
    tdir = tmp / "t" / "fake__letterbox"
    # one-hot per tracklet index inside the video: object k of one video looks like object k of every other video
    make_fake_templates(tdir, split, group_by_video(recs), sha256_file(split_p), noise=0.1, dim=16, seed=5)
    out = evaluation.run_threshold_modes(tdir, split_p, ds, BINS, tmp / "res", ["site"], device="cpu", verbose=False,
                                         per_subset=False, site_queries=3)["site"]
    return ds, split, tdir, out, infos, lonely, tmp


def sites_of(split, infos):
    d = {}
    for v in split["test"]:
        d.setdefault(infos[v].site if v != split["test"][0] else "lonely_site", []).append(v)
    return d


def test_files_and_links(run):
    ds, split, tdir, out, infos, lonely, tmp = run
    sm = out / "site_matches"
    assert (sm / "index.md").exists() and (sm / "index.csv").exists()
    idx = pd.read_csv(sm / "index.csv")
    assert set(idx.site) == set(sites_of(split, infos))
    for site in idx.site:
        d = sm / site
        assert (d / "object_matrix.png").exists() and (d / "cross_video_candidates.csv").exists() and (d / "index.md").exists()
        q = pd.read_csv(d / "queries_index.csv")
        n_obj = int(idx[idx.site == site].objects.iloc[0])
        assert len(q) == min(3, n_obj) and len(q) < n_obj                     # SOME objects, not all
        assert all((d / s).exists() for s in q.sheet)
    summary = read_json(out / "summary.json")
    assert summary["site_matches"] == "site_matches/index.md"
    assert "Site-level matching" in (out / "report.md").read_text()


def test_site_threshold_and_cross_video_candidates_against_brute_force(run):
    ds, split, tdir, out, infos, lonely, tmp = run
    bins = Bins.load(BINS)
    idx = pd.read_csv(out / "site_matches" / "index.csv").set_index("site")
    for site, vids in sites_of(split, infos).items():
        pairs = {v: video_pairs(tdir, v) for v in vids}
        hist = sum(neg_hist(p[0][p[1] != p[2]]) for p in pairs.values())
        assert idx.loc[site, "threshold"] == pytest.approx(thr_for(hist, 0.01), abs=1e-9)     # all within-video negatives of the site
        cand = pd.read_csv(out / "site_matches" / site / "cross_video_candidates.csv")
        if len(vids) == 1:
            assert len(cand) == 0                                              # one video: nothing to match across
            continue
        assert (cand.video_a != cand.video_b).all()                            # only cross-video pairs
        # one-hot embeddings: object k of video A ~ object k of video B -> the best candidates are the same index
        top = cand.head(5)
        assert (top.object_a.str.rsplit("_", n=1).str[1] == top.object_b.str.rsplit("_", n=1).str[1]).all() and (top.similarity > 0.8).all()
        assert cand.similarity.is_monotonic_decreasing and (cand.above_site_threshold == (cand.similarity >= idx.loc[site, "threshold"])).all()
        # rank 1 = the cross-video object pair with the highest MEAN cosine similarity over all their crop pairs (brute force)
        from reid_eval.templates import load_video_npz
        objs = {}
        for v in vids:
            t = load_video_npz(tdir / "test" / f"{v}.npz"); e = t["emb"].astype(np.float64); e /= np.linalg.norm(e, axis=1, keepdims=True)
            for tid in dict.fromkeys(t["tracklet_id"].tolist()):
                objs[(v, tid)] = e[t["tracklet_id"] == tid]
        best = max(float((objs[a] @ objs[b].T).mean()) for a in objs for b in objs if a[0] < b[0])
        assert cand.similarity.iloc[0] == pytest.approx(best, abs=1e-5)
        assert (cand.max_crop_pair_similarity >= cand.similarity - 1e-6).all()          # the best crop pair is at least the mean


def test_query_sheet_layout_and_rows(run):
    ds, split, tdir, out, infos, lonely, tmp = run
    sm = out / "site_matches"
    for site, vids in sites_of(split, infos).items():
        q = pd.read_csv(sm / site / "queries_index.csv")
        for r in q.itertuples():
            im = cv2.imread(str(sm / site / r.sheet))
            rows = 3 if len(vids) > 1 else 2                                    # same object / same video / other videos
            assert im.shape[0] == 30 + rows * (TILE + 46) + 6 and im.shape[1] == PAD + (TOPK + 1) * (TILE + PAD)
            if len(vids) > 1:
                from reid_eval.templates import load_video_npz
                k = r.tracklet_id.rsplit("_", 1)[1]
                twins = [v for v in vids if v != r.video_id and any(t.endswith("_" + k) for t in set(load_video_npz(tdir / "test" / f"{v}.npz")["tracklet_id"].tolist()))]
                if twins:                                                       # an object with the same index exists elsewhere: it must be found
                    assert r.top_other_video_sim > 0.8
                assert isinstance(r.top_other_video_object, str) and r.top_other_video_object.split("_")[0] != r.video_id
            else:
                assert np.isnan(r.top_other_video_sim)
            assert r.best_same_object_sim > 0.5


def test_queries_are_seeded_and_candidate_sheets_written(run):
    ds, split, tdir, out, infos, lonely, tmp = run
    again = evaluation.run_threshold_modes(tdir, tmp / "split.json", ds, BINS, tmp / "res2", ["site"], device="cpu", verbose=False,
                                           per_subset=False, site_queries=3)["site"]
    for site in sites_of(split, infos):
        a = pd.read_csv(out / "site_matches" / site / "queries_index.csv")
        b = pd.read_csv(again / "site_matches" / site / "queries_index.csv")
        assert a[["video_id", "tracklet_id"]].equals(b[["video_id", "tracklet_id"]])        # same seed -> same queries
    multi = [s for s, v in sites_of(split, infos).items() if len(v) > 1][0]
    assert list((out / "site_matches" / multi).glob("cross_video_candidates_*.png"))


def test_site_queries_zero_disables(run):
    ds, split, tdir, out, infos, lonely, tmp = run
    o = evaluation.run_threshold_modes(tdir, tmp / "split.json", ds, BINS, tmp / "res3", ["site"], device="cpu", verbose=False,
                                       per_subset=False, site_queries=0)["site"]
    assert not (o / "site_matches").exists() and read_json(o / "summary.json")["site_matches"] is None
