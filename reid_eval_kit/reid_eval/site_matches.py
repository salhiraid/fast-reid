"""Site-level matching visuals: every object of a site against the gallery of ALL crops of ALL videos of that site.

Labels exist only inside a video (identity_id == tracklet_id per video), so:
  * same video, same object  = positive, same video, other object = negative  (labelled; flagged FR / FA against the site threshold);
  * other videos of the site = identity UNKNOWN: the same vehicle may really reappear. These matches are never counted as errors; a
    similarity above the site threshold is flagged '?' (possible same vehicle), nothing more.
The site threshold is the FAR threshold (default nearest 1 %) of ALL within-video negative pairs of the site's videos: a qualitative
aid, set on the data it is drawn from, not a result. Outputs per site: sampled query sheets (NOT all objects), one object x object
similarity matrix, and the top cross-video candidate pairs (CSV + contact sheets).

Object-to-object similarity = the MEAN cosine similarity over all crop pairs of the two objects (= the dot product of their mean
normalised embeddings). The maximum over crop pairs would be compared unfairly against a crop-level threshold (the best of ~100 pairs
passes a 1 %-FAR threshold far more often than 1 % of the time).
"""
from __future__ import annotations

import zlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .failures import contact_sheets
from .matches import BG, INK, _Tiles, _put, _row, _slug


def _object_mean_matrix(E, obj_of, T):
    """(T, T) mean cosine similarity over all crop pairs of two objects; the diagonal (an object with itself) is not meaningful."""
    idx = torch.as_tensor(obj_of, device=E.device)
    Em = torch.zeros(T, E.shape[1], device=E.device).index_add_(0, idx, E)
    Em /= torch.bincount(idx, minlength=T).clamp(min=1).unsqueeze(1).float()
    return (Em @ Em.T).cpu().numpy()


def plot_site_matrix(M, obj_video, video_names, thr, thr_label, path, title=""):
    """Object x object mean pair similarity. Blocks = videos. Red dots: same-video pair above the threshold (labelled false accept);
    purple dots: other-video pair above it (identity unknown, possible same vehicle)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .plots import GREY, INK as INK_, INK2, SEQ
    T = len(M)
    M = M.astype(np.float64).copy()
    np.fill_diagonal(M, np.nan)
    size = float(np.clip(3.2 + T / 45, 4.5, 13))
    fig, ax = plt.subplots(figsize=(size + 1.2, size), dpi=130)
    ax.imshow(M, cmap=SEQ, vmin=0, vmax=1, interpolation="nearest", aspect="equal")
    same_video = obj_video[:, None] == obj_video[None, :]
    hit = (M >= thr) & ~np.eye(T, dtype=bool)
    for mask, color, lab in ((hit & same_video, "#d03b3b", "same video, above threshold (false accept)"),
                             (hit & ~same_video, "#7b2d8e", "other video, above threshold (identity unknown)")):
        yy, xx = np.nonzero(mask)
        if len(yy):
            ax.scatter(xx, yy, s=max(2, 90 / np.sqrt(max(T, 1))), color=color, label=f"{lab}: {len(yy) // 2} pairs", zorder=3)
    edges = np.nonzero(np.diff(obj_video))[0] + 0.5
    for e in edges:
        ax.axhline(e, color=GREY, lw=0.5)
        ax.axvline(e, color=GREY, lw=0.5)
    centres = [np.mean(np.nonzero(obj_video == v)[0]) for v in range(len(video_names))]
    if len(video_names) <= 40:
        ax.set_xticks(centres)
        ax.set_xticklabels(video_names, rotation=90, fontsize=6)
        ax.set_yticks(centres)
        ax.set_yticklabels(video_names, fontsize=6)
    else:
        ax.set_xticks([]), ax.set_yticks([])
    ax.set_title((title + "\n" if title else "") + f"object x object mean pair similarity, threshold {thr_label} = {thr:.3f}", fontsize=8, loc="left", color=INK_)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2 if len(video_names) <= 40 else -0.02))
    cb = fig.colorbar(ax.images[0], ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def site_visuals(site, video_ids, emb_by_video, recs_by_video, thr, thr_label, out_dir, n_queries=10, topk=10, seed=0, tile=112,
                 n_candidates=200, device=None):
    """Write the site-level visuals into `out_dir` and return a small summary dict."""
    out_dir = Path(out_dir)
    (out_dir / "queries").mkdir(parents=True, exist_ok=True)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    vids = sorted(video_ids)
    recs = [r for v in vids for r in recs_by_video[v]]
    E = F.normalize(torch.as_tensor(np.concatenate([np.asarray(emb_by_video[v]) for v in vids]), dtype=torch.float32, device=dev), dim=1)
    N = len(recs)
    obj_key = [(r.video_id, r.tracklet_id) for r in recs]
    objs = list(dict.fromkeys(obj_key))
    obj_of = np.array([objs.index(k) for k in obj_key]) if N else np.zeros(0, int)
    # canonical order is grouped by object, so objects are contiguous
    starts = np.concatenate([[0], np.cumsum(np.bincount(obj_of, minlength=len(objs)))])
    T = len(objs)
    vid_index = {v: i for i, v in enumerate(vids)}
    obj_video = np.array([vid_index[o[0]] for o in objs])
    crop_video = obj_video[obj_of]
    tiles = _Tiles(tile)

    # ---- object x object matrix, candidates
    M = _object_mean_matrix(E, obj_of, T)
    plot_site_matrix(M, obj_video, vids, thr, thr_label, out_dir / "object_matrix.png", title=f"site {site}: {len(vids)} video(s), {T} objects")
    iu, ju = np.triu_indices(T, 1)
    cross = obj_video[iu] != obj_video[ju]
    cand_df = pd.DataFrame(columns=["rank", "similarity", "max_crop_pair_similarity", "video_a", "object_a", "video_b", "object_b", "crop_a", "crop_b",
                                    "above_site_threshold", "delta_position_m"])
    if cross.any():
        ci, cj, cs = iu[cross], ju[cross], M[iu[cross], ju[cross]]
        top = np.argsort(-cs, kind="stable")[:n_candidates]
        rows = []
        for rank, k in enumerate(top, 1):
            a, b = int(ci[k]), int(cj[k])
            ra, rb = np.arange(starts[a], starts[a + 1]), np.arange(starts[b], starts[b + 1])
            S = (E[ra] @ E[rb].T).cpu().numpy()
            x, y = np.unravel_index(int(np.argmax(S)), S.shape)
            rows.append({"rank": rank, "similarity": float(cs[k]), "max_crop_pair_similarity": float(S.max()), "video_a": objs[a][0], "object_a": objs[a][1], "video_b": objs[b][0],
                         "object_b": objs[b][1], "crop_a": recs[ra[x]].crop_uid, "crop_b": recs[rb[y]].crop_uid,
                         "above_site_threshold": bool(cs[k] >= thr), "delta_position_m": np.nan})
        cand_df = pd.DataFrame(rows)
    cand_df.to_csv(out_dir / "cross_video_candidates.csv", index=False)

    # ---- sampled query sheets (NOT every object)
    rng = np.random.RandomState((zlib.crc32(str(site).encode()) + seed) % (2 ** 32))
    chosen = sorted(rng.permutation(T)[:n_queries].tolist()) if n_queries else []
    index = []
    pad = 6
    for o in chosen:
        idx = np.arange(starts[o], starts[o + 1])
        q = int(idx[np.argmax((E[idx] @ E[idx].T).sum(1).cpu().numpy())]) if len(idx) > 1 else int(idx[0])    # medoid
        sims = (E[q] @ E.T).cpu().numpy()
        own = np.zeros(N, bool); own[idx] = True; own[q] = False
        vid_q = obj_video[o]
        same_vid = (crop_video == vid_q) & ~own
        same_vid[idx] = False
        other_vid = crop_video != vid_q

        def closest_objects(mask):
            """The k other objects (among the crops in `mask`) most similar to the query OBJECT (mean pair similarity); for each,
            the crop of that object closest to the query crop is shown, with the object-level similarity."""
            objs_here = np.unique(obj_of[np.nonzero(mask)[0]])
            objs_here = objs_here[np.argsort(-M[o, objs_here], kind="stable")][:topk]
            out = []
            for oj in objs_here:
                rows_oj = np.arange(starts[oj], starts[oj + 1])
                out.append((int(rows_oj[np.argmax(sims[rows_oj])]), float(M[o, oj])))
            return out

        pos = np.nonzero(own)[0]
        pos = pos[np.argsort(-sims[pos], kind="stable")][:topk]
        neg_same = closest_objects(same_vid)
        neg_other = closest_objects(other_vid)
        sections = [("SAME OBJECT (same video): most similar crops [FR = below the site threshold]", pos, lambda s: "ok" if s >= thr else "FR",
                     lambda j: f"f{recs[j].frame}"),
                    ("OTHER OBJECTS, SAME VIDEO: most similar objects (value = mean pair similarity of the two objects) "
                     "[FA = above the site threshold]", neg_same, lambda s: "FA" if s >= thr else "",
                     lambda j: "obj " + recs[j].tracklet_id.rsplit("_", 1)[-1])]
        if len(vids) > 1:
            sections.append(("OTHER VIDEOS OF THE SITE (identity UNKNOWN): most similar objects (value = mean pair similarity) "
                             "[? = above the site threshold: possible same vehicle]", neg_other, lambda s: "cand" if s >= thr else "",
                             lambda j: f"{recs[j].video_id[-6:]}/{recs[j].tracklet_id.rsplit('_', 1)[-1]}"))
        H = 30 + len(sections) * (tile + 46) + 6
        W = pad + (topk + 1) * (tile + pad)
        canvas = np.full((H, W, 3), BG, np.uint8)
        _put(canvas, f"site {site} | {objs[o][0]} | object {objs[o][1]} | {len(idx)} crops | query = medoid crop | site threshold "
                     f"{thr_label} = {thr:.3f}", (pad, 18), 0.5, INK, 1)
        y = 26
        qt = tiles.get(recs[q])
        for k_sec, (title, sel, flagf, labf) in enumerate(sections):
            if k_sec == 0:                                    # same object: crop-to-crop similarity
                crops, sl = list(sel), [float(sims[j]) for j in sel]
            else:                                             # other objects: object-level similarity
                crops, sl = [c for c, _ in sel], [v for _, v in sel]
            y = _row(canvas, y, title, qt, [recs[j] for j in crops], sl, [flagf(v) for v in sl], tiles, tile, pad, [labf(j) for j in crops])
        name = f"{_slug(objs[o][0])}__{_slug(objs[o][1])}.png"
        cv2.imwrite(str(out_dir / "queries" / name), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        index.append({"video_id": objs[o][0], "tracklet_id": objs[o][1], "n_crops": len(idx), "query_crop": recs[q].crop_uid,
                      "best_same_object_sim": float(sims[pos[0]]) if len(pos) else np.nan,
                      "top_other_object_same_video_sim": neg_same[0][1] if neg_same else np.nan,
                      "top_other_video_sim": neg_other[0][1] if neg_other else np.nan,
                      "top_other_video_object": recs[neg_other[0][0]].tracklet_id if neg_other else "",
                      "other_video_objects_above_threshold": int(sum(M[o, k] >= thr for k in range(T) if obj_video[k] != vid_q)),
                      "sheet": f"queries/{name}"})
    pd.DataFrame(index).to_csv(out_dir / "queries_index.csv", index=False)
    if len(cand_df):
        contact_sheets(cand_df, _data_root_of(recs[0]), out_dir / "cross_video_candidates")
    md = [f"# Site {site}", "",
          f"{len(vids)} test video(s), {T} objects, {N} crops. Site threshold {thr_label} = {thr:.3f} (set on all within-video negative pairs "
          "of the site's videos; a visualisation aid, not a result).", "",
          "Only objects of the same video have labels. **Matches between videos have no ground truth** (the same vehicle may reappear): they are "
          "never counted as errors; `?` marks a similarity above the site threshold = possible same vehicle.", "",
          "## Object x object similarity", "", "![matrix](object_matrix.png)", "",
          f"## Top cross-video candidate pairs\n\n`cross_video_candidates.csv` ({len(cand_df)} pairs, best crop pair of each); "
          f"{int(cand_df['above_site_threshold'].sum()) if len(cand_df) else 0} above the site threshold.", ""]
    for p in sorted(out_dir.glob("cross_video_candidates_*.png"))[:3]:
        md += [f"![{p.stem}]({p.name})", ""]
    md += [f"## Sampled queries ({len(index)} of {T} objects, seeded)", ""]
    for r in index:
        md += [f"### {r['video_id']} / {r['tracklet_id']}", "", f"![{r['tracklet_id']}]({r['sheet']})", ""]
    (out_dir / "index.md").write_text("\n".join(md), encoding="utf-8")
    return {"site": site, "videos": len(vids), "objects": T, "crops": N, "queries": len(index), "candidates": len(cand_df),
            "candidates_above_threshold": int(cand_df["above_site_threshold"].sum()) if len(cand_df) else 0, "threshold": float(thr)}


def _data_root_of(rec):
    """Dataset root from a record path: <root>/videos/<video>/crops/<track>/<file> -> <root>."""
    return Path(rec.path).parents[4]
