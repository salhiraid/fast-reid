"""Per-object match sheets: for every object (tracklet) of a video, ONE image with its best positive and negative matches.

Query = the object's medoid crop (the crop most similar, on average, to the object's other crops). All similarities are
cosine similarities inside the video, exactly as in the evaluation. Rows of a sheet:
  1. top positives    the k crops of the SAME object most similar to the query
  2. hardest positives the k crops of the same object LEAST similar to the query (only if the object has more than k)
  3. top negatives    the k crops of OTHER objects most similar to the query (what a verifier would wrongly accept)
Each tile shows the similarity, and a flag against the global threshold t(FAR 1 %):
  positive below it = FR (false reject);  negative at or above it = FA (false accept).
Tiles are drawn as stored (224x224 letterbox), not unpadded.
"""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

GREEN, ORANGE, RED, GREY, INK, BG = (60, 150, 40), (0, 140, 255), (40, 40, 210), (170, 170, 170), (30, 30, 30), (245, 245, 245)


def _slug(name):
    return re.sub(r"[^\w.\-]+", "_", str(name)).strip("_") or "unnamed"


class _Tiles:
    def __init__(self, tile):
        self.tile, self.cache = tile, {}

    def get(self, rec):
        t = self.cache.get(rec.crop_uid)
        if t is None:
            img = cv2.imread(str(rec.path), cv2.IMREAD_COLOR)
            t = np.zeros((self.tile, self.tile, 3), np.uint8) if img is None else cv2.resize(img, (self.tile, self.tile), interpolation=cv2.INTER_AREA)
            self.cache[rec.crop_uid] = t
        return t


def _put(img, text, org, scale=0.38, color=INK, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _row(canvas, y, title, query_tile, recs, sims, flags, tiles, tile, pad, extra_labels=None):
    _put(canvas, title, (pad, y + 12), 0.45, INK, 1)
    y0 = y + 18
    cols = [("query", query_tile, None, None, "query")] + [(r, tiles.get(r), s, f, l) for r, s, f, l in zip(recs, sims, flags, extra_labels or [""] * len(recs))]
    for k, (r, im, s, f, lab) in enumerate(cols):
        x = pad + k * (tile + pad)
        canvas[y0:y0 + tile, x:x + tile] = im
        color = INK if k == 0 else {"ok": GREEN, "FR": ORANGE, "FA": RED, "": GREY}[f]
        cv2.rectangle(canvas, (x - 1, y0 - 1), (x + tile, y0 + tile), color, 3 if k else 1)
        if k:
            _put(canvas, f"{s:.3f}" + (f" {f}" if f in ("FR", "FA") else ""), (x + 1, y0 + tile + 11), 0.38, color if f in ("FR", "FA") else INK)
            if lab:
                _put(canvas, lab[:16], (x + 1, y0 + tile + 22), 0.32, (90, 90, 90))
        else:
            _put(canvas, "query", (x + 1, y0 + tile + 11), 0.38, INK)
    return y0 + tile + 28


def object_sheets(emb, records, out_dir, thr, thr_label, topk=10, tile=112, device=None):
    """Write one PNG per object into `out_dir` (+ index.csv, index.md); return the index rows (list of dicts)."""
    import pandas as pd
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    E = F.normalize(torch.as_tensor(np.asarray(emb), dtype=torch.float32, device=dev), dim=1)
    S = (E @ E.T).cpu().numpy()
    trk = np.array([r.tracklet_id for r in records])
    tiles = _Tiles(tile)
    pad, rows = 6, []
    for tid in dict.fromkeys(trk.tolist()):                      # objects in canonical order
        idx = np.nonzero(trk == tid)[0]
        others = np.nonzero(trk != tid)[0]
        if len(idx) > 1:
            q = int(idx[np.argmax((S[np.ix_(idx, idx)].sum(1) - 1.0) / (len(idx) - 1))])   # medoid
        else:
            q = int(idx[0])
        pos = idx[idx != q]
        pos = pos[np.argsort(-S[q, pos], kind="stable")] if len(pos) else pos
        neg = others[np.argsort(-S[q, others], kind="stable")[:topk]] if len(others) else others
        flag_pos = lambda s: "ok" if s >= thr else "FR"
        flag_neg = lambda s: "FA" if s >= thr else ""
        qt = tiles.get(records[q])
        sections = [("TOP POSITIVES: most similar crops of the same object", pos[:topk], flag_pos, False)]
        if len(pos) > topk:
            sections.append(("HARDEST POSITIVES: least similar crops of the same object", pos[::-1][:topk], flag_pos, False))
        sections.append(("TOP NEGATIVES: most similar crops of OTHER objects", neg, flag_neg, True))
        H = 30 + len(sections) * (tile + 46) + 6
        W = pad + (topk + 1) * (tile + pad)
        canvas = np.full((H, W, 3), BG, np.uint8)
        _put(canvas, f"{records[q].video_id} | object {tid} | {len(idx)} crops | query = medoid crop, frame {records[q].frame} | "
                     f"threshold {thr_label} = {thr:.3f}", (pad, 18), 0.5, INK, 1)
        y = 26
        for title, sel, flagf, is_neg in sections:
            sims = [float(S[q, j]) for j in sel]
            y = _row(canvas, y, title, qt, [records[j] for j in sel], sims, [flagf(s) for s in sims], tiles, tile, pad,
                     [("obj " + records[j].tracklet_id.rsplit("_", 1)[-1]) for j in sel] if is_neg else
                     [f"f{records[j].frame}" for j in sel])
        name = f"{_slug(tid)}.png"
        cv2.imwrite(str(out_dir / name), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        ps = S[q, pos] if len(pos) else np.array([])
        ns_all = S[q, others] if len(others) else np.array([])
        rows.append({
            "video_id": records[q].video_id, "tracklet_id": tid, "n_crops": len(idx), "query_crop": records[q].crop_uid,
            "query_frame": records[q].frame, "n_positives": len(pos),
            "best_positive_sim": float(ps.max()) if len(ps) else np.nan, "worst_positive_sim": float(ps.min()) if len(ps) else np.nan,
            "false_rejects": int((ps < thr).sum()) if len(ps) else 0,
            "top_negative_sim": float(ns_all.max()) if len(ns_all) else np.nan,
            "top_negative_object": records[int(others[np.argmax(ns_all)])].tracklet_id if len(ns_all) else "",
            "false_accepts": int((ns_all >= thr).sum()) if len(ns_all) else 0,
            "separation": float(ps.min() - ns_all.max()) if len(ps) and len(ns_all) else np.nan, "sheet": name})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "index.csv", index=False)
    srt = df.sort_values("separation", kind="stable", na_position="last")
    md = [f"# Object matches: {records[0].video_id}", "",
          f"One sheet per object. Query = medoid crop; top {topk} positives / hardest positives / top {topk} negatives; "
          f"flags against the global threshold {thr_label} = {thr:.3f} (FR = false reject, FA = false accept). "
          "Objects are listed most confusable first (separation = lowest positive - highest negative similarity of the query).", "",
          "| object | crops | lowest positive | highest negative (object) | separation | FR | FA |", "|---|---|---|---|---|---|---|"]
    for r in srt.itertuples():
        md.append(f"| [{r.tracklet_id}](#{_slug(r.tracklet_id).lower()}) | {r.n_crops} | {r.worst_positive_sim:.3f} | "
                  f"{r.top_negative_sim:.3f} ({r.top_negative_object}) | {r.separation:.3f} | {r.false_rejects} | {r.false_accepts} |")
    for r in srt.itertuples():
        md += ["", f"<a id=\"{_slug(r.tracklet_id).lower()}\"></a>", f"### {r.tracklet_id}", "", f"![{r.tracklet_id}]({r.sheet})"]
    (out_dir / "index.md").write_text("\n".join(md), encoding="utf-8")
    return rows
