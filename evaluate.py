#!/usr/bin/env python
"""Within-video verification evaluation (spec section 8).

  python evaluate.py --templates templates/<model>__<mode> --split splits/eval_split_v1.json --data <root> \
         --bins configs/bins_v1.yaml --out results/

  python evaluate.py --templates ... --plain       # same, WITHOUT the difficulty criteria (pose/occlusion/keypoints): global metrics,
                                                   # per site and per video only -> results/<model>__<mode>/<split>__plain/

  python evaluate.py --choose-mode templates/<m>__letterbox templates/<m>__unpad_stretch --split ... --data <root> --out results/
      picks the preprocessing mode with the higher pooled AUC on the VALIDATION videos only

  python evaluate.py --report-only results/<model>__<mode>/<split_version>   # rebuild report.md from saved files
"""
import argparse
import sys
from pathlib import Path

from reid_data import group_by_video, load_dataset
from reid_eval import evaluation
from reid_eval.common import read_json, write_json_atomic


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--templates")
    ap.add_argument("--split")
    ap.add_argument("--data")
    ap.add_argument("--bins", default="configs/bins_v3.yaml", help="bins + FAR targets (v2: 0.1, 1, 2, 5, 10 %%)")
    ap.add_argument("--out", default="results/")
    ap.add_argument("--device", default=None)
    ap.add_argument("--choose-mode", nargs="+", metavar="TEMPLATE_DIR")
    ap.add_argument("--report-only", metavar="RESULTS_DIR")
    ap.add_argument("--plain", action="store_true", help="no difficulty criteria: global / per-site / per-video metrics only")
    ap.add_argument("--no-per-subset", action="store_true", help="skip the per-site and per-video reports (faster)")
    ap.add_argument("--match-sheets", dest="match_sheets", action="store_true", default=None, help="force per-object match images (default: on for full, off for plain)")
    ap.add_argument("--no-match-sheets", dest="match_sheets", action="store_false")
    ap.add_argument("--match-topk", type=int, default=10, help="matches per row in the per-object images")
    ap.add_argument("--no-report", action="store_true")
    a = ap.parse_args(argv)

    if a.report_only:
        from reid_eval.report import build_report_from_dir
        root = Path(a.report_only)
        print(build_report_from_dir(root, a.data))
        for sub in sorted(list((root / "per_site").glob("*/summary.json")) + list((root / "per_video").glob("*/summary.json"))):
            build_report_from_dir(sub.parent, a.data)
        return 0
    if not (a.split and a.data):
        ap.error("--split and --data are required")
    if a.choose_mode:
        split = read_json(a.split)
        by_video = group_by_video(load_dataset(a.data, video_ids=split["validation"], verbose=False))
        res = evaluation.select_mode(a.choose_mode, split, by_video, a.device)
        model = read_json(Path(a.choose_mode[0]) / "manifest.json")["model_name"]
        path = Path(a.out) / f"{model}__mode_selection_{split['version']}.json"
        write_json_atomic(path, res)
        print(res, "->", path)
        return 0
    if not a.templates:
        ap.error("--templates is required")
    out = evaluation.run(a.templates, a.split, a.data, a.bins, a.out, a.device, plain=a.plain, per_subset=not a.no_per_subset,
                         match_sheets=a.match_sheets, match_topk=a.match_topk)
    print(f"results in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
