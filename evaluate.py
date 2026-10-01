#!/usr/bin/env python
"""Within-video verification evaluation (spec section 8).

  python evaluate.py --templates templates/<model>__<mode> --split splits/eval_split_v1.json --data <root> \
         --bins configs/bins_v1.yaml --out results/

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
    ap.add_argument("--bins", default="configs/bins_v1.yaml")
    ap.add_argument("--out", default="results/")
    ap.add_argument("--device", default=None)
    ap.add_argument("--choose-mode", nargs="+", metavar="TEMPLATE_DIR")
    ap.add_argument("--report-only", metavar="RESULTS_DIR")
    ap.add_argument("--no-report", action="store_true")
    a = ap.parse_args(argv)

    if a.report_only:
        from reid_eval.report import build_report_from_dir
        print(build_report_from_dir(a.report_only, a.data))
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
    out = evaluation.run(a.templates, a.split, a.data, a.bins, a.out, a.device)
    if not a.no_report:
        from reid_eval.report import build_report_from_dir
        build_report_from_dir(out, a.data)
    print(f"results in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
