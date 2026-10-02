#!/usr/bin/env python
"""Universal evaluation: dataset + a folder of templates (one sub-folder per model) -> all evaluations -> model comparison.

  python run_eval.py --data DATASET --templates templates/ [--split splits/eval_split_v1.json] [--out results/]
        runs every model found in templates/, then writes results/comparison_<split>/report.md (models side by side)
  python run_eval.py --data DATASET --templates templates/ --models mymodel other --variants full thr-site site-gallery
  python run_eval.py --compare results/ [--split-version v1] [--models a b] [--reference a]    # comparison tables/curves only
  python run_eval.py --list templates/                                                          # what was found

Variants: full plain thr-video thr-site site-gallery (default) and the optimistic thr-video-oracle thr-site-oracle site-gallery-oracle.
The templates must have been made with the SAME split file (give it with --split, or copy it into the templates folder).
"""
import argparse
import sys
from pathlib import Path

from reid_eval.model_compare import build_model_comparison
from reid_eval.universal import ALL_VARIANTS, DEFAULT_VARIANTS, find_models, run_all

HERE = Path(__file__).resolve().parent


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="dataset root (folder with videos/<id>/meta.json)")
    ap.add_argument("--templates", help="folder with one sub-folder per model: <model>__<mode>/{manifest.json,validation/,test/}")
    ap.add_argument("--split", default=None, help="split file used to extract the templates (default: eval_split*.json next to the templates, "
                                                  "or the path stored in a manifest)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--models", nargs="+", default=None, help="only these models (default: all found)")
    ap.add_argument("--variants", nargs="+", choices=ALL_VARIANTS, default=list(DEFAULT_VARIANTS))
    ap.add_argument("--bins", default=str(HERE / "configs" / "bins_v3.yaml"))
    ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    ap.add_argument("--all-modes", action="store_true", help="evaluate every preprocessing mode of a model instead of choosing one on validation")
    ap.add_argument("--no-per-subset", action="store_true", help="skip the per-site / per-video reports (faster)")
    ap.add_argument("--no-match-sheets", action="store_true", help="skip the per-object top-10 match images (faster)")
    ap.add_argument("--site-queries", type=int, default=10)
    ap.add_argument("--match-topk", type=int, default=10)
    ap.add_argument("--no-compare", action="store_true", help="do not build the model comparison at the end")
    ap.add_argument("--reference", default=None, help="reference model of the paired differences in the comparison (default: the first)")
    ap.add_argument("--compare", metavar="RESULTS_DIR", help="only build the model comparison from an existing results folder")
    ap.add_argument("--split-version", default="v1", help="with --compare: split version prefix of the result folders")
    ap.add_argument("--methods", nargs="+", default=None, help="with --compare: only these methods (global thr-video thr-site site-gallery ...)")
    ap.add_argument("--list", metavar="TEMPLATES_DIR", help="list the models / preprocessing modes found in a templates folder")
    a = ap.parse_args(argv)

    if a.list:
        for name, modes in find_models(a.list).items():
            print(f"{name}: " + ", ".join(f"{m} ({d.name})" for m, d in sorted(modes.items())))
        return 0
    if a.compare:
        path = build_model_comparison(a.compare, a.split_version, models=a.models, methods=a.methods, reference=a.reference)
        print(path if path else f"no evaluated model folders for split '{a.split_version}' in {a.compare}")
        return 0 if path else 1
    if not (a.data and a.templates):
        ap.error("--data and --templates are required (or use --compare / --list)")
    res = run_all(a.data, a.templates, a.out, a.split, a.models, a.variants, a.bins, a.device, not a.no_per_subset, not a.no_match_sheets,
                  a.site_queries, a.match_topk, a.all_modes, not a.no_compare, a.reference)
    print("\nDONE")
    for label, d in res["results"].items():
        print(f"  {label}: {d / (res['version'] + '__all_methods') / 'report.md'}")
    if res["comparison"]:
        print(f"  model comparison: {res['comparison']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
