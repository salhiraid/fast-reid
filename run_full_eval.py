#!/usr/bin/env python
"""Whole pipeline for one model in one command (cross-platform):
split (created once, reused if it exists) -> templates for both preprocessing modes -> choose the mode on VALIDATION
-> evaluate the chosen mode TWICE: "full" (all difficulty criteria: delta position / azimuth / occlusion / keypoints) and
"plain" (no pose criteria, global result) -> report.md for the whole test set, every site and every video.

  python run_full_eval.py --data D:/data/Re-ID_safe_test --model fastreid_veriwild_r50ibn --weights veriwild_bot_R50-ibn.pth

Re-running resumes: the split is reused, finished templates are skipped, evaluation is recomputed (it is fast).
"""
import argparse
import sys
from pathlib import Path

import evaluate
import extract_templates
import make_split
from reid_eval.common import read_json
from reid_eval.templates import template_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="dataset root")
    ap.add_argument("--model", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--split", default="splits/eval_split_v1.json")
    ap.add_argument("--bins", default="configs/bins_v3.yaml", help="FAR targets 0.1, 1, 2, 5, 10 %%")
    ap.add_argument("--variants", nargs="+", choices=["full", "plain", "thr-video", "thr-site", "site-gallery", "thr-global-oracle", "thr-video-oracle", "thr-site-oracle", "site-gallery-oracle"],
                    default=["full", "plain", "thr-video", "thr-site", "site-gallery", "thr-global-oracle"],
                    help="thr-*: threshold per video / per site (held out); the -oracle ones are tuned on the evaluated data (optimistic, opt-in)")
    ap.add_argument("--no-all-methods", action="store_true", help="skip the final all-methods comparison (tables + curves)")
    ap.add_argument("--no-per-subset", action="store_true", help="skip the per-site and per-video reports (faster)")
    ap.add_argument("--no-match-sheets", action="store_true", help="skip the per-object top-10 match images (faster)")
    ap.add_argument("--match-topk", type=int, default=10)
    ap.add_argument("--site-queries", type=int, default=10, help="sampled query objects per site in the site-level matching visuals (0 = none)")
    ap.add_argument("--templates", default="templates/")
    ap.add_argument("--out", default="results/")
    ap.add_argument("--modes", nargs="+", default=["letterbox", "unpad_stretch"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--flip-tta", action="store_true")
    ap.add_argument("--config", default=None)
    ap.add_argument("--clipreid-repo", default=None)
    ap.add_argument("--num-classes", type=int, default=None)
    ap.add_argument("--n-val", type=int, default=30)
    ap.add_argument("--n-test", type=int, default=150)
    a = ap.parse_args(argv)

    if Path(a.split).exists():
        print(f"[1/4] reusing frozen split {a.split}", file=sys.stderr)
    else:
        print(f"[1/4] creating split {a.split}", file=sys.stderr)
        make_split.main(["--data", a.data, "--out", a.split, "--n-val", str(a.n_val), "--n-test", str(a.n_test)])

    ext = ["--model", a.model, "--split", a.split, "--data", a.data, "--out", a.templates, "--preproc", *a.modes,
           "--batch-size", str(a.batch_size), "--num-workers", str(a.num_workers)]
    for flag, val in (("--weights", a.weights), ("--device", a.device), ("--config", a.config),
                      ("--clipreid-repo", a.clipreid_repo), ("--num-classes", a.num_classes)):
        if val is not None:
            ext += [flag, str(val)]
    if a.flip_tta:
        ext.append("--flip-tta")
    print("[2/4] extracting templates", file=sys.stderr)
    extract_templates.main(ext)

    dirs = [str(template_dir(a.templates, a.model, m)) for m in a.modes]
    common = ["--split", a.split, "--data", a.data, "--out", a.out] + (["--device", a.device] if a.device else [])
    if len(dirs) > 1:
        print("[3/4] choosing preprocessing mode on validation videos only", file=sys.stderr)
        evaluate.main(["--choose-mode", *dirs, *common])
        chosen = read_json(Path(a.out) / f"{a.model}__mode_selection_{read_json(a.split)['version']}.json")["chosen"]
    else:
        chosen = a.modes[0]
    v = read_json(a.split)["version"]
    thr_modes = [x[4:] for x in a.variants if x.startswith("thr-")] + [x for x in a.variants if x.startswith("site-gallery")]
    if thr_modes:
        print(f"[4/4] evaluating {a.model}__{chosen} (threshold per unit: {', '.join(thr_modes)})", file=sys.stderr)
        evaluate.main(["--templates", str(template_dir(a.templates, a.model, chosen)), "--bins", a.bins, *common, "--thr-mode", *thr_modes,
                      "--site-queries", str(a.site_queries), "--match-topk", str(a.match_topk)]
                      + (["--no-per-subset"] if a.no_per_subset else []))
    for variant in [x for x in a.variants if x in ("full", "plain")]:
        print(f"[4/4] evaluating {a.model}__{chosen} ({variant})", file=sys.stderr)
        evaluate.main(["--templates", str(template_dir(a.templates, a.model, chosen)), "--bins", a.bins, *common]
                      + (["--plain"] if variant == "plain" else []) + (["--no-per-subset"] if a.no_per_subset else [])
                      + (["--no-match-sheets"] if a.no_match_sheets else []) + ["--match-topk", str(a.match_topk)])
    model_dir = Path(a.out) / f"{a.model}__{chosen}"
    if not a.no_all_methods:
        print("[all methods] building the comparison of every evaluation method", file=sys.stderr)
        evaluate.main(["--all-methods", str(model_dir), "--split-version", v])
    print("\nDONE. Reports:")
    for variant in a.variants:
        print("  ", Path(a.out) / f"{a.model}__{chosen}" / (v + ("" if variant == "full" else "__plain" if variant == "plain" else "__" + variant)) / "report.md")
    if not a.no_all_methods:
        print("  ", model_dir / f"{v}__all_methods" / "report.md", " <- all methods side by side (site averages, curves)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
