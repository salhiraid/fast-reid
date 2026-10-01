#!/usr/bin/env python
"""Whole pipeline for one model in one command (cross-platform):
split (created once, reused if it exists) -> templates for both preprocessing modes -> choose the mode on VALIDATION
-> evaluate the chosen mode -> report.md.

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
    ap.add_argument("--bins", default="configs/bins_v1.yaml")
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
    print(f"[4/4] evaluating {a.model}__{chosen}", file=sys.stderr)
    evaluate.main(["--templates", str(template_dir(a.templates, a.model, chosen)), "--bins", a.bins, *common])
    v = read_json(a.split)["version"]
    print(f"\nDONE. Report: {Path(a.out) / f'{a.model}__{chosen}' / v / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
