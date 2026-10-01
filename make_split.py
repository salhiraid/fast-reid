#!/usr/bin/env python
"""Create the frozen eval split once (spec section 5). Refuses to overwrite an existing split file.

  python make_split.py --data D:/data/Re-ID_safe_test --out splits/eval_split_v1.json
  python make_split.py --data ... --out splits/eval_split_v1.json --check    # re-derive and compare, never writes
"""
import argparse
import sys

from reid_eval.common import read_json, write_json_atomic
from reid_eval.split import build_split


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="dataset root")
    ap.add_argument("--out", default="splits/eval_split_v1.json")
    ap.add_argument("--version", default=None, help="default: taken from the file name (eval_split_<version>.json)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-val", type=int, default=30)
    ap.add_argument("--n-test", type=int, default=150)
    ap.add_argument("--min-tracklets", type=int, default=2)
    ap.add_argument("--min-crops-per-tracklet", type=int, default=2)
    ap.add_argument("--min-crops", type=int, default=20)
    ap.add_argument("--no-require-calibration", action="store_true")
    ap.add_argument("--created", default=None, help="fixed timestamp (only to make two runs byte-identical)")
    ap.add_argument("--check", action="store_true", help="re-derive the split and compare to --out (ignoring `created`)")
    a = ap.parse_args(argv)

    version = a.version or (a.out.rsplit("eval_split_", 1)[-1].rsplit(".json", 1)[0] if "eval_split_" in a.out else "v1")
    split = build_split(a.data, version=version, seed=a.seed, n_val=a.n_val, n_test=a.n_test, created=a.created,
                        criteria={"min_tracklets": a.min_tracklets, "min_crops_per_tracklet": a.min_crops_per_tracklet,
                                  "min_crops": a.min_crops, "require_calibration": not a.no_require_calibration})
    for w in split["warnings"]:
        print(f"\n*** WARNING: {w}\n", file=sys.stderr)
    print(f"eligible {split['n_videos_eligible']}/{split['n_videos_total']} videos; site_disjoint={split['site_disjoint']}; "
          f"counts={split['counts']}")
    assert not set(split["validation"]) & set(split["test"]), "validation and test overlap"

    if a.check:
        old = read_json(a.out)
        old.pop("created", None); split.pop("created", None)
        same = old == split
        print("IDENTICAL to " + a.out if same else "DIFFERENT from " + a.out)
        return 0 if same else 1
    from pathlib import Path
    if Path(a.out).exists():
        sys.exit(f"{a.out} exists. Splits are frozen: never regenerate or edit one. If the dataset changed, "
                 f"write eval_split_v2.json instead (or use --check).")
    write_json_atomic(a.out, split)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
