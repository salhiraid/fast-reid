#!/usr/bin/env python
"""Paired-bootstrap comparison of two evaluated models (same split file, same bins file).

  python compare_models.py --a results/<modelA>__<mode>/v1 --b results/<modelB>__<mode>/v1 --out results/compare_A_vs_B_v1
"""
import argparse
import sys
from pathlib import Path

from reid_eval.compare import compare, to_markdown


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--bins", default="configs/bins_v1.yaml")
    ap.add_argument("--out", required=True, help="output prefix: writes <out>.csv and <out>.md")
    a = ap.parse_args(argv)
    df, meta = compare(a.a, a.b, a.bins)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out.with_suffix(".csv"), index=False)
    out.with_suffix(".md").write_text(to_markdown(df, meta), encoding="utf-8")
    print(f"wrote {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
