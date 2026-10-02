#!/usr/bin/env python
"""Write templates from ANY encoder: you only provide one Python function (see reid_eval/export.py).

  python export_templates.py --data DATASET --split splits/eval_split_v1.json --out templates --model-name mymodel \
      --preproc unpad_stretch --encoder mypackage.mymodule:encode
Use --preproc letterbox unpad_stretch to write both modes (the evaluation then picks one on the validation videos).
"""
import argparse
import sys

from reid_eval.export import export_templates, load_callable
from reid_eval.preprocess import MODES


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="dataset root (folder with videos/<id>/meta.json)")
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", default="templates")
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--encoder", required=True, help="module:function, function(list of BGR uint8 images) -> (N, D) array")
    ap.add_argument("--preproc", nargs="+", default=["unpad_stretch"], choices=MODES)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    fn = load_callable(a.encoder)
    for mode in a.preproc:
        print(export_templates(a.data, a.split, a.out, a.model_name, fn, mode, a.batch_size, overwrite=a.overwrite))
    return 0


if __name__ == "__main__":
    sys.exit(main())
