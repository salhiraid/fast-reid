"""Universal evaluation runner: give it a dataset root and a TEMPLATES folder holding the templates of one or several models.

    templates/
        <model_a>__<mode>/manifest.json, validation/<video>.npz, test/<video>.npz     (see TEMPLATE_FORMAT.md)
        <model_b>__<mode>/...
For each model it picks the preprocessing mode on the validation videos (if several modes exist), runs the evaluation variants, writes the
per-model all-methods report, and finally the model comparison tables/curves. No model code is needed: only the saved templates.
"""
from __future__ import annotations

import sys
from pathlib import Path

from reid_data import group_by_video, load_dataset
from . import evaluation
from .all_methods import build_all_methods
from .common import read_json, sha256_file
from .model_compare import build_model_comparison

DEFAULT_VARIANTS = ("full", "plain", "thr-video", "thr-site", "site-gallery", "thr-global-oracle")
ALL_VARIANTS = DEFAULT_VARIANTS + ("thr-video-oracle", "thr-site-oracle", "site-gallery-oracle")


def find_models(templates_dir):
    """{model_name: {preproc_mode: template folder}} for every sub-folder with a manifest.json and a test/ folder."""
    root = Path(templates_dir)
    if (root / "manifest.json").is_file() and (root / "test").is_dir():   # a single template folder was given
        root_dirs = [root]
    else:
        root_dirs = sorted(d for d in root.iterdir() if d.is_dir() and (d / "manifest.json").is_file() and (d / "test").is_dir())
    models = {}
    for d in root_dirs:
        man = read_json(d / "manifest.json")
        name = man.get("model_name") or d.name.split("__")[0]
        mode = man.get("preproc_mode") or (d.name.split("__")[1] if "__" in d.name else "default")
        if mode in models.setdefault(name, {}):
            raise ValueError(f"two template folders for model {name!r}, mode {mode!r}: {models[name][mode]} and {d}")
        models[name][mode] = d
    if not models:
        raise FileNotFoundError(f"no template folder (manifest.json + test/) found in {templates_dir}")
    return models


def resolve_split(split_arg, templates_dir, manifests):
    """--split, else an eval_split*.json in the templates folder, else the split_file recorded in a manifest. Checks the sha256."""
    cands = []
    if split_arg:
        cands.append(Path(split_arg))
    root = Path(templates_dir)
    cands += sorted(root.glob("eval_split*.json")) + sorted(root.parent.glob("eval_split*.json")) if not split_arg else []
    cands += [Path(m["split_file"]) for m in manifests if m.get("split_file")] if not split_arg else []
    for c in cands:
        if c.is_file():
            sha = sha256_file(c)
            wanted = {m.get("split_sha256") for m in manifests if m.get("split_sha256")}
            if wanted and sha not in wanted:
                raise ValueError(f"split file {c} (sha256 {sha[:12]}...) is not the one the templates were extracted with "
                                 f"({', '.join(w[:12] + '...' for w in wanted)}). Give the original split with --split.")
            return c
    raise FileNotFoundError("split file not found. Give --split <the eval_split_*.json used to extract the templates> "
                            "(or copy it into the templates folder).")


def run_all(data, templates, out="results", split=None, models=None, variants=DEFAULT_VARIANTS, bins="configs/bins_v3.yaml", device=None,
            per_subset=True, match_sheets=True, site_queries=10, match_topk=10, all_modes=False, compare=True, reference=None, verbose=True):
    """Evaluate every model of the templates folder, then compare them. Returns {"results": {model_label: folder}, "comparison": path}."""
    found = find_models(templates)
    if models:
        missing = [m for m in models if m not in found]
        if missing:
            raise ValueError(f"model(s) {missing} not in {templates}; available: {sorted(found)}")
        found = {m: found[m] for m in models}
    manifests = [read_json(d / "manifest.json") for modes in found.values() for d in modes.values()]
    split_path = resolve_split(split, templates, manifests)
    split_json = read_json(split_path)
    version = split_json["version"]
    variants = list(variants)
    bins = str(Path(bins).resolve())          # stored in run.json: must not depend on the working directory
    results = {}
    for name, modes in found.items():
        # ---- preprocessing mode: all of them, or the best one on the VALIDATION videos
        if len(modes) > 1 and not all_modes:
            if all((d / "validation").is_dir() for d in modes.values()):
                by_val = group_by_video(load_dataset(data, video_ids=split_json["validation"], verbose=False))
                sel = evaluation.select_mode(list(modes.values()), split_json, by_val, device)
                chosen = [sel["chosen"]]
                if verbose:
                    print(f"[{name}] preprocessing mode chosen on validation: {chosen[0]}  {sel['validation_pooled_auc']}", file=sys.stderr)
            else:
                chosen = [sorted(modes)[0]]
                print(f"WARNING: [{name}] no validation templates to choose a mode: using {chosen[0]!r}", file=sys.stderr)
        else:
            chosen = sorted(modes)
        for mode in chosen:
            tdir = modes[mode]
            has_val = (tdir / "validation").is_dir()
            label = f"{name}__{mode}"
            if verbose:
                print(f"\n=== {label} ===", file=sys.stderr)
            std = [v[4:] for v in variants if v.startswith("thr-")]
            gal = [v for v in variants if v.startswith("site-gallery")]
            for v in [x for x in variants if x in ("full", "plain")]:
                if not has_val:
                    print(f"WARNING: [{label}] no validation templates: skipping the global-threshold variant {v!r}", file=sys.stderr)
                    continue
                evaluation.run(tdir, split_path, data, bins, out, device, verbose, plain=(v == "plain"), per_subset=per_subset,
                               match_sheets=(match_sheets if v == "full" else False), match_topk=match_topk)
            if std:
                evaluation.run_threshold_modes(tdir, split_path, data, bins, out, std, device, verbose, per_subset=per_subset,
                                               site_queries=site_queries, match_topk=match_topk)
            if gal:
                evaluation.run_site_gallery(tdir, split_path, data, bins, out, gal, device, verbose, per_subset=per_subset)
            model_dir = Path(out) / label
            build_all_methods(model_dir, version)
            results[label] = model_dir
    comparison = None
    if compare:
        comparison = build_model_comparison(out, version, models=list(found) if models else None, reference=reference)
        if verbose and comparison:
            print(f"\nmodel comparison: {comparison}", file=sys.stderr)
    return {"results": results, "comparison": comparison, "split": str(split_path), "version": version}
