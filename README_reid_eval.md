# Vehicle ReID: dataset loader, templates and within-video verification evaluation

Adds, on top of FastReID, everything needed to (1) load the crop dataset, (2) encode the validation/test videos into saved
embeddings ("templates") with any model behind one interface, and (3) run a within-video verification evaluation with
difficulty bins. The dataset is read-only; nothing here writes under `--data`.

```
reid_data/loader.py          CropRecord loader (kept crops from meta.json only)
make_split.py                frozen validation/test split         -> splits/eval_split_v1.json
reid_eval/encoders/          registry + FastReID wrapper + CLIP-ReID wrapper (+ debug encoder)
extract_templates.py         encode once, one .npz per video       -> templates/<model>__<mode>/
validate_encoder.py          wrapper check (VeRi-776 mAP, or same-vs-different tracklet pairs)
evaluate.py                  thresholds on validation, metrics on test -> results/<model>__<mode>/<split>/
compare_models.py            paired bootstrap between two evaluated models
configs/bins_v3.yaml         bin edges, FAR targets (0.1/1/2/5/10 %) + fixed threshold 0.5, minimum support, bootstrap (v2: no fixed threshold; v1: FAR 1 % and 0.1 % only)
tests/test_reid_*.py         synthetic tests (no real data or weights needed)
```

## Environment

Tested with Python 3.11 / torch 2.14 (CPU). The code avoids Python >= 3.8 and torch >= 1.12 features, but **the GPU build of torch
must support your card**: an RTX 2000 Ada / 40-series is sm_89 and needs torch with CUDA >= 11.8 (e.g. `torch>=2.0` + cu118/cu121,
Python >= 3.8). A torch that lists only up to `sm_75` in the "not compatible" warning will fail at the first kernel
("no kernel image is available"); use `--device cpu` or a newer env, e.g.

```bash
conda create -n reid python=3.10 -y && conda activate reid
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install yacs termcolor tabulate scikit-learn opencv-python pandas matplotlib pyyaml pillow
```

FastReID model-zoo checkpoints (older format: `heads.classifier.weight`, stored `pixel_mean/std`) load directly; the differences are
checked and listed under `notes` in the manifest. `faiss` is not needed.

## What `--data` is, and the split

`--data` is the **output folder of `build_reid_crops.py`** (`--out`, the one containing `videos/<id>/meta.json`), not the raw
`--root` with Videos/annotations/calibration. The dataset has no split of its own: `make_split.py` derives
`splits/eval_split_v1.json` from the per-video `meta.json` files (eligibility, site-disjoint validation, seeded shuffle).
The split is checked against the dataset by a fingerprint of the **kept crops** of its videos, not by `summary.json`
(which `build_reid_crops.py` rewrites on every run). If `quality_filter.py` / `filter_roi.py` / a rebuild changes the kept crops
of a split video, extraction and evaluation stop and ask for a new split version.

## One command

```bash
python run_full_eval.py --data D:/data/Re-ID_safe_test --model fastreid_veriwild_r50ibn --weights veriwild_bot_R50-ibn.pth
```

Runs split (reused if it exists) -> templates for both modes -> mode chosen on validation -> evaluation -> `report.md`; re-running resumes.

## Run order (step by step)

```bash
D=D:/data/Re-ID_safe_test                      # dataset root

# 1. split (once; refuses to overwrite; --check re-derives and compares)
python make_split.py --data $D --out splits/eval_split_v1.json

# 2. validate the wrapper (published number within ~1 mAP, or same-tracklet > different-vehicle)
python validate_encoder.py --model fastreid_veriwild_r50ibn --weights veriwild_bot_R50-ibn.pth --data $D --split splits/eval_split_v1.json

# 3. templates, both preprocessing modes (validation is enough to choose the mode)
python extract_templates.py --model fastreid_veriwild_r50ibn --weights veriwild_bot_R50-ibn.pth \
    --split splits/eval_split_v1.json --data $D --out templates/ --preproc letterbox unpad_stretch

# 4. choose the preprocessing mode on VALIDATION (higher pooled AUC), then evaluate that mode
python evaluate.py --choose-mode templates/fastreid_veriwild_r50ibn__letterbox templates/fastreid_veriwild_r50ibn__unpad_stretch \
    --split splits/eval_split_v1.json --data $D --out results/
python evaluate.py --templates templates/fastreid_veriwild_r50ibn__unpad_stretch --split splits/eval_split_v1.json \
    --data $D --bins configs/bins_v3.yaml --out results/
#    ... add --plain for the evaluation without the difficulty criteria

# 5. rebuild report.md from the saved files only
python evaluate.py --report-only results/fastreid_veriwild_r50ibn__unpad_stretch/v1 --data $D

# 6. second model + paired comparison
python compare_models.py --a results/<A>__<mode>/v1 --b results/<B>__<mode>/v1 --out results/compare_A_vs_B_v1
```

`pytest tests/test_reid_*.py tests/test_validate_encoder.py` runs the synthetic checks (about 1.5 min on CPU).

## Models

`--model` names (see `reid_eval/encoders`): `fastreid_veriwild_r50ibn`, `fastreid_veri_sbs_r50ibn`, `clipreid_vit_veri`,
`debug_colorgrid` (pipeline tests only). Add a model by writing a class with `name`, `input_size`, `preprocess`, `encode`,
`describe` and registering it with `@register("<name>")`; extraction and evaluation contain no model-specific code.

* **FastReID**: the feature is `Baseline.forward` in eval mode = `EmbeddingHead` BN-neck feature, the same tensor
  `ReidEvaluator` uses (`MODEL.HEADS.NECK_FEAT` only selects the *training* feature). Size 256x256 and ImageNet mean/std come from the
  repo's config; FastReID normalises inside the model (0-255 scale), so `preprocess` returns raw RGB 0-255. Resize is PIL bicubic like
  the repo's test transform. The checkpoint is loaded **strictly**, `NUM_CLASSES` is read from the checkpoint, no ImageNet weights are
  downloaded, no AMP. `fastreid.engine` is never imported, so `faiss` is not needed.
* **CLIP-ReID**: wraps the repo's own `make_model` / `model(img)` with SIE off and camera/view labels `None`; CLIP mean/std. It needs
  `--clipreid-repo` (or `$CLIPREID_REPO`), `--num-classes` matching the checkpoint, and a checkpoint trained **without** SIE.
* Licences: weights trained on VeRi-776 / VehicleID / VERI-Wild are research-only. `checkpoint_source` in the manifest is a reminder
  field to fill in the download URL and licence.

## Two evaluations, several FAR points, per site and per video

`run_full_eval.py` runs **two evaluations** on the same templates and the same validation thresholds:

| variant | folder | what it contains |
|---|---|---|
| `full` | `results/<model>__<mode>/<split>/` | all difficulty criteria: delta position, delta azimuth, occlusion, keypoints (curves, heatmaps, joint cells) |
| `plain` | `results/<model>__<mode>/<split>__plain/` | NO pose / occlusion / keypoint criteria: every pair counts; global ROC, TAR at the FAR targets, per site, per video. Computed from neutral metadata, so it does not depend on calibration |

`plain` gives exactly the same global numbers as `full` (tested), because both use the same thresholds; it just does not slice by difficulty.
To also include videos **without calibration** in the plain evaluation, make a separate split
(`make_split.py --no-require-calibration --out splits/eval_split_v2.json`) and run that split.

TAR is reported at several FAR operating points (`configs/bins_v2.yaml`): **0.1 %, 1 %, 2 %, 5 %, 10 %** (10^-3 = 0.1 %, 10^-2 = 1 %,
10^-1 = 10 %, so those requests coincide). Every threshold is set on the pooled validation negatives; the FAR actually measured on test is
shown next to each target. Column names: `tar_at_0.1pct`, `far_at_1pct`, ... Change the list in a new `bins_v3.yaml`, never in v2.

The **same figures and tables are produced for every site and every video**: `per_site/<site>/` and `per_video/<video_id>/`
each hold `report.md`, `summary.json`, `roc.csv`, `figures/` (ROC, 4 difficulty curves, heatmaps) and `bins_*.csv`; the main report links
to them from its per-site and per-video tables. Bootstrap CIs need at least 5 videos, so they exist globally and for large sites only;
single videos have none, and most of their bins fall below the minimum support (greyed). `--no-per-subset` skips these reports (faster).

## Accuracy at a fixed threshold and at every FAR threshold

`configs/bins_v3.yaml` (the default) adds a **fixed cosine threshold of 0.5** next to the five FAR thresholds (0.1, 1, 2, 5, 10 %). Every
operating point gets TAR, FAR, FRR, **accuracy** and **balanced accuracy** in every table (global, per bin, per site, per video, with bootstrap
CIs where there are >= 5 videos): columns `acc_at_th0.5`, `bacc_at_th0.5`, `acc_at_1pct`, ... (`pooled_*` and `balanced_*` prefixes in the bin CSVs).
All of them are exact counts, checked against a brute-force computation in `tests/test_reid_accuracy.py`.

* accuracy = (positive pairs accepted + negative pairs rejected) / all pairs. About 85 % of the pairs are negatives, so it mostly measures the
  negatives; **balanced accuracy** = (TAR + (1 - FAR)) / 2 is the number to compare. The object-balanced variants weight objects equally.
* `roc.csv` (global, per site, per video) also holds accuracy and balanced accuracy at **every** threshold (2,000 steps), and
  `figures/accuracy_vs_threshold.png` draws them with the operating points as vertical lines.
* The fixed threshold is not tuned on anything. To choose another one, add it to `fixed_thresholds` in a new bins file (e.g. `bins_v4.yaml`).
  A threshold that maximises accuracy would have to be chosen on the validation videos, never on test.

## Threshold per video and per site (extra evaluations)

In addition to `full` and `plain` (one global threshold per FAR target, set on the validation videos), `run_full_eval.py` runs two
evaluations with **a threshold per video** and **a threshold per site**. A threshold calibrated on the data it is then measured on is
tuning on test, so the default protocols hold the calibration data out of the measurement:

| folder | protocol |
|---|---|
| `<split>__thr-video` | each video's objects are split in two folds (seeded, by tracklet); fold A is evaluated with the threshold set on the negative pairs of fold B and vice versa; pairs across folds are not evaluated. Videos with < 4 objects are skipped |
| `<split>__thr-site` | leave-one-video-out: each test video is evaluated with the threshold set on the negatives of the OTHER test videos of its site. A site with one test video is skipped (the validation videos cannot be used: the split is site-disjoint) |
| `<split>__thr-video-oracle`, `__thr-site-oracle` | opt-in (`--variants ... thr-video-oracle thr-site-oracle`). The threshold is set on the evaluated data itself (FAR is forced to the target): an optimistic **upper bound**, labelled as such in the report |

They use the plain kind (no pose / occlusion / keypoint criteria) and the same FAR targets and fixed threshold; each has global,
per-site and per-video tables/figures like the other evaluations. Per-unit thresholds and the number of calibration negatives are in
`thresholds_per_unit.csv`; videos that could not be evaluated are listed in the report. `<split>__threshold_comparison.md/.csv`
(in the model folder) puts the protocols side by side. Caveats: a single video has few negatives, so strict-FAR thresholds (0.1 %) are noisy;
and the rows do not evaluate exactly the same pairs, so differences mix the effect of the threshold with the change of pairs.
Only these two run by default; choose with `--variants full plain thr-video thr-site`, or run them alone with
`evaluate.py --templates ... --thr-mode video site`.

## Per-object match images (top-10 positives and negatives)

For **every object of every test video** the `full` evaluation writes one image, `results/<model>__<mode>/<split>/matches/<video>/<object>.png`
(plus `index.csv` and `index.md` per video, objects listed most confusable first; the per-video reports link to them). The query is the
object's *medoid* crop (the crop most similar, on average, to its other crops). Rows: **top positives** (10 most similar crops of the same
object), **hardest positives** (10 least similar, only when the object has more than 10 positives), **top negatives** (10 most similar
crops of other objects, labelled with the other object's id). Each tile shows its cosine similarity and a flag against the global threshold at
FAR 1 %: positive below it = `FR` (false reject, orange), negative at or above it = `FA` (false accept, red). `index.csv` has, per object,
the best/worst positive, the top negative and its object, the number of FR/FA of that query and the separation (lowest positive - highest
negative). It shows one query per object, so it is for understanding failures, not a metric. Options: `--match-topk N`,
`--no-match-sheets` (skip; faster), `--match-sheets` (also in the plain variant). About 4,000 objects x ~30 crops are read once per video.

## Output layout

```
splits/eval_split_v1.json                         frozen; never edit (dataset changed -> eval_split_v2.json)
templates/<model>__<mode>/manifest.json           model, repo commit, checkpoint sha256, preprocessing, split sha256, notes, throughput
templates/<model>__<mode>/{validation,test}/<video>.npz    emb (raw, float32), crop_uid, tracklet_id, frame; canonical record order
results/<model>__<mode>/<split_version>/
    thresholds.json  summary.json  report.md  figures/  run.json
    bins_<axis>.csv  heatmap_<a>_x_<b>.csv/.png  cells.csv  per_video.csv  per_object.csv
    failures/ (negatives_highest_similarity, positives_lowest_similarity_dpos_lt_1m: CSV + contact sheets)
    acc/{validation,test}/<video>.npz              per-video accumulators; every breakdown is a sum over their joint cells
```

Per-video accumulator (`acc/...npz`): sparse joint cells `cell_ids` over (delta position, delta azimuth, occlusion, keypoint IoU), each
with a 200-bin histogram per pair type (plain `hist` and object-balanced `hist_bal`), exact counts at both thresholds (`cnt`,
`cnt_bal`), first and second moments, a 2,000-bin fine histogram (`fine`, `fine_bal`), distinct objects / object pairs per cell
(`pos_keys`, `neg_keys`), per-object stats (`obj_*`) and per-object-pair median/max (`pair_*`), plus the worst pairs (`fail_*`).

## Choices the spec left open (check they suit you)

* **Cell count.** The occlusion axis has the 3 bins of the spec table (both visible / one / both occluded), so there are
  8 x 7 x 3 x 5 = 840 joint cells, not 1,120.
* **Histogram resolution.** Per-bin AUC, EER, best-TAR, median/p5/p95 come from 200-bin histograms (about 0.01 in cosine, linear
  interpolation inside a bin); the global AUC and the thresholds use the 2,000-bin histogram. TAR/FAR/FRR at the thresholds are
  exact counts. The thresholds are interpolated inside a 0.001-wide bin, so on a large validation set the measured validation FAR lands
  within a few percent (relative) of the target (synthetic check: 1.003e-3 for a 1e-3 target); the FAR actually measured on
  validation and test is always reported next to the target.
* **Validation negatives for thresholds** are pooled plain pairs (as in the spec), not object-balanced.
* **Site-disjoint validation.** Whole sites are added in seeded random order until >= `--n-val` videos, skipping a site that would
  push validation beyond 1.5x `--n-val`. If that is impossible the split falls back to random videos, sets `site_disjoint: false`
  and prints a warning (also stored in the split's `warnings`).
* **Test videos** that are eligible but fewer than 150: all are taken, with a loud warning.
* **`created` timestamp** in the split file is the only non-deterministic field; two runs with the same seed give identical files
  apart from it (`--created` fixes it; `--check` compares ignoring it).
* **Shared-keypoints IoU** is unknown if either crop has no keypoints *or* both have zero visible keypoints (0 / 0).
* **Objects falsely matched** (per_object.csv) = other objects with at least one negative pair >= t(1e-3). **Confused object pair** =
  median similarity of the object pair > t(1e-3); medians and maxima are exact.
* **Bootstrap** is over videos only (percentile CI, seed in the bins file). The optional second level (resampling objects inside a
  video) is **not implemented**. Video-averaged CIs are given for TAR and AUC only.
* **Keypoints**: parsed in one function, `reid_data.loader.parse_keypoints`; a missing field is `None` ("unknown"). The secondary
  breakdown by minimum visible keypoints is written to `bins_min_visible_keypoints.csv` when keypoints exist.

## Known limitations

See section 10 of the spec: the occlusion axis is almost empty on the default dataset (quality control removes occluded crops),
the keypoint axis is all "unknown" until keypoints are added, within-video results are optimistic, and tracklet = vehicle is an
assumption: review `failures/` before trusting any number.
