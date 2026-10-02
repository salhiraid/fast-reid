# Universal ReID evaluation kit

Copy this folder into any repository. Give it the dataset root and a **templates folder** (one sub-folder of saved embeddings per model, see
`TEMPLATE_FORMAT.md`): it evaluates every model on the same pairs and writes reports, tables, curves and a **model comparison table**.
No model code is needed (no FastReID / CLIP / PyTorch model): only the saved templates.

```bash
pip install -r requirements.txt
python selftest.py --quick                      # checks the install on synthetic data (no dataset, no model needed)

python make_split.py --data DATASET --out splits/eval_split_v1.json          # once, before extracting templates
python export_templates.py --data DATASET --split splits/eval_split_v1.json --out templates \
       --model-name mymodel --preproc letterbox unpad_stretch --encoder mypackage.mymodule:encode      # or write the npz yourself

python run_eval.py --list templates/                                          # which models / modes were found
python run_eval.py --data DATASET --templates templates/ --split splits/eval_split_v1.json --out results/
python run_eval.py --compare results/ --reference mymodel__unpad_stretch      # comparison only (tables + curves)
```

`run_eval.py` runs, for **each model** in `templates/` (the preprocessing mode is chosen on the validation videos only; `--all-modes` keeps all):

| variant | what it is |
|---|---|
| `full` | one threshold per FAR target (0.1 / 1 / 2 / 5 / 10 %) + a fixed threshold 0.5, set on the **validation** videos; pairs within each video; with the difficulty bins (delta position, delta azimuth, occlusion, keypoints) |
| `plain` | the same without the difficulty criteria (global, per site, per video) |
| `thr-global-oracle` | the same single global threshold, but set on **all test videos pooled** (optimistic; the static counterpart of the oracles below, on the same pairs as `full`) |
| `thr-video`, `thr-site` | a threshold per video / per site, **held out** (no tuning on the measured pairs); within-video pairs |
| `site-gallery` | matching against the **whole site** (all crops of all the site's videos; cross-video pairs assumed negative), threshold per site, held out |
| `*-oracle` (opt-in, `--variants ...`) | the same thresholds tuned on the evaluated data: optimistic upper bounds |

Per model: `results/<model>__<mode>/<split>__threshold_comparison.csv` (one row per method: TAR / recall, precision, accuracy, balanced accuracy and the
threshold at every FAR target 0.1-10 %, refreshed after every evaluation so it is complete in any run order) and
`results/<model>__<mode>/<split>[__variant]/report.md` (metrics, accuracy, curves, per-site/per-video tables, failures, match images) and
`results/<model>__<mode>/<split>__all_methods/report.md` (all methods side by side, averaged over sites).
**Model comparison:** `results/comparison_<split>/report.md`: for every method, a table with one row per model (TAR at each FAR, AUC, EER, accuracy /
balanced accuracy, averaged over sites ± std and pooled with CIs; best value in bold), a paired bootstrap of the differences to a reference model,
a models x methods matrix, ROC / TAR-vs-FAR curves per model, and `models_summary.csv` / `models_per_site.csv`.

Notes
* All templates must come from the **same split file** (checked through `split_sha256`); give it with `--split` or copy it next to the templates.
* Metrics are TAR/FAR/accuracy at thresholds (exact counts), AUC/EER from histograms, CIs from a bootstrap over videos (never over pairs). Plain
  accuracy depends on the share of positive pairs: compare **balanced accuracy** across methods.
* Speed: a GPU makes the pairwise matrices fast (`--device cuda`); `--no-per-subset --no-match-sheets` skip the per-site/per-video reports and images.
* Other entry points: `evaluate.py` (one variant, `--thr-mode`, `--all-methods`), `compare_models.py` (paired bootstrap of two evaluated models).
* The folder provides the Python packages `reid_data/` and `reid_eval/`; do not copy it into a repository that already has packages of these names.
