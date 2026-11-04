# Evaluation metrics

`mammography_robustness.evaluation` reads the prediction CSVs described in [PREDICTION_OUTPUT.md](PREDICTION_OUTPUT.md) and computes AUC, sensitivity, specificity and F1-score for every model, training seed and image-quality condition. It does not train models, produce predictions or draw figures.

## Usage

```bash
# Experiment 001: clean baseline only, breast level (primary)
python -m mammography_robustness.evaluation outputs/test_breast_predictions.csv \
    --threshold youden --validation outputs/val_breast_predictions.csv --clean-only

# Full robustness evaluation, breast level, with the exporter's breast metadata
python -m mammography_robustness.evaluation outputs/test_breast_predictions.csv \
    --metadata outputs/test_breast_metadata.csv \
    --threshold youden --validation outputs/val_breast_predictions.csv

# Image level (secondary)
python -m mammography_robustness.evaluation outputs/test_image_predictions.csv \
    --threshold youden --validation outputs/val_image_predictions.csv
```

The level (image or breast) is detected from the CSV header. Results are written to `--out-dir` (default `outputs/metrics/`):

- `<level>_metrics_by_run.csv`: one row per subset, model, seed and condition
- `<level>_metrics_summary.csv`: mean and SD across seeds
- `<level>_evaluation_metadata.json`: the input files, including any `--metadata` file (path and SHA-256 hash), the threshold chosen for each model/seed, and the settings used

Other options: `--metadata <csv>` (see below), `--clean-only` (evaluate only the clean rows), `--bootstrap N` (default 2000; `0` disables intervals), `--confidence` (default 0.95), `--bootstrap-seed` (default 0). A later run with the same `--out-dir` and level overwrites the earlier files.

## Protocol (agreed)

- **Evaluation unit.** Breast level is the primary analysis. The breast score is the maximum score across the breast's available CC and MLO views, as exported. Image level is a secondary analysis.
- **Positive call.** A score at or above the threshold is a malignant call (label 1). Normal and benign are both negative (label 0).
- **Threshold.** Youden's J (sensitivity + specificity − 1) on **clean validation** predictions, chosen separately for each model and training seed (`--threshold youden --validation <csv>`). Candidates are the observed scores; ties go to the highest threshold. The threshold is then fixed for the clean test set and every degradation and severity, and is never re-tuned on test or degraded data. The validation CSV must be the same level as the test CSV. A fixed threshold (`--threshold 0.5`) is still available for exploratory use.
- **AUC** is threshold-free (Mann–Whitney form; tied scores count one half).
- **Undefined values** (e.g. sensitivity with no positive cases, AUC with one class) are left empty (NaN) rather than set to 0.
- **Deltas** are `degraded − clean` for the same model, seed and subset, so negative values mean the degradation hurt performance. Clean rows have delta 0.
- **Confidence intervals** use a percentile bootstrap that resamples **patients** with replacement, keeping all of a patient's breasts or images together. Each resample is reused for every condition, so the delta intervals are paired. Resamples where a metric is undefined are ignored for that metric. Resampling is seeded per model, seed and subset, so results are reproducible and do not depend on which other models are in the file.
- **Seeds.** Three training seeds per model (agreed). The summary file gives the mean and sample SD across seeds; the SD is empty for a single seed. Bootstrap intervals are per run and are not pooled across seeds.

## Optional columns and subsets

These columns are optional. They can appear anywhere in the prediction CSV, or come from a separate metadata CSV passed with `--metadata` (see below). Their names are set at the top of `evaluation.py` (`VIEW_STATUS`, `ORIGINAL_CLASS`, `LABEL_FLAG`).

| Column | Values | Effect |
| --- | --- | --- |
| `view_status` | `single`/`paired` or `single_view`/`paired_view` | Adds `paired_view` and `single_view` subset rows |
| `original_class` | `normal`, `benign`, `malignant` | Fills `n_normal`, `n_benign`, `fp_normal`, `fp_benign` |
| `label_flag` (or `view_label_conflict` in metadata) | `0`/`1` (or `false`/`true`) | Adds an `unflagged` subset that excludes flagged units (sensitivity analysis) |

### Metadata CSVs from the exporter

The exporter keeps the agreed prediction CSVs unchanged and writes these fields to separate files (`write_breast_metadata_csv`, `write_image_metadata_csv`; see [PREDICTION_OUTPUT.md](PREDICTION_OUTPUT.md)). Pass the matching file with `--metadata`. Rows are matched on the breast (or image: patient, side, view) plus model, seed, degradation and severity.

- The two files must cover exactly the same rows; a missing, extra or duplicated metadata row is an error.
- If the metadata has a `label` column, it must agree with the predictions. This catches pairing the wrong files.
- `view_label_conflict` is read as `label_flag`, so the `unflagged` subset excludes breasts whose CC and MLO labels differ.
- Other metadata columns (`image_id`, `score`, `case_id`) are ignored.
- A field may not appear in both files.

The exporter's breast metadata currently has `view_status` and `view_label_conflict`; the image metadata has `original_class`. The breast-level normal/benign false-positive split needs `original_class` added to the breast metadata.

The `subset` column is `all` for the main result. Every subset uses the same per-run threshold. Subsets can be small: the current test split has 20 single-view breasts with 3 cancers, so check `n_positive` before reading single-view sensitivity or its interval.

## Input checks

- The agreed columns are present, and any other column is one of the optional ones above.
- Labels are 0/1 and scores are finite numbers in [0, 1].
- The baseline uses `clean`/`none`; degradations use `low`/`medium`/`high`.
- There are no duplicate rows.
- The label (and any optional column) has one value per breast or image.
- `label` is 1 exactly when `original_class` is `malignant`.
- At image level, `view_status` matches the number of views present for that breast.
- Within each model/seed there is a clean baseline, and every condition covers the same breasts or images as that baseline.

## Output columns

`<level>_metrics_by_run.csv`:

| Column | Meaning |
| --- | --- |
| `level`, `subset`, `model`, `seed`, `degradation`, `severity` | Identify the subset, run and condition |
| `threshold`, `threshold_source` | Threshold applied, and `youden_validation` or `fixed` |
| `n_patients`, `n_units`, `n_positive`, `n_negative` | Sample size (units = breasts or images) |
| `tp`, `fp`, `tn`, `fn` | Confusion counts at the threshold |
| `n_normal`, `n_benign`, `fp_normal`, `fp_benign` | Negatives and false positives split by original class (empty without `original_class`) |
| `auc`, `sensitivity`, `specificity`, `f1` | Point estimates |
| `<metric>_ci_low`, `<metric>_ci_high` | Bootstrap interval |
| `delta_<metric>`, `delta_<metric>_ci_low`, `delta_<metric>_ci_high` | Change from clean, with paired interval |
| `n_bootstrap`, `confidence` | Bootstrap settings used |

`<level>_metrics_summary.csv`: one row per subset, model and condition, with `n_seeds`, `threshold_source`, and `<metric>_mean`, `<metric>_sd`, `delta_<metric>_mean` and `delta_<metric>_sd` for each metric.

## Checks and limits

```bash
python -m unittest discover -s tests -v
```

Requires Python 3.10+, NumPy and pandas. The tests use synthetic records only. With the current test split (49 patients) the intervals will be wide, so report them alongside point estimates. The `--validation` CSV must come from the validation partition, never the test partition; the code cannot check this. With 2000 resamples and all subsets present, expect roughly 30 s per model/seed run.
