# Evaluation metrics

`mammography_robustness.evaluation` reads the prediction CSVs described in [PREDICTION_OUTPUT.md](PREDICTION_OUTPUT.md) and computes AUC, sensitivity, specificity and F1-score for every model, training seed and image-quality condition. It does not train models, produce predictions or draw figures.

## Usage

```bash
# Breast level (primary), threshold chosen on clean validation predictions
python -m mammography_robustness.evaluation outputs/test_breast_predictions.csv \
    --threshold youden --validation outputs/val_breast_predictions.csv

# Image level (secondary), fixed threshold
python -m mammography_robustness.evaluation outputs/test_image_predictions.csv --threshold 0.5
```

The level (image or breast) is detected from the CSV header. Results are written to `--out-dir` (default `outputs/metrics/`) as `<level>_metrics_by_run.csv` and `<level>_metrics_summary.csv`. Other options: `--bootstrap N` (default 2000; `0` disables intervals), `--confidence` (default 0.95), `--bootstrap-seed` (default 0).

## Protocol

- **Evaluation unit.** Breast level is the primary analysis (score = maximum of the CC and MLO scores, as exported). Image level is reported as a secondary analysis.
- **Positive call.** A score at or above the threshold is a malignant call (label 1).
- **Threshold.** The group has not yet chosen the threshold rule, so both options are supported:
  - `--threshold <number>`: one fixed threshold for every model and seed (`threshold_source = fixed`).
  - `--threshold youden --validation <csv>`: for each model/seed, the threshold maximising sensitivity + specificity − 1 on that run's **clean validation** predictions, among observed scores; ties go to the highest threshold (`threshold_source = youden_validation`). The validation CSV must be the same level as the test CSV.

  In both cases the threshold is fixed per model/seed and reused for the clean test set and every degraded condition. Degradations never get their own threshold.
- **AUC** is threshold-free (Mann–Whitney form; tied scores count one half).
- **Undefined values** (e.g. sensitivity with no positive cases, AUC with one class) are left empty (NaN) rather than set to 0.
- **Deltas** are `degraded − clean` for the same model and seed, so negative values mean the degradation hurt performance. Clean rows have delta 0.
- **Confidence intervals** use a percentile bootstrap that resamples **patients** with replacement, keeping all of a patient's breasts or images together. Each resample is reused for every condition in the run, so the delta intervals are paired. Resamples where a metric is undefined are ignored for that metric. Resampling is seeded per model/seed, so the results are reproducible and do not depend on which other models are in the file.
- **Seeds.** The summary file gives the mean and sample SD across training seeds. The SD is empty when there is only one seed. Bootstrap intervals are per run and are not pooled across seeds.

Input checks: the exact agreed header; labels 0/1; finite scores in [0, 1]; `clean`/`none` for the baseline and `low`/`medium`/`high` for degradations; no duplicates; one label per breast or image; and, within each model/seed, a clean baseline plus identical breast/image coverage for every condition.

## Output columns

`<level>_metrics_by_run.csv`: one row per model, seed and condition.

| Column | Meaning |
| --- | --- |
| `level`, `model`, `seed`, `degradation`, `severity` | Identify the run and condition |
| `threshold`, `threshold_source` | Threshold applied, and `fixed` or `youden_validation` |
| `n_patients`, `n_units`, `n_positive`, `n_negative` | Sample size (units = breasts or images) |
| `tp`, `fp`, `tn`, `fn` | Confusion counts at the threshold |
| `auc`, `sensitivity`, `specificity`, `f1` | Point estimates |
| `<metric>_ci_low`, `<metric>_ci_high` | Bootstrap interval |
| `delta_<metric>`, `delta_<metric>_ci_low`, `delta_<metric>_ci_high` | Change from clean, with paired interval |
| `n_bootstrap`, `confidence` | Bootstrap settings used |

`<level>_metrics_summary.csv`: one row per model and condition, with `n_seeds`, `threshold_source`, and `<metric>_mean`, `<metric>_sd`, `delta_<metric>_mean` and `delta_<metric>_sd` for each metric.

## Checks and limits

```bash
python -m unittest discover -s tests -v
```

Requires Python 3.10+, NumPy and pandas. The tests use synthetic records only. With the current test split (about 49 patients) the intervals will be wide, so report them alongside point estimates. The `--validation` CSV must come from the validation partition, never the test partition. The code cannot check this.
