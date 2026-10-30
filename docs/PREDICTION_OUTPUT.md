# Prediction exports

`mammography_robustness.predictions` validates image-level predictions, aggregates breast-level scores, and writes CSVs consumed by the current evaluator. It does not run model inference or calculate evaluation metrics.

## Input records

Each `ImagePrediction` specifies `case_id`, `patient_id`, `breast_side`, `image_id`, `view`, `label`, `score`, `model`, `degradation`, `severity`, and `seed`. `original_class` is optional and can hold `normal`, `benign`, or `malignant`.

- `image_id` identifies a single source acquisition across model runs, seeds, and degradation conditions. It is kept internally and in the supplementary image metadata export.
- `case_id` is the dataset identifier. `patient_id` and `breast_side` identify a breast for aggregation and patient-level analysis.
- `label` is the original **image-level** binary annotation: malignant = 1, normal or benign = 0. The exporter verifies that `original_class`, when supplied, agrees with this mapping.
- `view` must be `CC` or `MLO`. Either a single available view or a CC/MLO pair is valid.
- `score` is the model's finite malignancy score in [0, 1], not necessarily a calibrated clinical probability.

Input validation requires stable image identifiers, unchanged source annotations, a clean baseline per model/seed, and matching image coverage across evaluation conditions. It rejects different source-image selections across models or seeds.

## Breast aggregation

For each breast, model, seed, and condition:

- The breast-level binary **ground truth** is malignant if **any** available image has `label=1`. An image's own label is never rewritten. In particular, different CC and MLO labels can coexist.
- The breast-level score is the **maximum** of the available image scores, or the sole score for a single-view breast.
- `view_status` is `single` or `paired` in the supplementary breast metadata.
- `view_label_conflict` is 1 when the included CC and MLO images have different binary diagnoses, and 0 otherwise. This derived field is **not** interchangeable with the evaluator's separately defined `label_flag` until the team confirms its meaning and source.

## Evaluation CSV schemas

The prediction CSVs retain the agreed column order and remain compatible with the current evaluator.

Image-level predictions:

```csv
case_id,patient_id,breast_side,view,label,score,model,degradation,severity,seed
```

Breast-level predictions:

```csv
patient_id,breast_side,label,score,model,degradation,severity,seed
```

The image-level CSV preserves each view's original binary label. The breast-level CSV uses the aggregated ground truth. Both use `degradation=clean` and `severity=none` for baseline records; degraded records use `low`, `medium`, or `high` severity. When `expected_conditions` is supplied, those conditions plus clean must appear for each breast/model/seed.

### Optional supplementary metadata

`write_image_metadata_csv()` writes the stable `image_id` and `original_class` alongside each source prediction; it requires `original_class` on all records. `write_breast_metadata_csv()` writes `view_status` and `view_label_conflict`. These are separate metadata outputs; they are **not** accepted as replacement prediction inputs by the current evaluator.

The normal-versus-benign false-positive breakdown and flagged-breast sensitivity analysis require compatible evaluation fields. Verify their definitions and join keys against the revised evaluator before changing the primary CSV schemas.

## Repeated acquisitions

Patients 99 and 281 each have two distinct left-breast MLO acquisitions in the annotation manifest. Keep both in the source records. Do not equate them with a duplicated file, remove one automatically, or choose one using predicted scores.

The current image-level evaluation key is `(patient_id, breast_side, view, model, seed, degradation, severity)`, which cannot distinguish two acquisitions of the same view. The exporter therefore **rejects repeated views**. Before including either or both acquisitions in the primary analysis, the team must document a reproducible inclusion rule or extend the evaluation key. The same source-image set must be used across all compared conditions.

## Example

```python
from pathlib import Path
from mammography_robustness.predictions import ImagePrediction, export_prediction_tables

rows = [
    ImagePrediction("case-1", "p1", "R", "p1-r-cc", "CC", 0, 0.2,
                    "baseline", "clean", "none", 1, original_class="normal"),
    ImagePrediction("case-1", "p1", "R", "p1-r-mlo", "MLO", 1, 0.7,
                    "baseline", "clean", "none", 1, original_class="malignant"),
]

Path("outputs").mkdir(exist_ok=True)
export_prediction_tables(rows, "outputs/image_predictions.csv", "outputs/breast_predictions.csv")
```

For this synthetic example, the image labels remain 0 and 1, the breast label is 1, and the breast score is 0.7.

Both output tables are validated before writing. The exporter stages their complete contents before replacing the output paths, although replacing two separate files is not a filesystem transaction.

## Split summary and scope

`SplitImage` holds `patient_id`, `breast_side`, `image_id`, and `split`. Use `write_split_summary_json()` with the **full manifest** to count distinct patients, breasts, and images and detect patients assigned to multiple partitions. It does not independently establish file provenance or match test predictions against the original split manifest.

Run all project tests with:

```bash
python -m unittest discover -s tests -v
```

Tests use synthetic predictions and do not establish model performance. Before exporting real validation or test predictions, verify annotation provenance, acquisition inclusion criteria, and compatibility with the evaluation code. Do not commit mammograms or other restricted data.
