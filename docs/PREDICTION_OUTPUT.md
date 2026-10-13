# Prediction exports

`mammography_robustness.predictions` checks paired image predictions, produces breast-level scores and writes the two CSV formats defined in the evaluation protocol. It does not load mammograms, train models, assign pathology labels, compute metrics or measure inference cost.

## Inputs and identifiers

`ImagePrediction` requires `case_id`, `patient_id`, `breast_side`, `image_id`, `view`, `label`, `score`, `model`, `degradation`, `severity`, and `seed`. The internal `image_id` identifies a source image across model runs and degradation conditions; it is not exported to the agreed CSV.

- `patient_id` is used to prevent patients from crossing training, validation and test partitions.
- `breast_side` is normalized to `L` or `R`. Together with `patient_id`, it identifies the breast-level evaluation unit.
- `case_id` preserves the source case identifier and may cover both breasts of one patient; it is not used for patient-disjoint splitting.
- `view` is normalized to `CC` or `MLO`. Each evaluated breast must have exactly one of each view. Missing/duplicate views are rejected, not filled in.
- `label` is 1 for malignant and 0 for benign/normal, provided the source annotations support that mapping. Scores must be finite in [0, 1]. A model output score is **not** necessarily calibrated cancer risk.

Dataset labels, laterality and view mapping must be verified from the source metadata by the dataset-preparation workflow before exporting predictions.

## Output schemas

Image-level predictions:

```csv
case_id,patient_id,breast_side,view,label,score,model,degradation,severity,seed
```

Breast-level predictions:

```csv
patient_id,breast_side,label,score,model,degradation,severity,seed
```

Each image receives a clean baseline (`degradation=clean`, `severity=none`) for each model and training seed. Degraded conditions use `low`, `medium` or `high` severity. When `expected_conditions` is supplied, export requires the exact condition grid plus the clean baseline for every breast/model/seed.

The breast score is always the **maximum** of that breast's CC and MLO image scores. Input pairs must be identical across models, seeds and degradation settings. The image-to-breast export checks these requirements with stable internal `image_id` values and sorts both exports deterministically. `write_breast_csv` can also serialize previously computed breast scores, but cannot independently establish how those scores were aggregated. Use `export_prediction_tables` when both outputs must be derived from the same image predictions.

## Example

```python
from pathlib import Path
from mammography_robustness.predictions import ImagePrediction, export_prediction_tables

rows = [
    ImagePrediction("c1", "p1", "L", "im-cc", "CC", 1, 0.3, "small", "clean", "none", 1),
    ImagePrediction("c1", "p1", "L", "im-mlo", "MLO", 1, 0.8, "small", "clean", "none", 1),
    ImagePrediction("c1", "p1", "L", "im-cc", "CC", 1, 0.2, "small", "blur", "low", 1),
    ImagePrediction("c1", "p1", "L", "im-mlo", "MLO", 1, 0.6, "small", "blur", "low", 1),
]

Path("outputs").mkdir(exist_ok=True)
export_prediction_tables(rows, "outputs/image_predictions.csv", "outputs/breast_predictions.csv",
                         expected_conditions=[("blur", "low")])
```

The example uses synthetic identifiers and scores. The CSVs contain four image rows and two breast rows; the breast scores are 0.8 (clean) and 0.6 (blur/low). The export writes temporary files before replacing either destination to avoid leaving a partial output pair when validation or staging fails. Two separate file replacements are not a transactional operation.

## Split counts

`SplitImage` holds `patient_id`, `breast_side`, `image_id`, and `split` (`train`, `validation`, `test`). Use `write_split_summary_json` on the **full dataset manifest**, not just test predictions, to obtain distinct patient, breast, and image counts. It rejects any patient appearing in multiple partitions. The module does not construct a split, verify the provenance of supplied identifiers, or cross-check the split manifest against exported predictions. The dataset pipeline must ensure predictions come from the intended held-out partition.

## Checks and limits

```bash
python -m unittest discover -s tests -v
```

Requires Python 3.10 or later. Tests use synthetic records only; they do not establish correctness of dataset annotations, predictive performance, calibration or clinical utility.

The export currently assumes paired CC/MLO views; it will fail for breasts without both rather than silently changing the evaluation sample. Missing-view handling and inclusion criteria need to be specified before running the study. Transformation intensities, model choice, data access and train/validation/test partitions must also be recorded separately.

Do not commit patient-identifiable or restricted data. Local exports belong in an ignored directory such as `outputs/`; verify dataset licensing and sharing permissions before distributing files.
