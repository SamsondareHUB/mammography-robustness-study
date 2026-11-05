# Clean-baseline prediction exports

This command loads trained checkpoints and exports clean validation and test
predictions for the breast-level evaluator. It does not train models, select
thresholds, or perform degradation experiments.

## Inputs

- `data/cdd_cesm_low_energy_splits.csv` and `data/splits.json` from the approved patient split.
- Local CDD-CESM low-energy images whose filename stems match the manifest `case_id` values.
- A verified image-level three-class annotation CSV, provided separately with `--annotations`. The file must include `patient_id`, `case_id`, `breast_side`, `view`, `label`, `split`, and `original_class`, with one matching row for every manifest image.
- One checkpoint per model and seed, produced by the training script using the exact same manifest and patient split files.

The annotation mapping is checked against the complete manifest. Missing cases,
wrong patient or split assignments, and disagreement between the original class
and binary label cause an error. Do not derive normal versus benign from a binary
negative label.

## Evaluation image selection

The fixed acquisition rule is applied before inference. The additional Patient
281 MLO is excluded from validation predictions; the unsuffixed MLO is kept.
Neither Patient 99 nor Patient 281 belongs to the test split, so test-image
coverage is unchanged. Patient 99's additional MLO remains available during
training. Single-view breasts and Patient 16's view-specific labels are preserved.

The same validation and test images are used for all model/seed checkpoints.

## Run

Install the dependencies used by the training pipeline on the chosen machine
and confirm that your checkpoints can be loaded with its PyTorch version.

Example with one checkpoint, from the repository root:

```bash
python -m mammography_robustness.clean_inference \
  --annotations /local/path/cdd_cesm_low_energy_splits_with_original_class.csv \
  --images "/local/path/Low energy images of CDD-CESM" \
  --checkpoint /local/path/mobilenet_v3_small_seed42_best.pt \
  --output-dir /local/path/clean_outputs
```

For the main experiment, supply all **six** independently trained checkpoints:
MobileNetV3-Small and ResNet-18, each with three distinct training seeds.
Repeat `--checkpoint` for each path. Outputs are combined across checkpoints
but remain separated by the `model` and `seed` CSV columns. The program rejects
duplicate model/seed pairs. Keep each model/seed's training checkpoint in a
separate output directory so training cannot overwrite an earlier run.

Each checkpoint is checked for model identity, seed, architecture state,
preprocessing size, and SHA-256 hashes of the original manifest and split JSON.
Checkpoint hashes must match the exact files supplied to inference. Images are
converted to RGB and aspect-preserving padded and resized exactly as in the
training code. The output score is the softmax probability of class 1
(malignant), not a calibrated clinical probability.

## Outputs

The command creates eight CSV files in the specified output directory:

- `val_image_predictions.csv` and `val_breast_predictions.csv`
- `val_image_metadata.csv` and `val_breast_metadata.csv`
- `test_image_predictions.csv` and `test_breast_predictions.csv`
- `test_image_metadata.csv` and `test_breast_metadata.csv`

The prediction files use the existing fixed CSV schemas. The metadata files
contain the original three-class diagnoses, image identifiers, view status, and
view-label disagreement flag. Breast scores use maximum probability across the
available views. Files are not overwritten if the destination names already
exist; choose a fresh output directory for a new experiment.

For the clean-image evaluation, the evaluator uses clean validation predictions
to select Youden's J threshold for each model/seed and applies that threshold to
clean test predictions:

```bash
python -m mammography_robustness.evaluation \
  /local/path/clean_outputs/test_breast_predictions.csv \
  --metadata /local/path/clean_outputs/test_breast_metadata.csv \
  --validation /local/path/clean_outputs/val_breast_predictions.csv \
  --threshold youden --clean-only \
  --out-dir /local/path/clean_metrics
```

Review the output for model/seed coverage, patient-level confidence intervals,
thresholds, and validation-file provenance. Only report measured results from
actual trained checkpoints. Do not upload raw mammograms, sensitive images,
model checkpoints, or restricted data-derived outputs to the public repository.

## Limits

This script does not establish model quality, calibration, or clinical utility.
The training architectures currently use randomly initialized weights unless
the team adds a documented pretraining procedure. Validate preprocessing and
checkpoint compatibility on a small permitted test set before conducting the
full six-run experiment.
