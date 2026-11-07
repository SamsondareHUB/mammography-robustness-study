# Clean-baseline prediction exports

This command loads trained checkpoints and exports clean validation and test
predictions for the breast-level evaluator. It does not train models, select
thresholds, or perform degradation experiments.

## Inputs

- `data/cdd_cesm_low_energy_splits.csv` and `data/splits.json` from the approved patient split.
- Local CDD-CESM low-energy images whose filename stems match the manifest `case_id` values.
- A verified image-level three-class annotation CSV, supplied separately using `--annotations`. The CSV must contain `patient_id`, `case_id`, `breast_side`, `view`, `label`, `split`, and `original_class`, with a matching row for every manifest image.
- One checkpoint per model and seed, produced using the same manifest and split files.

The annotation mapping is checked against the complete manifest. Missing cases,
wrong patient or split assignments, and disagreement between original class
and binary label cause an error. Do not derive normal versus benign from a
binary negative label.

## Evaluation image selection

The fixed acquisition rule is applied before inference. The additional Patient
281 MLO is excluded from validation predictions; the unsuffixed MLO is kept.
Patient 99's additional MLO remains available during training. Neither
Patient 99 nor Patient 281 is in the test set. Single-view breasts and
Patient 16's view-specific labels are preserved.

All model/seed checkpoints use the same selected validation and test images.

## Preprocessing and checkpoint compatibility

The approved baseline uses ImageNet-pretrained MobileNetV3-Small and ResNet-18
with newly initialized two-class output layers. The exact pretrained-weight
identifiers are `MobileNet_V3_Small_Weights.IMAGENET1K_V1` and
`ResNet18_Weights.IMAGENET1K_V1`. During inference, the model architecture
is initialized without downloading pretrained weights and the trained
checkpoint state is loaded instead.

Images are converted to RGB and aspect-preserving padded to the checkpoint's
recorded image size using bilinear interpolation and black padding. For the
approved baseline, tensors are normalized with ImageNet mean
`(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`.
No random augmentation is applied to validation or test images. The code
reads the normalization setting from each checkpoint, rejects inconsistent
pretraining metadata, and preserves compatibility with older checkpoints
that recorded no normalization.

Each checkpoint is checked for model identity, seed, architecture state,
preprocessing size, and SHA-256 hashes of the original manifest and split
JSON. Checkpoint hashes must match the exact files supplied for inference.
The output score is the softmax probability of class 1 (malignant), not a
calibrated clinical risk estimate.

## Run

Install the training dependencies on the chosen machine. From the repository
root, run the following Bash example with permitted private data:

```bash
python -m mammography_robustness.clean_inference \
  --annotations "/private/path/verified_original_classes.csv" \
  --images "/private/path/low_energy_images" \
  --checkpoint "/private/path/mobilenet_v3_small_seed42_best.pt" \
  --output-dir "/private/path/clean_smoke_first" --device cpu
```

The equivalent Windows PowerShell command uses your project virtual environment:

```powershell
& "$HOME\.venvs\mammography-robustness-study\Scripts\python.exe" -m mammography_robustness.clean_inference `
  --annotations "C:\private-data\verified_original_classes.csv" `
  --images "C:\private-data\low_energy_images" `
  --checkpoint "C:\private-data\mobilenet_v3_small_seed42_best.pt" `
  --output-dir "C:\private-data\clean_smoke_first" --device cpu
```

Use a new output directory for the repeat run. The complete
[CLEAN_BASELINE_SMOKE_TEST.md](CLEAN_BASELINE_SMOKE_TEST.md) verification
procedure currently uses Bash syntax and must be run in Bash unless
translated to PowerShell. It checks the counts and hashes of both runs.

For the main experiment, supply all six independently trained checkpoints:
MobileNetV3-Small and ResNet-18, each with three distinct training seeds.
Repeat `--checkpoint` for each path. Results are kept separate by `model`
and `seed` columns. Duplicate model/seed checkpoint pairs are rejected.
The project lead must review the seed-42 smoke test before the other five
checkpoints are trained.

## Outputs

The command creates eight CSVs:

- `val_image_predictions.csv` and `val_breast_predictions.csv`
- `val_image_metadata.csv` and `val_breast_metadata.csv`
- `test_image_predictions.csv` and `test_breast_predictions.csv`
- `test_image_metadata.csv` and `test_breast_metadata.csv`

The prediction files use the existing fixed CSV schemas. Metadata includes
the original three-class diagnoses, image identifiers, view status, and
view-label disagreement flag. Breast scores use the maximum malignancy
probability across the available views. Destination CSVs are not overwritten.

Once the approved clean-baseline experiment is complete, the evaluator can
select Youden's J threshold from clean validation predictions and apply it
to the clean test set:

```bash
python -m mammography_robustness.evaluation \
  /private/path/clean_outputs/test_breast_predictions.csv \
  --metadata /private/path/clean_outputs/test_breast_metadata.csv \
  --validation /private/path/clean_outputs/val_breast_predictions.csv \
  --threshold youden --clean-only \
  --out-dir /private/path/clean_metrics
```

Review model/seed coverage, confidence intervals, thresholds, and
validation-file provenance. Only report measured results from actual trained
checkpoints. Do not commit raw mammograms, model checkpoints, restricted
annotations, or restricted data-derived outputs to the public repository.

## Limits

This script does not establish model quality, calibration, or clinical
utility. Validate preprocessing and checkpoint compatibility using permitted
data before reporting results from the six-run experiment.
