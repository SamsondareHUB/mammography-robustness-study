# Experiment Log

Add one entry for every experiment. Record unsuccessful runs too.

## Experiment 001
- Date:
- Responsible member:
- Research question:
- Dataset and split:
- Model:
- Training settings:
- Image-quality condition:
- Evaluation metrics:
- Results:
- Notes, issues, or next steps:

## Pre-experiment annotation review — 2026-10-30

- CDD-CESM low-energy split manifest: 1,003 image records with patient-level train, validation, and test assignments.
- Patient 16, right breast: CC is annotated normal; MLO is malignant. Preserve the image labels and use malignant breast-level ground truth under the agreed severity rule.
- Patients 99 (train) and 281 (validation), left breast: each has an additional MLO image. Sallew compared the images and reported different dimensions, file contents, and breast positioning. They are distinct images and appear to be separate exposures; the acquisition history remains unverified.
- Both additional MLO images are annotated normal. Preserve the source records. A documented inclusion rule or an acquisition-specific image key is required for evaluation.
- Exporter tests use synthetic data; no model performance is reported. Experiment 001 has not been run.
