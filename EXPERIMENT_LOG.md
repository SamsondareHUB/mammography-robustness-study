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

## Pre-experiment implementation update (2026-11-05)

- [PR #7](https://github.com/SamsondareHUB/mammography-robustness-study/pull/7) merged the approved repeated-MLO policy. The likely re-takes are handled using a fixed, score-independent image selection rule. Patient 99 retains both training acquisitions; Patient 281's extra MLO remains in the manifest but is excluded from validation loss and prediction exports. Test images are unaffected.
- [PR #8](https://github.com/SamsondareHUB/mammography-robustness-study/pull/8) merged clean validation/test checkpoint inference and prediction exports. Tests used synthetic images and checkpoints.
- Experiment 001 has not run on CDD-CESM images. It requires verified original-class annotations, image access, agreed training settings, and trained checkpoints. No measured performance results are available.
