# Project Decisions

This log distinguishes approved study rules from choices currently implemented in the baseline code.

## Decision 1: Dataset
- Status: CDD-CESM low-energy baseline prepared; final study dataset not yet confirmed.
- The repository contains a CDD-CESM image manifest and patient-level splits. CBIS-DDSM was the original proposal.
- Pending: Confirm dataset selection, provenance, access, and permitted use.

## Decision 2: Prediction task
- Status: Clean-baseline pipeline implemented.
- Binary malignancy classification uses image-level training and breast-level evaluation. Breast scores are the maximum of available CC and MLO scores, including single-view breasts.
- The study-wide task definition remains subject to group confirmation.

## Decision 3: Classes and labels
- Status: Binary-label handling implemented; verified original-class annotations needed for real-data inference.
- Normal and benign map to non-malignant (0); malignant maps to positive (1). Preserve original `normal`, `benign`, and `malignant` annotations in analysis metadata rather than reconstructing them from binary labels.
- Breast-level ground truth takes the more severe view diagnosis. Source image labels remain unchanged, including Patient 16's reviewed CC/MLO disagreement.

## Decision 4: Models and baseline training

- Status: Initial training protocol approved by Sammie on November 7, 2026. Implementation pending.
- Architectures: MobileNetV3-Small and ResNet-18, each with a newly initialized two-class classification head.
- Initialization: ImageNet-pretrained feature weights for both architectures. Record the exact torchvision weight identifiers and software versions. Proposed explicit identifiers: `MobileNet_V3_Small_Weights.IMAGENET1K_V1` and `ResNet18_Weights.IMAGENET1K_V1`.
- Training seeds: 42, 43, and 44. Train MobileNetV3-Small seed 42 first and complete its smoke test before training the remaining checkpoints.
- Input: 224 x 224 RGB images, preserving aspect ratio through padding.
- Normalization: ImageNet channel means (0.485, 0.456, 0.406) and standard deviations (0.229, 0.224, 0.225), consistently applied during training and inference.
- Augmentation: Horizontal flips and small rotations on training images only. No random augmentation on validation or test images.
- Initial augmentation parameters: horizontal-flip probability 0.5 and rotation range -10 to +10 degrees. These numerical values are implementation choices within the approved augmentation types.
- Optimizer: AdamW, learning rate 0.0001, weight decay 0.0001.
- Batch size: 16.
- Maximum epochs: 30.
- Loss: Class-weighted cross-entropy, with weights derived from training-split class counts only.
- Validation: Use the approved evaluation-image selection and aggregate image malignancy probabilities by maximum score across available views for each breast.
- Checkpoint selection: Highest breast-level validation ROC AUC. Break ties using lower validation loss, then earlier epoch.
- Early stopping: Patience of five epochs without improvement in breast-level validation AUC.
- Artifacts: Save the best checkpoint, training history, final configuration, and validation predictions for each seed outside the public repository.
- Pre-training requirement: Implement and test the approved protocol, then complete the first MobileNetV3-Small seed-42 training run.
- Experiment sequence: Perform the two-run clean-inference smoke test before seeking approval for the remaining five checkpoints.

## Decision 5: Image-quality variations
- Status: Proposed.
- Candidates: Noise, blur, reduced contrast, and lower resolution.
- Pending: Agree on transformations, parameter ranges, and severity levels before running degradation experiments.

## Decision 6: Data split
- Status: Patient-disjoint split implemented for the CDD-CESM baseline.
- The manifest assigns 228 patients to training, 49 to validation, and 49 to test.
- The evaluation-image selection in Decision 9 does not alter source split assignments.

## Decision 7: Evaluation metrics
- Status: Evaluation code implemented; final reporting protocol pending.
- Breast-level metrics: ROC AUC, sensitivity, specificity, and F1-score. The evaluator supports patient-level bootstrap confidence intervals and summaries across seeds.
- Youden's J thresholds can be selected from clean validation predictions per model/seed, then applied unchanged to the corresponding test conditions.

## Decision 8: Data and code sharing
- Status: Dataset usage and redistribution terms not yet verified.
- The repository contains code and metadata, not the mammogram files or trained checkpoints. Keep images, checkpoints, and restricted data-derived outputs out of the public repository unless sharing is permitted.

## Decision 9: Repeated MLO acquisitions
- Status: Approved by the project lead on November 2, 2026; reaffirmed in the November 5 review of PR #7.
- The repeated MLO acquisitions are treated as likely re-takes, not confirmed duplicates. Their acquisition history is unverified.
- Use the unsuffixed MLO image for evaluation for Patients 99 and 281. This selection was fixed before evaluation and does not depend on model scores.
- Patient 99 (training): Retain both `P99_L_DM_MLO` and `P99_L_DM_MLO_2` for training. Select `P99_L_DM_MLO` when an evaluation view is needed.
- Patient 281 (validation): Use `P281_L_DM_MLO` for validation loss and prediction exports. Keep `P281_L_DM_MLO2` and its annotation in the source manifest but exclude that image from those calculations and exports.
- Apply the same selection across models, seeds, and image-quality conditions. No test images are excluded. Single-view and Patient 16 label-conflict handling remain unchanged.
- References: [Acquisition policy](docs/ACQUISITION_POLICY.md) and [PR #7](https://github.com/SamsondareHUB/mammography-robustness-study/pull/7).
