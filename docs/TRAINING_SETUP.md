# CDD-CESM clean-baseline training protocol

The initial baseline uses two ImageNet-pretrained architectures with newly initialized binary classification heads: MobileNetV3-Small and ResNet-18. The approved seeds are 42, 43, and 44. **Train MobileNetV3-Small seed 42 only first.** The project lead must review its two-run inference smoke test before the remaining five runs.

## Data and environment

Use the approved, committed `data/cdd_cesm_low_energy_splits.csv` and `data/splits.json`, with access to the permitted CDD-CESM low-energy images. Images and model checkpoints must remain in approved private storage; do not commit restricted images, annotations, checkpoints, or local paths. Dataset and patient-split preflight requires no images or PyTorch:

```bash
python -m mammography_robustness.training
```

Install training dependencies using `requirements-training.txt`. Train on an appropriate CPU/GPU system; a machine without CUDA is suitable for development and tests, but the real 30-epoch training run may be impractical on a laptop CPU.

## Model and preprocessing

The configuration in `configs/training_baseline.json` specifies the first seed-42 model. Exact pretrained weights are `MobileNet_V3_Small_Weights.IMAGENET1K_V1` and `ResNet18_Weights.IMAGENET1K_V1` from torchvision, recorded along with the Python, torch, and torchvision versions. The final two-class classification layer is newly initialized. Training and inference instantiate identical architectures; inference loads the saved state without downloading pretrained weights.

Images are converted to RGB, aspect-preserving padded to 224×224 with bilinear interpolation and black padding, converted to tensors, and normalized using ImageNet mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`. Only training images receive random horizontal flip at probability 0.5 and rotation sampled within ±10 degrees (bilinear interpolation; black fill). Validation and test images have no random augmentation. The flip probability and degree range are explicit implementation choices under Sammie's approved augmentation types; note them in `DECISIONS.md` before training.

## Training and checkpoint selection

- AdamW optimizer, learning rate 0.0001, weight decay 0.0001; batch size 16, maximum 30 epochs.
- Image-level class-weighted cross-entropy; class weights are `N / (2 * count)` per class, calculated using training-split images only.
- Validation uses the fixed evaluation-image selection, excluding the additional Patient 281 MLO image without altering source records. AUC uses breast-level labels (maximum available-view malignancy label) and breast-level scores (maximum available-view malignancy probability).
- Select the epoch with highest **breast-level validation ROC AUC**. On exact ties, choose lower unweighted image-level validation cross-entropy, then the earlier epoch.
- Stop after 5 consecutive epochs without a **strict AUC increase**. A lower loss at tied AUC may update the saved checkpoint but does not reset the AUC-based patience counter.
- Each private run records its configuration, dataset hashes, pretrained-weight identity, library versions, chosen checkpoint, history, and selected validation image/breast predictions.

## Commands

Before training, finalize the numerical augmentation choices in `DECISIONS.md` and verify the code. The suggested values (flip probability 0.5 and rotations up to 10 degrees) are implementation choices, not values explicitly selected by Sammie. Train **only seed 42** first on a permitted machine with access to the real images:

```bash
python -m mammography_robustness.training \
  --config configs/training_baseline.json \
  --images "/private/path/Low energy images of CDD-CESM" \
  --train --allow-incomplete-views --allow-conflicting-labels
```

The following is the equivalent command for Windows PowerShell. Replace the example image path with the actual permitted dataset location. A Windows laptop without CUDA can run development and smoke tests, but real training is better suited to an approved cloud GPU.

```powershell
& "$HOME\.venvs\mammography-robustness-study\Scripts\python.exe" -m mammography_robustness.training `
  --config configs/training_baseline.json `
  --images "C:\private-data\CDD-CESM\Low energy images" `
  --train --allow-incomplete-views --allow-conflicting-labels
```

The two flags reflect already reviewed single-view and Patient 16 source-label policies. They do not alter the original labels. Do not use them to waive newly discovered data-integrity problems.

Outputs are saved to `artifacts/checkpoints/` by default, which must be ignored by Git. Move them to approved private storage as needed. The checkpoint name is `mobilenet_v3_small_seed42_best.pt`; accompanying files include `_history.json`, `_config.json`, `mobilenet_v3_small_seed42_val_image_predictions.csv`, and `mobilenet_v3_small_seed42_val_breast_predictions.csv`. Do not overwrite an existing training run.

Use [CLEAN_BASELINE_SMOKE_TEST.md](CLEAN_BASELINE_SMOKE_TEST.md) for its complete dataset-level verification instructions. That guide uses Bash syntax; [CLEAN_BASELINE_INFERENCE.md](CLEAN_BASELINE_INFERENCE.md) also supplies a PowerShell inference command. Run checkpoint inference **twice** into distinct output folders and verify all eight CSV files, counts, hashes, original labels, and breast-level aggregation. The first smoke test is a reproducibility/coverage check, **not** a performance result. Do not train the other five checkpoints before the project lead reviews its summary.
