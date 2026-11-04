# Training setup

This module prepares image-level binary classification on CDD-CESM low-energy images. The two candidate architectures are MobileNetV3-Small and ResNet-18. Model selection, training hyperparameters, and evaluation inclusion criteria remain subject to the team's experimental decisions.

## Environment

Python 3.10 or newer is required. From the repository root:

```bash
python -m venv .venv
```

Activate the environment using the appropriate command for your operating system, then install the dependencies:

```bash
python -m pip install -r requirements-training.txt
```

Install a PyTorch build compatible with the intended CPU or GPU platform when necessary.

## Validate the dataset metadata

```bash
python -m mammography_robustness.training
```

By default the command checks the image manifest, the JSON patient assignments, duplicate CC/MLO entries, and per-view label differences. It reports split counts, incomplete view pairs, and label conflicts without changing source image labels. **It does not open mammogram files or train a model.** Validation requires only the Python standard library.

The expected inputs are `data/cdd_cesm_low_energy_splits.csv` and `data/splits.json`, with patient IDs assigned consistently in both files. Both inputs must agree before model training. Missing view pairs must be addressed in the evaluation protocol; they are not silently excluded by the training loader.

## Training configuration

`configs/training_baseline.json` contains example settings. Model choice and hyperparameters are configurable. Both architectures start without pretrained weights; no model weights are downloaded. A transfer-learning strategy can be incorporated after the team agrees on it.

The manifest's `case_id` is used as the filename stem for each image. Images may be located in subdirectories and must use `.jpg`, `.jpeg`, or `.png` extensions. The image directory is supplied at runtime and is not committed to the repository.

Once the data checks, inclusion criteria, and experimental settings are approved, training can be enabled explicitly:

```bash
python -m mammography_robustness.training \
  --config configs/training_baseline.json \
  --images "/path/to/Low energy images of CDD-CESM" \
  --train
```

The optional `--allow-incomplete-views` flag permits image-level training when some breasts lack CC or MLO images. The separate `--allow-conflicting-labels` flag permits training with different labels for the two views of a breast after those source annotations have been reviewed. Neither flag changes image-level labels, resolves the breast-level ground truth, or bypasses checks for inconsistent patient splits and duplicate views. Both flags require approval of the applicable inclusion and annotation rules before use.

Input images are converted to RGB, resized with their aspect ratio preserved, and padded to a square without cropping. The baseline uses a cross-entropy loss for image-level training, reports training and validation loss, and saves the lowest-validation-loss checkpoint to `artifacts/checkpoints/`. Checkpoints include the random seed, configuration, and SHA-256 digests of the input manifest and split file. The test partition is not used during training or checkpoint selection.

The existing prediction exporter handles breast-level output generation; it is not invoked by this training script. Its current release requires paired views and consistent breast labels, so the planned single-view and label-conflict evaluation changes must be implemented separately after the annotations and split files are reconciled. This setup does not generate AUC, sensitivity, specificity, F1-score, or clinical-performance results. Dataset-specific preprocessing and final model choices must be documented before running experiments.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests use synthetic manifests and images. They do not train a classifier on CDD-CESM data or establish predictive performance. Do not commit patient images, local checkpoints, or restricted datasets.
