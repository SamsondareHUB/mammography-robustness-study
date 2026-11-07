"""Reproducible CDD-CESM baseline training with breast-level validation AUC."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from .preprocessing import IMAGENET_MEAN, IMAGENET_STD, make_transform
from .training_protocol import breast_auc, inverse_frequency_weights, should_replace_best
from .training_data import (
    index_images, read_manifest, read_patient_splits,
    select_evaluation_images, validate_records,
)


@dataclass(frozen=True)
class TrainingConfig:
    model: str
    image_size: int
    batch_size: int
    epochs: int
    learning_rate: float
    num_workers: int
    seed: int
    output_dir: str
    pretrained: bool = False
    normalization: str = "none"
    optimizer: str = "adamw"
    weight_decay: float = 0.0001
    flip_probability: float = 0.5
    rotation_degrees: float = 10.0
    early_stopping_patience: int = 5
    class_weighted: bool = True


MODEL_NAMES = ("mobilenet_v3_small", "resnet18")
WEIGHT_IDS = {
    "mobilenet_v3_small": "MobileNet_V3_Small_Weights.IMAGENET1K_V1",
    "resnet18": "ResNet18_Weights.IMAGENET1K_V1",
}
REQUIRED_CONFIG_KEYS = (
    "model", "image_size", "batch_size", "epochs", "learning_rate",
    "num_workers", "seed", "output_dir",
)


def load_config(path: str | Path) -> TrainingConfig:
    """Read a versioned training config; preserve old preflight-config tests."""
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    allowed = set(TrainingConfig.__dataclass_fields__)
    if not isinstance(payload, dict) or not set(REQUIRED_CONFIG_KEYS).issubset(payload) or (set(payload) - allowed):
        raise ValueError("Training configuration has missing or unexpected fields")
    cfg = TrainingConfig(**payload)
    if cfg.model not in MODEL_NAMES:
        raise ValueError(f"model must be one of {MODEL_NAMES}")
    for name in ("image_size", "batch_size", "epochs", "seed", "num_workers", "early_stopping_patience"):
        value = getattr(cfg, name)
        minimum = 64 if name == "image_size" else 0 if name in ("seed", "num_workers") else 1
        if type(value) is not int or value < minimum:
            raise ValueError(f"Invalid {name}")
    for name in ("learning_rate", "weight_decay", "flip_probability", "rotation_degrees"):
        value = getattr(cfg, name)
        if type(value) not in (float, int) or not float("-inf") < value < float("inf"):
            raise ValueError(f"Invalid {name}")
    if not 0 < cfg.learning_rate < 1 or not 0 <= cfg.weight_decay < 1:
        raise ValueError("Invalid optimizer hyperparameter")
    if not 0 <= cfg.flip_probability <= 1 or not 0 <= cfg.rotation_degrees <= 180:
        raise ValueError("Invalid augmentation parameter")
    if cfg.normalization not in ("none", "imagenet") or cfg.optimizer != "adamw":
        raise ValueError("Unsupported normalization or optimizer")
    if type(cfg.pretrained) is not bool or type(cfg.class_weighted) is not bool:
        raise ValueError("pretrained and class_weighted must be booleans")
    if cfg.pretrained and cfg.normalization != "imagenet":
        raise ValueError("ImageNet-pretrained baseline requires ImageNet normalization")
    if not isinstance(cfg.output_dir, str) or not cfg.output_dir.strip():
        raise ValueError("output_dir cannot be empty")
    return cfg


def preflight(manifest_path: Path, splits_path: Path):
    records = read_manifest(manifest_path)
    summary = validate_records(records, read_patient_splits(splits_path))
    return records, summary


def create_model(name: str, *, pretrained: bool = False):
    """Load explicit ImageNet V1 weights only when opted in for training."""
    from torch import nn
    from torchvision.models import (
        MobileNet_V3_Small_Weights, ResNet18_Weights,
        mobilenet_v3_small, resnet18,
    )
    if name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif name == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
    else:
        raise ValueError(f"Unknown model: {name}")
    return model


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_prediction_csvs(output: Path, cfg: TrainingConfig, data):
    """Save predictions for the selected best epoch without publishing them."""
    image_rows, breast_rows = data
    for name, columns, rows in (
        (f"{cfg.model}_seed{cfg.seed}_val_image_predictions.csv", (
            "case_id", "patient_id", "breast_side", "view", "label", "score",
            "model", "degradation", "severity", "seed"), image_rows),
        (f"{cfg.model}_seed{cfg.seed}_val_breast_predictions.csv", (
            "patient_id", "breast_side", "label", "score", "model",
            "degradation", "severity", "seed"), breast_rows),
    ):
        path = output / name
        temp = path.with_suffix(".tmp")
        try:
            with temp.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)


def _validation_predictions(rows, probabilities, cfg):
    from collections import defaultdict
    images = []
    groups = defaultdict(list)
    for r, score in zip(rows, probabilities, strict=True):
        score = float(score)
        attrs = {"model": cfg.model, "degradation": "clean", "severity": "none", "seed": cfg.seed}
        images.append({
            "case_id": r.case_id, "patient_id": r.patient_id,
            "breast_side": r.breast_side, "view": r.view,
            "label": r.label, "score": score, **attrs,
        })
        groups[(r.patient_id, r.breast_side)].append((r.label, score))
    breasts = []
    for (patient, side), values in sorted(groups.items()):
        breasts.append({
            "patient_id": patient, "breast_side": side,
            "label": max(item[0] for item in values),
            "score": max(item[1] for item in values),
            "model": cfg.model, "degradation": "clean", "severity": "none", "seed": cfg.seed,
        })
    return images, breasts


def run_training(records, image_paths, cfg: TrainingConfig, manifest_path: Path, splits_path: Path):
    """Train one approved model/seed; never access test images for selection."""
    import torch
    import torchvision
    from PIL import Image
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_records = [r for r in records if r.split == "train"]
    val_records = select_evaluation_images([r for r in records if r.split == "val"])
    if not train_records or not val_records:
        raise ValueError("Training and validation splits must be nonempty")
    train_transform = make_transform(
        cfg.image_size, training=True, normalization=cfg.normalization,
        flip_probability=cfg.flip_probability, rotation_degrees=cfg.rotation_degrees,
    )
    val_transform = make_transform(cfg.image_size, training=False, normalization=cfg.normalization)

    class DatasetWithRecords(Dataset):
        def __init__(self, rows, transform):
            self.rows, self.transform = rows, transform

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            r = self.rows[idx]
            with Image.open(image_paths[r.case_id]) as image:
                x = self.transform(image)
            return x, r.label

    loader_gen = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(
        DatasetWithRecords(train_records, train_transform), batch_size=cfg.batch_size,
        shuffle=True, num_workers=cfg.num_workers, generator=loader_gen,
    )
    val_loader = DataLoader(
        DatasetWithRecords(val_records, val_transform), batch_size=cfg.batch_size,
        shuffle=False, num_workers=cfg.num_workers,
    )
    output = Path(cfg.output_dir)
    stem = f"{cfg.model}_seed{cfg.seed}"
    path = output / f"{stem}_best.pt"
    history_path = output / f"{stem}_history.json"
    config_path = output / f"{stem}_config.json"
    prediction_paths = (
        output / f"{stem}_val_image_predictions.csv",
        output / f"{stem}_val_breast_predictions.csv",
    )
    # Guard every run artifact before downloading weights or starting training.
    if any(p.exists() for p in (path, history_path, config_path, *prediction_paths)):
        raise FileExistsError(f"Training output already exists for {stem}; use a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(cfg.model, pretrained=cfg.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    loss_weights = inverse_frequency_weights(r.label for r in train_records)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(loss_weights, device=device)) if cfg.class_weighted else nn.CrossEntropyLoss()
    val_criterion = nn.CrossEntropyLoss()
    manifest_sha, splits_sha = _sha256(manifest_path), _sha256(splits_path)
    history = []
    best = None
    best_predictions = None
    stagnant_epochs = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_sum, train_count = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_sum += loss.item() * labels.size(0)
            train_count += labels.size(0)
        model.eval()
        val_sum, val_count, val_probabilities = 0.0, 0, []
        with torch.inference_mode():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                val_sum += val_criterion(logits, labels).item() * labels.size(0)
                val_count += labels.size(0)
                val_probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
        val_loss = val_sum / val_count
        val_auc = breast_auc(
            (r.patient_id for r in val_records), (r.breast_side for r in val_records),
            (r.label for r in val_records), val_probabilities,
        )
        history.append({
            "epoch": epoch, "train_loss": train_sum / train_count,
            "val_loss": val_loss, "val_breast_auc": val_auc,
        })
        print(f"Epoch {epoch}/{cfg.epochs}: train_loss={train_sum / train_count:.4f} "
              f"val_loss={val_loss:.4f} val_breast_auc={val_auc:.4f}")
        strict_auc_improvement = best is None or val_auc > best[0]
        if should_replace_best(val_auc, val_loss, epoch, best):
            best = (val_auc, val_loss, epoch)
            best_predictions = _validation_predictions(val_records, val_probabilities, cfg)
            checkpoint = {
                "model": cfg.model, "model_state_dict": model.state_dict(),
                "epoch": epoch, "seed": cfg.seed, "val_loss": val_loss,
                "val_breast_auc": val_auc, "config": asdict(cfg),
                "training_class_weights": loss_weights if cfg.class_weighted else None,
                "manifest_sha256": manifest_sha, "splits_sha256": splits_sha,
                "pretrained_weights": WEIGHT_IDS[cfg.model] if cfg.pretrained else None,
                "torch_version": str(torch.__version__),
                "torchvision_version": str(torchvision.__version__),
            }
            temp_path = path.with_suffix(".tmp")
            try:
                torch.save(checkpoint, temp_path)
                os.replace(temp_path, path)
            finally:
                temp_path.unlink(missing_ok=True)
        stagnant_epochs = 0 if strict_auc_improvement else stagnant_epochs + 1
        if stagnant_epochs >= cfg.early_stopping_patience:
            print(f"Early stopping after {epoch} epochs (no AUC increase for {stagnant_epochs})")
            break
    history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    config_path.write_text(json.dumps({
        **asdict(cfg), "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__), "python_version": sys.version,
        "pretrained_weights": WEIGHT_IDS[cfg.model] if cfg.pretrained else None,
        "manifest_sha256": manifest_sha, "splits_sha256": splits_sha,
        "training_class_weights": loss_weights if cfg.class_weighted else None,
        "normalization_mean": IMAGENET_MEAN if cfg.normalization == "imagenet" else None,
        "normalization_std": IMAGENET_STD if cfg.normalization == "imagenet" else None,
        "best_epoch": best[2], "best_breast_auc": best[0],
    }, indent=2, default=str) + "\n", encoding="utf-8")
    _write_prediction_csvs(output, cfg, best_predictions)
    print(f"Best validation checkpoint: {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate splits and train approved image-level baseline")
    parser.add_argument("--manifest", type=Path, default=Path("data/cdd_cesm_low_energy_splits.csv"))
    parser.add_argument("--splits", type=Path, default=Path("data/splits.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/training_baseline.json"))
    parser.add_argument("--images", type=Path)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--allow-incomplete-views", action="store_true")
    parser.add_argument("--allow-conflicting-labels", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    records, summary = preflight(args.manifest, args.splits)
    for split in ("train", "val", "test"):
        print(f"{split}: {summary.patient_counts[split]} patients, "
              f"{summary.breast_counts[split]} breasts, {summary.image_counts[split]} images")
    if summary.incomplete_breasts:
        print(f"NOTICE: {summary.incomplete_breasts} breasts lack a complete CC/MLO pair; "
              "confirm the breast-level evaluation inclusion criteria")
    if summary.conflicting_label_breasts:
        count = summary.conflicting_label_breasts
        noun = "breast has" if count == 1 else "breasts have"
        print(f"NOTICE: {count} {noun} different CC/MLO image labels; "
              "review source annotations before training")
    if not args.train:
        print("Preflight complete. No images loaded and no model training performed.")
        return 0
    if summary.incomplete_breasts and not args.allow_incomplete_views:
        parser.error("Incomplete CC/MLO pairs require an approved inclusion policy; "
                     "pass --allow-incomplete-views only if the team agrees")
    if summary.conflicting_label_breasts and not args.allow_conflicting_labels:
        parser.error("Conflicting CC/MLO image labels require source review; "
                     "pass --allow-conflicting-labels only after confirming the annotations")
    if args.images is None:
        parser.error("--images is required with --train")
    image_paths = index_images(args.images, [r for r in records if r.split in ("train", "val")])
    run_training(records, image_paths, cfg, args.manifest, args.splits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
