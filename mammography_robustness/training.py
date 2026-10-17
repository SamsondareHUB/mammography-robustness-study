"""Image-level classification baseline and dataset preflight checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from .training_data import index_images, read_manifest, read_patient_splits, validate_records


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


MODEL_NAMES = ("mobilenet_v3_small", "resnet18")


def load_config(path: str | Path) -> TrainingConfig:
    """Validate a baseline configuration before loading model dependencies."""
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or set(payload) != set(TrainingConfig.__dataclass_fields__):
        raise ValueError("Training configuration has missing or unexpected fields")
    cfg = TrainingConfig(**payload)
    if cfg.model not in MODEL_NAMES:
        raise ValueError(f"model must be one of {MODEL_NAMES}")
    for name in ("image_size", "batch_size", "epochs", "seed"):
        value = getattr(cfg, name)
        if type(value) is not int or value < (64 if name == "image_size" else 0 if name == "seed" else 1):
            raise ValueError(f"Invalid {name}")
    if type(cfg.num_workers) is not int or cfg.num_workers < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    if type(cfg.learning_rate) not in (float, int) or not 0 < cfg.learning_rate < 1:
        raise ValueError("learning_rate must be between 0 and 1")
    if not isinstance(cfg.output_dir, str) or not cfg.output_dir.strip():
        raise ValueError("output_dir cannot be empty")
    return cfg


def preflight(manifest_path: Path, splits_path: Path):
    """Verify patient assignments and breast annotations before training starts."""
    records = read_manifest(manifest_path)
    summary = validate_records(records, read_patient_splits(splits_path))
    return records, summary


def create_model(name: str):
    """Instantiate a two-logit RGB classifier without downloading weights."""
    from torch import nn
    from torchvision.models import mobilenet_v3_small, resnet18

    if name == "resnet18":
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif name == "mobilenet_v3_small":
        model = mobilenet_v3_small(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
    else:
        raise ValueError(f"Unknown model: {name}")
    return model


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_training(records, image_paths, cfg: TrainingConfig, manifest_path: Path, splits_path: Path):
    """Train only when invoked explicitly with --train after successful preflight."""
    import torch
    from PIL import Image, ImageOps
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision.transforms import ToTensor

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tensorize = ToTensor()

    class MammogramDataset(Dataset):
        def __init__(self, split):
            self.rows = [row for row in records if row.split == split]
            if not self.rows:
                raise ValueError(f"No records for {split}")

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            row = self.rows[idx]
            with Image.open(image_paths[row.case_id]) as opened:
                rgb = opened.convert("RGB")
                # Pad rather than crop so that the full mammogram remains visible.
                resized = ImageOps.pad(rgb, (cfg.image_size, cfg.image_size),
                                       method=Image.Resampling.BILINEAR, color=(0, 0, 0))
            return tensorize(resized), row.label

    train_loader = DataLoader(MammogramDataset("train"), batch_size=cfg.batch_size,
                              shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(MammogramDataset("val"), batch_size=cfg.batch_size,
                            shuffle=False, num_workers=cfg.num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(cfg.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.CrossEntropyLoss()
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_sum = 0.0
        train_count = 0
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
        val_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                loss = criterion(model(images), labels)
                val_sum += loss.item() * labels.size(0)
                val_count += labels.size(0)
        val_loss = val_sum / val_count
        print(f"Epoch {epoch}/{cfg.epochs}: train_loss={train_sum / train_count:.4f}, "
              f"val_loss={val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "model": cfg.model, "model_state_dict": model.state_dict(),
                "epoch": epoch, "seed": cfg.seed, "val_loss": val_loss,
                "config": asdict(cfg),
                "manifest_sha256": _sha256(manifest_path),
                "splits_sha256": _sha256(splits_path),
            }
            temp_file = output / f".{cfg.model}_best.tmp"
            checkpoint_path = output / f"{cfg.model}_best.pt"
            try:
                torch.save(checkpoint, temp_file)
                os.replace(temp_file, checkpoint_path)
            finally:
                temp_file.unlink(missing_ok=True)
    print(f"Best validation checkpoint: {output / f'{cfg.model}_best.pt'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate splits and prepare image-level model training")
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/cdd_cesm_low_energy_splits.csv"))
    parser.add_argument("--splits", type=Path, default=Path("data/splits.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/training_baseline.json"))
    parser.add_argument("--images", type=Path, help="Directory containing the low-energy image files")
    parser.add_argument("--train", action="store_true", help="Explicitly enable model training")
    parser.add_argument("--allow-incomplete-views", action="store_true",
                        help="Permit image-level training with breasts lacking CC/MLO pairs")
    parser.add_argument("--allow-conflicting-labels", action="store_true",
                        help="Permit training with verified per-view label differences")
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
    image_paths = index_images(args.images, records)
    run_training(records, image_paths, cfg, args.manifest, args.splits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
