"""Export clean validation and test predictions from trained model checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import tempfile
from pathlib import Path

from .preprocessing import make_transform
from .predictions import (
    ImagePrediction, export_prediction_tables, write_breast_metadata_csv,
    write_image_metadata_csv,
)
from .training import WEIGHT_IDS, create_model
from .training_data import (
    index_images, read_manifest, read_patient_splits,
    select_evaluation_images, validate_records,
)


ORIGINAL_CLASS_LABELS = {"normal": 0, "benign": 0, "malignant": 1}
OUTPUT_NAMES = {
    split: {
        "image": f"{split}_image_predictions.csv",
        "breast": f"{split}_breast_predictions.csv",
        "image_metadata": f"{split}_image_metadata.csv",
        "breast_metadata": f"{split}_breast_metadata.csv",
    }
    for split in ("val", "test")
}


def load_original_classes(path: Path, records) -> dict[str, str]:
    """Validate a complete, case-ID-aligned copy of the verified annotations."""
    expected = {row.case_id: row for row in records}
    required = {"case_id", "patient_id", "breast_side", "view", "label", "split", "original_class"}
    annotations: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Annotation CSV must include {', '.join(sorted(required))}")
        for line, source in enumerate(reader, start=2):
            if None in source:
                raise ValueError(f"Malformed annotation row {line}")
            values = {key: str(source.get(key) or "").strip() for key in required}
            case_id = values["case_id"]
            if case_id not in expected:
                raise ValueError(f"Annotation case_id not in manifest: {case_id}")
            if case_id in annotations:
                raise ValueError(f"Repeated annotation case_id: {case_id}")
            row = expected[case_id]
            for field in ("patient_id", "breast_side", "view", "label", "split"):
                if values[field] != str(getattr(row, field)):
                    raise ValueError(f"Annotation {field} disagrees with manifest for {case_id}")
            original = values["original_class"].lower()
            if original not in ORIGINAL_CLASS_LABELS:
                raise ValueError(f"Unknown original_class for {case_id}: {original!r}")
            if ORIGINAL_CLASS_LABELS[original] != row.label:
                raise ValueError(f"Original and binary labels disagree for {case_id}")
            annotations[case_id] = original
    missing = set(expected) - annotations.keys()
    if missing:
        raise ValueError(f"Original classes missing for {len(missing)} images")
    return annotations


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(data)
    return hasher.hexdigest()


def _load_checkpoint(path: Path, manifest_hash: str, splits_hash: str):
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid checkpoint format: {path}")
    for key, expected in (("manifest_sha256", manifest_hash), ("splits_sha256", splits_hash)):
        if checkpoint.get(key) != expected:
            raise ValueError(f"Checkpoint {path} has a mismatched {key}")
    name, seed, config = checkpoint.get("model"), checkpoint.get("seed"), checkpoint.get("config")
    if name not in ("mobilenet_v3_small", "resnet18"):
        raise ValueError(f"Unknown checkpoint model in {path}")
    if type(seed) is not int or seed < 0:
        raise ValueError(f"Invalid checkpoint seed in {path}")
    if not isinstance(config, dict) or config.get("model") != name or config.get("seed") != seed:
        raise ValueError(f"Checkpoint configuration disagrees with model/seed in {path}")
    size, batch = config.get("image_size"), config.get("batch_size")
    if type(size) is not int or size < 64 or type(batch) is not int or batch < 1:
        raise ValueError(f"Invalid preprocessing configuration in {path}")
    if config.get("normalization", "none") not in ("none", "imagenet"):
        raise ValueError("Checkpoint has an unsupported normalization configuration")
    if config.get("pretrained", False):
        if config.get("normalization") != "imagenet" or not checkpoint.get("pretrained_weights"):
            raise ValueError("Pretrained checkpoint is missing preprocessing or weight provenance")
        if checkpoint["pretrained_weights"] != WEIGHT_IDS[name]:
            raise ValueError("Pretrained checkpoint weight provenance does not match model")
    if not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ValueError(f"Missing model state in {path}")
    return checkpoint, name, seed, size, batch


def predict_images(model, records, image_paths, image_size: int, batch_size: int, device, *, normalization: str = "none"):
    """Return probabilities using the checkpoint's recorded image normalization."""
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    transform = make_transform(image_size, training=False, normalization=normalization)

    class ImageDataset(Dataset):
        def __len__(self):
            return len(records)

        def __getitem__(self, index):
            row = records[index]
            with Image.open(image_paths[row.case_id]) as opened:
                return transform(opened)

    model.eval()
    loader = DataLoader(ImageDataset(), batch_size=batch_size, shuffle=False, num_workers=0)
    scores = []
    with torch.inference_mode():
        for images in loader:
            logits = model(images.to(device))
            if logits.ndim != 2 or logits.shape[1] != 2:
                raise ValueError("Checkpoint model must output two class logits")
            batch_scores = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
            if any(not 0 <= value <= 1 for value in batch_scores):
                raise ValueError("Non-finite prediction score")
            scores.extend(float(value) for value in batch_scores)
    if len(scores) != len(records):
        raise ValueError("Inference did not cover every selected image")
    return scores


def export_clean_predictions(
    *, manifest: Path, splits: Path, annotations: Path, images: Path,
    checkpoints: list[Path], output_dir: Path, device_name: str = "auto",
) -> dict[str, int]:
    """Run all checkpoints on the same selected validation and test records."""
    if not checkpoints:
        raise ValueError("At least one checkpoint is required")
    if device_name not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda")
    destinations = [output_dir / name for files in OUTPUT_NAMES.values() for name in files.values()]
    existing = [path for path in destinations if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing predictions: {existing[0]}")
    records = read_manifest(manifest)
    validate_records(records, read_patient_splits(splits))
    original_classes = load_original_classes(annotations, records)
    selected = select_evaluation_images(records)
    by_split = {split: [r for r in selected if r.split == split] for split in ("val", "test")}
    if not all(by_split.values()):
        raise ValueError("Validation and test sets must be nonempty")
    image_paths = index_images(images, by_split["val"] + by_split["test"])

    import torch

    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    device = torch.device("cuda" if (device_name == "cuda" or (device_name == "auto" and torch.cuda.is_available())) else "cpu")
    manifest_hash, splits_hash = _sha256(manifest), _sha256(splits)
    predictions: dict[str, list[ImagePrediction]] = {"val": [], "test": []}
    seen_runs: set[tuple[str, int]] = set()
    for checkpoint_path in checkpoints:
        state, model_name, seed, image_size, batch_size = _load_checkpoint(checkpoint_path, manifest_hash, splits_hash)
        run_id = (model_name, seed)
        if run_id in seen_runs:
            raise ValueError(f"Duplicate model/seed checkpoint: {run_id}")
        seen_runs.add(run_id)
        model = create_model(model_name)
        model.load_state_dict(state["model_state_dict"], strict=True)
        model.to(device)
        normalization = state["config"].get("normalization", "none")
        for split, rows in by_split.items():
            scores = predict_images(
                model, rows, image_paths, image_size, batch_size, device,
                normalization=normalization,
            )
            predictions[split].extend(
                ImagePrediction(
                    case_id=row.case_id, patient_id=row.patient_id,
                    breast_side=row.breast_side, image_id=row.case_id,
                    view=row.view, label=row.label, score=score,
                    model=model_name, degradation="clean", severity="none", seed=seed,
                    original_class=original_classes[row.case_id],
                )
                for row, score in zip(rows, scores, strict=True)
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [path for path in destinations if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing predictions: {existing[0]}")
    with tempfile.TemporaryDirectory(prefix=".clean_exports_", dir=output_dir) as temp:
        staging = Path(temp)
        for split, names in OUTPUT_NAMES.items():
            rows = predictions[split]
            export_prediction_tables(rows, staging / names["image"], staging / names["breast"])
            write_image_metadata_csv(rows, staging / names["image_metadata"])
            write_breast_metadata_csv(rows, staging / names["breast_metadata"])
        for files in OUTPUT_NAMES.values():
            for name in files.values():
                os.replace(staging / name, output_dir / name)
    return {split: len(rows) for split, rows in by_split.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export clean validation and test predictions from checkpoints")
    parser.add_argument("--manifest", type=Path, default=Path("data/cdd_cesm_low_energy_splits.csv"))
    parser.add_argument("--splits", type=Path, default=Path("data/splits.json"))
    parser.add_argument("--annotations", type=Path, required=True, help="Verified image-level original three-class CSV")
    parser.add_argument("--images", type=Path, required=True, help="Root folder containing the low-energy image files")
    parser.add_argument("--checkpoint", type=Path, action="append", required=True, dest="checkpoints")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    counts = export_clean_predictions(
        manifest=args.manifest, splits=args.splits, annotations=args.annotations,
        images=args.images, checkpoints=args.checkpoints, output_dir=args.output_dir,
        device_name=args.device,
    )
    print(f"Exported clean predictions for {len(args.checkpoints)} model/seed checkpoints: "
          f"{counts['val']} validation images and {counts['test']} test images per run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
