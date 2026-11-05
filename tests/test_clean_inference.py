"""Synthetic checks for clean checkpoint inference and metadata provenance."""

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mammography_robustness.clean_inference import (
    _load_checkpoint, export_clean_predictions, load_original_classes, predict_images,
)
from mammography_robustness.training_data import ImageRecord


FIELDS = ("patient_id", "case_id", "breast_side", "view", "label", "split", "original_class")


def sample_rows():
    return [
        (ImageRecord("99", "P99_L_DM_CC", "L", "CC", 0, "train"), "normal"),
        (ImageRecord("99", "P99_L_DM_MLO", "L", "MLO", 0, "train"), "normal"),
        (ImageRecord("99", "P99_L_DM_MLO_2", "L", "MLO", 0, "train"), "normal"),
        (ImageRecord("281", "P281_L_DM_CC", "L", "CC", 0, "val"), "normal"),
        (ImageRecord("281", "P281_L_DM_MLO", "L", "MLO", 0, "val"), "normal"),
        (ImageRecord("281", "P281_L_DM_MLO2", "L", "MLO", 0, "val"), "normal"),
        (ImageRecord("16", "P16_R_DM_CC", "R", "CC", 0, "test"), "normal"),
        (ImageRecord("16", "P16_R_DM_MLO", "R", "MLO", 1, "test"), "malignant"),
    ]


class OriginalClassTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.annotation = Path(self.tmp.name) / "original_classes.csv"
        self.pairs = sample_rows()

    def write(self, pairs=None):
        with self.annotation.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=FIELDS)
            writer.writeheader()
            for row, original in self.pairs if pairs is None else pairs:
                writer.writerow({**row.__dict__, "original_class": original})
        return self.annotation

    def test_complete_annotations_join_by_case_id(self):
        self.assertEqual(
            load_original_classes(self.write(list(reversed(self.pairs))), [r for r, _ in self.pairs]),
            {row.case_id: original for row, original in self.pairs},
        )

    def test_missing_class_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Original classes missing"):
            load_original_classes(self.write(self.pairs[:-1]), [r for r, _ in self.pairs])

    def test_inconsistent_binary_label_is_rejected(self):
        pairs = self.pairs.copy()
        pairs[-1] = (pairs[-1][0], "normal")
        with self.assertRaisesRegex(ValueError, "binary labels disagree"):
            load_original_classes(self.write(pairs), [r for r, _ in self.pairs])

    def test_patient_and_split_mismatch_are_rejected(self):
        from dataclasses import replace

        for field, value in (("patient_id", "999"), ("split", "train")):
            with self.subTest(field=field):
                pairs = self.pairs.copy()
                pairs[3] = (replace(pairs[3][0], **{field: value}), pairs[3][1])
                with self.assertRaisesRegex(ValueError, "disagrees with manifest"):
                    load_original_classes(self.write(pairs), [r for r, _ in self.pairs])

    def test_duplicate_case_id_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Repeated annotation case_id"):
            load_original_classes(self.write(self.pairs + [self.pairs[0]]), [r for r, _ in self.pairs])

    def test_unknown_class_is_rejected(self):
        pairs = self.pairs.copy()
        pairs[0] = (pairs[0][0], "unverified")
        with self.assertRaisesRegex(ValueError, "Unknown original_class"):
            load_original_classes(self.write(pairs), [r for r, _ in self.pairs])


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch
            from PIL import Image  # noqa: F401
            import torchvision  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch, torchvision and Pillow are optional until training")
        self.torch = torch
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.pairs = sample_rows()
        self.manifest = self.root / "manifest.csv"
        self.annotations = self.root / "annotations.csv"
        self.splits = self.root / "splits.json"
        self.images = self.root / "images"
        self.images.mkdir()
        with self.manifest.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS[:-1])
            writer.writeheader()
            for r, _ in self.pairs:
                writer.writerow(r.__dict__)
        with self.annotations.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            for r, original in self.pairs:
                writer.writerow({**r.__dict__, "original_class": original})
        self.splits.write_text(json.dumps({"train": [99], "val": [281], "test": [16]}))
        from PIL import Image
        for r, _ in self.pairs:
            Image.new("RGB", (64, 80), color=(16, 32, 64)).save(self.images / f"{r.case_id}.png")

    def model(self):
        torch = self.torch

        class FixedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, x):
                z = self.bias.expand(x.size(0))
                return torch.stack((torch.zeros_like(z), z), dim=1)

        return FixedModel()

    def make_checkpoint(self, seed, name="resnet18", valid_hashes=True):
        state = {
            "model": name,
            "seed": seed,
            "model_state_dict": self.model().state_dict(),
            "config": {"model": name, "seed": seed, "image_size": 64, "batch_size": 2},
            "manifest_sha256": hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
            "splits_sha256": hashlib.sha256(self.splits.read_bytes()).hexdigest(),
        }
        if not valid_hashes:
            state["splits_sha256"] = "0" * 64
        path = self.root / f"seed{seed}.pt"
        self.torch.save(state, path)
        return path

    def test_hash_mismatch_is_rejected(self):
        cp = self.make_checkpoint(1, valid_hashes=False)
        with self.assertRaisesRegex(ValueError, "mismatched splits_sha256"):
            _load_checkpoint(cp, hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
                             hashlib.sha256(self.splits.read_bytes()).hexdigest())

    def test_softmax_selects_malignant_logit(self):
        rows = [r for r, _ in self.pairs if r.split == "test"]
        scores = predict_images(self.model(), rows,
                                {r.case_id: self.images / f"{r.case_id}.png" for r in rows},
                                64, 2, self.torch.device("cpu"))
        self.assertEqual(len(scores), 2)
        self.assertTrue(all(abs(score - 0.7310586) < 1e-6 for score in scores))

    def test_export_covers_selected_images_and_breast_metadata(self):
        with patch("mammography_robustness.clean_inference.create_model", side_effect=lambda name: self.model()):
            counts = export_clean_predictions(
                manifest=self.manifest, splits=self.splits, annotations=self.annotations,
                images=self.images, checkpoints=[self.make_checkpoint(1), self.make_checkpoint(2)],
                output_dir=self.root / "output", device_name="cpu",
            )
        self.assertEqual(counts, {"val": 2, "test": 2})
        out = self.root / "output"
        with (out / "val_image_predictions.csv").open(newline="") as f:
            images = list(csv.DictReader(f))
        self.assertEqual(len(images), 4)
        self.assertNotIn("P281_L_DM_MLO2", {r["case_id"] for r in images})
        self.assertEqual({r["seed"] for r in images}, {"1", "2"})
        with (out / "test_breast_metadata.csv").open(newline="") as f:
            breasts = list(csv.DictReader(f))
        self.assertEqual(len(breasts), 2)
        self.assertEqual({r["original_class"] for r in breasts}, {"malignant"})
        self.assertEqual({r["view_label_conflict"] for r in breasts}, {"1"})
        with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
            with patch("mammography_robustness.clean_inference.create_model", side_effect=lambda name: self.model()):
                export_clean_predictions(
                    manifest=self.manifest, splits=self.splits, annotations=self.annotations,
                    images=self.images, checkpoints=[self.make_checkpoint(3)],
                    output_dir=out, device_name="cpu",
                )

    def test_duplicate_checkpoint_seed_rejected(self):
        cp = self.make_checkpoint(1)
        with patch("mammography_robustness.clean_inference.create_model", side_effect=lambda name: self.model()):
            with self.assertRaisesRegex(ValueError, "Duplicate model/seed"):
                export_clean_predictions(
                    manifest=self.manifest, splits=self.splits, annotations=self.annotations,
                    images=self.images, checkpoints=[cp, cp],
                    output_dir=self.root / "dupes", device_name="cpu",
                )


if __name__ == "__main__":
    unittest.main()
