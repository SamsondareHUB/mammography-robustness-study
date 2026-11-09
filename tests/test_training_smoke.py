"""Synthetic end-to-end checks for the baseline training artifact contract."""

import csv
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mammography_robustness.training import TrainingConfig, run_training
from mammography_robustness.clean_inference import _load_checkpoint
from mammography_robustness.training_data import ImageRecord


class TrainingSmokeTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch
            import torchvision  # noqa: F401
            from PIL import Image
        except ImportError:
            self.skipTest("Optional training dependencies are not installed")
        self.torch = torch
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.image_dir = self.root / "images"
        self.image_dir.mkdir()
        self.rows = [
            ImageRecord("1", "train_neg", "L", "CC", 0, "train"),
            ImageRecord("2", "train_pos", "L", "CC", 1, "train"),
            ImageRecord("3", "val_neg", "R", "MLO", 0, "val"),
            ImageRecord("4", "val_pos", "R", "MLO", 1, "val"),
        ]
        self.paths = {}
        for idx, row in enumerate(self.rows):
            path = self.image_dir / (row.case_id + ".png")
            Image.new("RGB", (55, 80), (25 + 40 * idx, 70, 100)).save(path)
            self.paths[row.case_id] = path
        self.manifest = self.root / "manifest.csv"
        self.splits = self.root / "splits.json"
        self.manifest.write_text("synthetic manifest", encoding="utf-8")
        self.splits.write_text("synthetic splits", encoding="utf-8")
        self.cfg = TrainingConfig(
            model="mobilenet_v3_small", image_size=64, batch_size=2,
            epochs=1, learning_rate=0.0001, num_workers=0, seed=42,
            output_dir=str(self.root / "results"), pretrained=True,
            normalization="imagenet", flip_probability=0,
            rotation_degrees=0,
        )

    def tiny_model(self):
        torch = self.torch

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = torch.nn.Linear(3, 2)

            def forward(self, images):
                return self.head(images.mean(dim=(2, 3)))

        return TinyModel()

    def test_one_epoch_records_checkpoint_and_selected_predictions(self):
        with patch("mammography_robustness.training.create_model", side_effect=lambda *a, **k: self.tiny_model()):
            run_training(self.rows, self.paths, self.cfg, self.manifest, self.splits)
        out = Path(self.cfg.output_dir)
        stem = "mobilenet_v3_small_seed42"
        self.assertEqual(len(list(out.iterdir())), 5)
        ckpt = self.torch.load(out / f"{stem}_best.pt", map_location="cpu", weights_only=True)
        self.assertEqual(ckpt["pretrained_weights"], "MobileNet_V3_Small_Weights.IMAGENET1K_V1")
        self.assertEqual(ckpt["config"]["normalization"], "imagenet")
        self.assertEqual(ckpt["epoch"], 1)
        self.assertRegex(ckpt["git_commit"], r"^[0-9a-f]{40}$")
        self.assertEqual(
            ckpt["git_commit"], subprocess.check_output(
                ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
                text=True,
            ).strip(),
        )
        self.assertIsInstance(ckpt["git_dirty"], bool)
        self.assertIn("python_version", ckpt)
        self.assertIn("pillow_version", ckpt)
        with (out / f"{stem}_config.json").open(encoding="utf-8") as stream:
            saved = json.load(stream)
        for name in ("git_commit", "git_dirty", "python_version", "pillow_version"):
            self.assertEqual(saved[name], ckpt[name])
        state, name, seed, size, batch = _load_checkpoint(
            out / f"{stem}_best.pt",
            hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
            hashlib.sha256(self.splits.read_bytes()).hexdigest(),
        )
        self.assertEqual((name, seed, size, batch), ("mobilenet_v3_small", 42, 64, 2))
        self.assertEqual(state["pretrained_weights"], ckpt["pretrained_weights"])
        with (out / f"{stem}_val_breast_predictions.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        self.assertEqual({r["label"] for r in rows}, {"0", "1"})

    def test_existing_prediction_file_is_never_overwritten(self):
        out = Path(self.cfg.output_dir)
        out.mkdir()
        previous = out / "mobilenet_v3_small_seed42_val_breast_predictions.csv"
        previous.write_text("important previous run", encoding="utf-8")
        with patch("mammography_robustness.training.create_model") as create:
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                run_training(self.rows, self.paths, self.cfg, self.manifest, self.splits)
            create.assert_not_called()
        self.assertEqual(previous.read_text(encoding="utf-8"), "important previous run")


if __name__ == "__main__":
    unittest.main()
