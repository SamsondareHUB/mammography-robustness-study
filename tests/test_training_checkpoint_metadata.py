"""Checkpoint provenance and approved baseline configuration checks."""

import json
import tempfile
import unittest
from pathlib import Path

from mammography_robustness.training import WEIGHT_IDS, load_config
from mammography_robustness.clean_inference import _load_checkpoint


class TrainingCheckpointMetadataTests(unittest.TestCase):
    def test_explicit_pretrained_weights_for_both_architectures(self):
        self.assertEqual(WEIGHT_IDS["mobilenet_v3_small"],
                         "MobileNet_V3_Small_Weights.IMAGENET1K_V1")
        self.assertEqual(WEIGHT_IDS["resnet18"], "ResNet18_Weights.IMAGENET1K_V1")

    def test_pretrained_requires_matching_normalization(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            basic = {
                "model": "mobilenet_v3_small", "image_size": 224, "batch_size": 16,
                "epochs": 30, "learning_rate": 0.0001, "num_workers": 0,
                "seed": 42, "output_dir": "artifacts/checkpoints",
                "pretrained": True, "normalization": "none",
            }
            path.write_text(json.dumps(basic), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ImageNet-pretrained"):
                load_config(path)
            basic["normalization"] = "imagenet"
            path.write_text(json.dumps(basic), encoding="utf-8")
            self.assertTrue(load_config(path).pretrained)

    def test_pretrained_checkpoint_without_provenance_is_rejected(self):
        import torch
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.pt"
            state = {
                "model": "mobilenet_v3_small", "seed": 42,
                "config": {"model": "mobilenet_v3_small", "seed": 42,
                           "image_size": 224, "batch_size": 16,
                           "pretrained": True, "normalization": "imagenet"},
                "model_state_dict": {}, "manifest_sha256": "abc", "splits_sha256": "def",
            }
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, "missing preprocessing or weight provenance"):
                _load_checkpoint(path, "abc", "def")

    def test_pretrained_checkpoint_with_wrong_weight_id_is_rejected(self):
        import torch
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.pt"
            state = {
                "model": "mobilenet_v3_small", "seed": 42,
                "config": {"model": "mobilenet_v3_small", "seed": 42,
                           "image_size": 224, "batch_size": 16,
                           "pretrained": True, "normalization": "imagenet"},
                "pretrained_weights": WEIGHT_IDS["resnet18"],
                "model_state_dict": {}, "manifest_sha256": "abc", "splits_sha256": "def",
            }
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, "weight provenance does not match"):
                _load_checkpoint(path, "abc", "def")


if __name__ == "__main__":
    unittest.main()
