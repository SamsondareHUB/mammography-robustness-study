"""Configuration and command-line tests for the training setup."""

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from mammography_robustness.training import load_config, main


EXAMPLE_CONFIG = {
    "model": "mobilenet_v3_small", "image_size": 224, "batch_size": 2,
    "epochs": 1, "learning_rate": 0.0001, "num_workers": 0,
    "seed": 42, "output_dir": "artifacts/checkpoints",
}


class TrainingSetupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = self.root / "config.json"
        self.write_config(EXAMPLE_CONFIG)

    def write_config(self, d):
        self.config.write_text(json.dumps(d), encoding="utf-8")

    def write_inputs(self, incomplete=False, conflicting=False):
        csv_path = self.root / "manifest.csv"
        splits = self.root / "splits.json"
        ids = [("1", "train"), ("2", "val"), ("3", "test")]
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["patient_id", "case_id", "breast_side", "view", "label", "split"])
            for p, s in ids:
                for v in (["CC"] if incomplete and p == "3" else ["CC", "MLO"]):
                    label = "1" if conflicting and p == "1" and v == "MLO" else "0"
                    writer.writerow([p, f"P{p}_L_DM_{v}", "L", v, label, s])
        splits.write_text(json.dumps({"train": [1], "val": [2], "test": [3]}))
        return csv_path, splits

    def test_configuration_loads(self):
        self.assertEqual(load_config(self.config).model, "mobilenet_v3_small")

    def test_reject_unknown_model(self):
        self.write_config({**EXAMPLE_CONFIG, "model": "not-a-model"})
        with self.assertRaisesRegex(ValueError, "model must be"):
            load_config(self.config)

    def test_reject_missing_config_field(self):
        self.write_config({"model": "resnet18"})
        with self.assertRaisesRegex(ValueError, "missing or unexpected"):
            load_config(self.config)

    def test_reject_invalid_hyperparameter(self):
        self.write_config({**EXAMPLE_CONFIG, "batch_size": 0})
        with self.assertRaisesRegex(ValueError, "batch_size"):
            load_config(self.config)

    def test_preflight_never_trains(self):
        csv_path, splits = self.write_inputs()
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = main(["--config", str(self.config), "--manifest", str(csv_path),
                         "--splits", str(splits)])
        self.assertEqual(code, 0)
        self.assertIn("No images loaded and no model training performed", buffer.getvalue())
        self.assertFalse((self.root / "artifacts").exists())

    def test_run_training_flag_requires_image_dir(self):
        csv_path, splits = self.write_inputs()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            main(["--config", str(self.config), "--manifest", str(csv_path),
                  "--splits", str(splits), "--train"])
        self.assertEqual(caught.exception.code, 2)

    def test_incomplete_views_require_approval_flag(self):
        csv_path, splits = self.write_inputs(incomplete=True)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            main(["--config", str(self.config), "--manifest", str(csv_path),
                  "--splits", str(splits), "--train"])
        self.assertEqual(caught.exception.code, 2)

    def test_preflight_reports_conflicting_image_labels(self):
        csv_path, splits = self.write_inputs(conflicting=True)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = main(["--config", str(self.config), "--manifest", str(csv_path),
                         "--splits", str(splits)])
        self.assertEqual(code, 0)
        self.assertIn("1 breast has different CC/MLO image labels", buffer.getvalue())
        self.assertIn("No images loaded and no model training performed", buffer.getvalue())

    def test_training_requires_conflict_review(self):
        csv_path, splits = self.write_inputs(conflicting=True)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(buffer), self.assertRaises(SystemExit) as caught:
            main(["--config", str(self.config), "--manifest", str(csv_path),
                  "--splits", str(splits), "--train", "--images", str(self.root)])
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("require source review", buffer.getvalue())

    def test_conflict_review_flag_keeps_training_opt_in(self):
        csv_path, splits = self.write_inputs(conflicting=True)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = main(["--config", str(self.config), "--manifest", str(csv_path),
                         "--splits", str(splits), "--allow-conflicting-labels"])
        self.assertEqual(code, 0)
        self.assertIn("No images loaded and no model training performed", buffer.getvalue())

    def test_build_model_output_shapes(self):
        try:
            import torch
            import torchvision
            from mammography_robustness.training import create_model
        except ImportError:
            self.skipTest("torch and torchvision are optional for preflight tests")
        for name in ("mobilenet_v3_small", "resnet18"):
            with self.subTest(model=name):
                model = create_model(name).eval()
                with torch.no_grad():
                    output = model(torch.zeros((1, 3, 64, 64)))
                self.assertEqual(tuple(output.shape), (1, 2))


if __name__ == "__main__":
    unittest.main()
