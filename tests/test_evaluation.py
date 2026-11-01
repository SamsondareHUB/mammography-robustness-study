"""Tests use synthetic prediction records; no mammography data is included."""

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from mammography_robustness.evaluation import (
    RUN_COLUMNS, SUMMARY_COLUMNS, binary_metrics, evaluate_predictions, fixed_thresholds,
    load_predictions, main, roc_auc, summarize_over_seeds, validate_predictions,
    validation_thresholds, youden_threshold,
)
from mammography_robustness.predictions import (
    BREAST_COLUMNS, IMAGE_COLUMNS, ImagePrediction, export_prediction_tables,
)


def breast_rows(model="small", seed=1, n_patients=12, shift=0.0, conditions=(("noise", "low"),)):
    """Two breasts per patient; the left breast is malignant for even patients."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_patients):
        for side in ("L", "R"):
            label = int(side == "L" and p % 2 == 0)
            base = float(np.clip(0.3 + 0.4 * label + rng.normal(0, 0.15), 0, 1))
            rows.append([f"p{p}", side, label, base, model, "clean", "none", seed])
            for degradation, severity in conditions:
                score = float(np.clip(base - shift * label + shift * (1 - label), 0, 1))
                rows.append([f"p{p}", side, label, score, model, degradation, severity, seed])
    frame = pd.DataFrame(rows, columns=BREAST_COLUMNS)
    return validate_predictions(frame, "breast")


def write_csv(path, header, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


class MetricTests(unittest.TestCase):
    def test_auc_with_ties(self):
        # Pairs (pos, neg): (0.8>0.2), (0.8>0.5), (0.5=0.5 -> 0.5), (0.5>0.2) => 3.5 / 4
        self.assertAlmostEqual(roc_auc([1, 1, 0, 0], [0.8, 0.5, 0.5, 0.2]), 0.875)
        self.assertTrue(math.isnan(roc_auc([1, 1], [0.2, 0.3])))

    def test_auc_matches_sklearn(self):
        try:
            from sklearn.metrics import roc_auc_score
        except ImportError:
            self.skipTest("scikit-learn not installed")
        rng = np.random.default_rng(0)
        labels = rng.integers(0, 2, 200)
        scores = np.round(rng.random(200), 1)  # rounding forces many ties
        self.assertAlmostEqual(roc_auc(labels, scores), roc_auc_score(labels, scores))

    def test_threshold_metrics(self):
        labels = [1, 1, 1, 0, 0, 0, 0]
        scores = [0.9, 0.5, 0.2, 0.6, 0.4, 0.1, 0.5]
        # At 0.5 (inclusive): TP=2 FN=1 FP=2 TN=2
        result = binary_metrics(labels, scores, 0.5)
        self.assertAlmostEqual(result["sensitivity"], 2 / 3)
        self.assertAlmostEqual(result["specificity"], 2 / 4)
        self.assertAlmostEqual(result["f1"], 4 / 7)

    def test_undefined_metrics_are_nan(self):
        result = binary_metrics([0, 0], [0.1, 0.2], 0.5)
        self.assertTrue(math.isnan(result["sensitivity"]))
        self.assertTrue(math.isnan(result["f1"]))
        self.assertEqual(result["specificity"], 1.0)

    def test_youden_picks_best_and_highest_on_ties(self):
        self.assertEqual(youden_threshold([0, 0, 1, 1], [0.1, 0.3, 0.6, 0.9]), 0.6)
        # 0.4 and 0.6 both give J = 0.5; the higher one wins.
        self.assertEqual(youden_threshold([0, 1, 0, 1], [0.1, 0.4, 0.5, 0.6]), 0.6)
        with self.assertRaises(ValueError):
            youden_threshold([1, 1], [0.2, 0.4])


class ValidationTests(unittest.TestCase):
    def test_rejects_bad_tables(self):
        good = breast_rows().copy()
        good["label"] = good["label"].astype(str)
        cases = {
            "score": lambda f: f.assign(score=f["score"].astype(str).where(f.index != 3, "1.5")),
            "label": lambda f: f.assign(label=f["label"].where(f.index != 0, "2")),
            "severity": lambda f: f.assign(severity=f["severity"].where(f.index != 0, "low")),
            "duplicate": lambda f: pd.concat([f, f.iloc[[0]]]),
            "baseline": lambda f: f[f["degradation"] != "clean"],
            "coverage": lambda f: f.drop(index=1),
        }
        for name, mutate in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                validate_predictions(mutate(good.astype(str)), "breast")

    def test_detects_level_from_header(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "x.csv"
            write_csv(path, ["patient_id", "score"], [["p1", "0.2"]])
            with self.assertRaises(ValueError):
                load_predictions(path)


class EvaluationTests(unittest.TestCase):
    def test_clean_deltas_zero_and_degradation_lowers_auc(self):
        frame = breast_rows(shift=0.2)
        result = evaluate_predictions(frame, "breast", fixed_thresholds(frame, 0.5), n_bootstrap=0)
        self.assertEqual(tuple(result.columns), RUN_COLUMNS)
        clean, noisy = result.iloc[0], result.iloc[1]
        self.assertEqual((clean["degradation"], noisy["degradation"]), ("clean", "noise"))
        self.assertEqual(clean["delta_auc"], 0)
        self.assertLess(noisy["auc"], clean["auc"])
        self.assertAlmostEqual(noisy["delta_auc"], noisy["auc"] - clean["auc"])
        self.assertEqual((clean["n_patients"], clean["n_units"], clean["n_positive"]), (12, 24, 6))
        self.assertTrue(math.isnan(clean["auc_ci_low"]))

    def test_bootstrap_is_reproducible_and_brackets_estimate(self):
        frame = pd.concat([breast_rows(shift=0.2), breast_rows(model="large", shift=0.1)])
        thresholds = fixed_thresholds(frame, 0.5)
        a = evaluate_predictions(frame, "breast", thresholds, n_bootstrap=200, bootstrap_seed=3)
        b = evaluate_predictions(frame, "breast", thresholds, n_bootstrap=200, bootstrap_seed=3)
        pd.testing.assert_frame_equal(a, b)
        self.assertTrue((a["auc_ci_low"] <= a["auc"]).all() and (a["auc"] <= a["auc_ci_high"]).all())
        # Removing another model must not change this model's intervals.
        alone = evaluate_predictions(breast_rows(shift=0.2), "breast", thresholds,
                                     n_bootstrap=200, bootstrap_seed=3)
        pd.testing.assert_frame_equal(alone, a[a["model"] == "small"].reset_index(drop=True))

    def test_validation_threshold_is_frozen_across_conditions(self):
        validation = breast_rows(seed=7)
        validation["seed"] = 1
        thresholds = validation_thresholds(validation)
        expected = youden_threshold(*[validation.query("degradation == 'clean'")[c] for c in ("label", "score")])
        self.assertEqual(thresholds[("small", 1)], (expected, "youden_validation"))
        result = evaluate_predictions(breast_rows(shift=0.2), "breast", thresholds, n_bootstrap=0)
        self.assertTrue((result["threshold"] == expected).all())
        with self.assertRaises(ValueError):
            evaluate_predictions(breast_rows(model="other"), "breast", thresholds, n_bootstrap=0)

    def test_summary_over_seeds(self):
        frame = pd.concat([breast_rows(seed=1, shift=0.2), breast_rows(seed=2, shift=0.2)])
        per_run = evaluate_predictions(frame, "breast", fixed_thresholds(frame, 0.5), n_bootstrap=0)
        summary = summarize_over_seeds(per_run)
        self.assertEqual(tuple(summary.columns), SUMMARY_COLUMNS)
        self.assertEqual(len(summary), 2)
        clean = per_run[per_run["degradation"] == "clean"]["auc"]
        self.assertAlmostEqual(summary.iloc[0]["auc_mean"], clean.mean())
        self.assertAlmostEqual(summary.iloc[0]["auc_sd"], clean.std(ddof=1))
        self.assertEqual(summary.iloc[0]["n_seeds"], 2)


def with_extra_columns(frame):
    """Every third patient single-view, patient p2 flagged, odd patients' right breast benign."""
    out = frame.copy()
    number = out["patient_id"].str[1:].astype(int)
    out["view_status"] = np.where(number % 3 == 0, "single_view", "paired_view")
    out["original_class"] = np.where(out["label"] == 1, "malignant",
                                     np.where((number % 2 == 1) & (out["breast_side"] == "R"), "benign", "normal"))
    out["label_flag"] = (out["patient_id"] == "p2").astype(int)
    return out


class OptionalColumnTests(unittest.TestCase):
    def setUp(self):
        self.frame = with_extra_columns(breast_rows(shift=0.2))

    def test_accepts_optional_columns_in_any_order(self):
        reordered = self.frame[["label_flag", *BREAST_COLUMNS[:3], "view_status", *BREAST_COLUMNS[3:], "original_class"]]
        result = validate_predictions(reordered.astype(str), "breast")
        self.assertEqual(result["label_flag"].sum(), 4)

    def test_rejects_bad_optional_values(self):
        cases = {
            "unknown column": lambda f: f.assign(extra="x"),
            "view_status": lambda f: f.assign(view_status="both"),
            "class mismatch": lambda f: f.assign(original_class="normal"),
            "flag value": lambda f: f.assign(label_flag="maybe"),
            # Row 5 is p1's left breast (paired) under noise; its clean row stays paired.
            "inconsistent": lambda f: f.assign(view_status=f["view_status"].where(f.index != 5, "single_view")),
        }
        for name, mutate in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                validate_predictions(mutate(self.frame.astype(str)), "breast")

    def test_subsets_and_false_positive_sources(self):
        frame = validate_predictions(self.frame.astype(str), "breast")
        result = evaluate_predictions(frame, "breast", fixed_thresholds(frame, 0.5), n_bootstrap=20)
        clean = result[result["degradation"] == "clean"].set_index("subset")
        self.assertEqual(list(clean.index), ["all", "paired_view", "single_view", "unflagged"])
        # p0, p3, p6, p9 are single-view: 8 breasts out of 24.
        self.assertEqual(clean.loc["single_view", "n_units"], 8)
        self.assertEqual(clean.loc["paired_view", "n_units"], 16)
        self.assertEqual(clean.loc["unflagged", "n_units"], 22)
        self.assertTrue((result["threshold"] == 0.5).all())
        self.assertTrue((result["fp_normal"] + result["fp_benign"] == result["fp"]).all())
        self.assertTrue((result["n_normal"] + result["n_benign"] == result["n_negative"]).all())
        self.assertEqual(clean.loc["all", "n_benign"], 6)
        summary = summarize_over_seeds(result)
        self.assertEqual(sorted(summary["subset"].unique()), ["all", "paired_view", "single_view", "unflagged"])

    def test_class_counts_empty_without_original_class(self):
        frame = breast_rows()
        result = evaluate_predictions(frame, "breast", fixed_thresholds(frame, 0.5), n_bootstrap=0)
        self.assertEqual(set(result["subset"]), {"all"})
        self.assertTrue(result["fp_benign"].isna().all())

    def test_image_view_status_must_match_view_count(self):
        rows = []
        for view in ("CC", "MLO"):
            rows.append(["c1", "p1", "L", view, 1, 0.8, "m", "clean", "none", 1, "single_view"])
        rows.append(["c1", "p1", "R", "CC", 0, 0.2, "m", "clean", "none", 1, "single_view"])
        frame = pd.DataFrame(rows, columns=[*IMAGE_COLUMNS, "view_status"]).astype(str)
        with self.assertRaises(ValueError):
            validate_predictions(frame, "image")
        frame.loc[:1, "view_status"] = "paired_view"
        self.assertEqual(len(validate_predictions(frame, "image")), 3)


# Column layout of the exporter's breast metadata CSV (write_breast_metadata_csv).
EXPORTER_BREAST_METADATA = ("patient_id", "breast_side", "label", "view_status", "view_label_conflict",
                            "model", "degradation", "severity", "seed")


def exporter_breast_metadata(predictions):
    """Metadata in the exporter's format: p3 single-view, p4's left breast has conflicting views."""
    meta = predictions.copy()
    meta["view_status"] = np.where(meta["patient_id"] == "p3", "single", "paired")
    meta["view_label_conflict"] = ((meta["patient_id"] == "p4") & (meta["breast_side"] == "L")).astype(int)
    return meta[list(EXPORTER_BREAST_METADATA)]


class MetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.predictions = breast_rows(shift=0.2)
        self.pred_path = Path(self.temp.name, "breast.csv")
        self.meta_path = Path(self.temp.name, "breast_meta.csv")
        self.predictions.to_csv(self.pred_path, index=False)

    def load_with(self, metadata):
        metadata.to_csv(self.meta_path, index=False)
        return load_predictions(self.pred_path, self.meta_path)

    def test_joins_exporter_metadata_and_maps_names(self):
        frame, level = self.load_with(exporter_breast_metadata(self.predictions))
        self.assertEqual(level, "breast")
        self.assertEqual(set(frame["view_status"]), {"single_view", "paired_view"})
        self.assertEqual(frame["label_flag"].sum(), 2)  # p4 left, clean + noise
        result = evaluate_predictions(frame, "breast", fixed_thresholds(frame, 0.5), n_bootstrap=0)
        clean = result[result["degradation"] == "clean"].set_index("subset")
        self.assertEqual(clean.loc["single_view", "n_units"], 2)
        self.assertEqual(clean.loc["unflagged", "n_units"], 23)

    def test_metadata_row_order_does_not_matter(self):
        metadata = exporter_breast_metadata(self.predictions).iloc[::-1]
        frame, _ = self.load_with(metadata)
        pd.testing.assert_series_equal(frame["score"], self.predictions["score"])

    def test_rejects_mismatched_metadata(self):
        good = exporter_breast_metadata(self.predictions)
        cases = {
            "missing row": good.iloc[1:],
            "extra row": pd.concat([good, good.iloc[[0]].assign(seed=9)]),
            "duplicate row": pd.concat([good, good.iloc[[0]]]),
            "label disagrees": good.assign(label=1 - good["label"]),
            "no join column": good.drop(columns="seed"),
            "no useful field": good.drop(columns=["view_status", "view_label_conflict"]),
        }
        for name, metadata in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                self.load_with(metadata)

    def test_rejects_field_in_both_files(self):
        self.predictions.assign(view_status="paired_view").to_csv(self.pred_path, index=False)
        with self.assertRaises(ValueError):
            self.load_with(exporter_breast_metadata(self.predictions))


class CommandLineTests(unittest.TestCase):
    def test_metadata_option(self):
        with tempfile.TemporaryDirectory() as temp:
            pred, meta, out = Path(temp, "b.csv"), Path(temp, "m.csv"), Path(temp, "metrics")
            predictions = breast_rows(shift=0.2)
            predictions.to_csv(pred, index=False)
            exporter_breast_metadata(predictions).to_csv(meta, index=False)
            main([str(pred), "--metadata", str(meta), "--bootstrap", "0", "--out-dir", str(out)])
            result = pd.read_csv(out / "breast_metrics_by_run.csv")
            info = json.loads((out / "breast_evaluation_metadata.json").read_text())
        self.assertEqual(set(result["subset"]), {"all", "paired_view", "single_view", "unflagged"})
        self.assertEqual(info["metadata"]["path"], str(meta))

    def test_clean_only_and_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            test_path, val_path, out = Path(temp, "test.csv"), Path(temp, "val.csv"), Path(temp, "metrics")
            with_extra_columns(breast_rows(shift=0.2)).to_csv(test_path, index=False)
            validation = breast_rows(seed=7)
            validation["seed"] = 1
            validation.to_csv(val_path, index=False)
            main([str(test_path), "--threshold", "youden", "--validation", str(val_path),
                  "--clean-only", "--bootstrap", "0", "--out-dir", str(out)])
            result = pd.read_csv(out / "breast_metrics_by_run.csv")
            metadata = json.loads((out / "breast_evaluation_metadata.json").read_text())
        self.assertEqual(set(result["degradation"]), {"clean"})
        self.assertEqual(metadata["validation"]["path"], str(val_path))
        self.assertEqual(len(metadata["validation"]["sha256"]), 64)
        self.assertTrue(metadata["clean_only"])
        self.assertEqual(metadata["thresholds"][0]["threshold"], result["threshold"].iloc[0])

    def test_end_to_end_from_exported_tables(self):
        records = []
        for p in range(6):
            label = p % 2
            for view, bump in (("CC", 0.0), ("MLO", 0.1)):
                score = 0.2 + 0.5 * label + bump
                for degradation, severity, s in (("clean", "none", score), ("blur", "high", score * 0.8)):
                    records.append(ImagePrediction(f"c{p}", f"p{p}", "L", f"p{p}-{view}", view, label,
                                                   s, "small", degradation, severity, 1))
        with tempfile.TemporaryDirectory() as temp:
            images, breasts, out = Path(temp, "img.csv"), Path(temp, "breast.csv"), Path(temp, "metrics")
            export_prediction_tables(records, images, breasts)
            main([str(breasts), "--threshold", "youden", "--validation", str(breasts),
                  "--bootstrap", "50", "--out-dir", str(out)])
            main([str(images), "--bootstrap", "0", "--out-dir", str(out)])
            breast_result = pd.read_csv(out / "breast_metrics_by_run.csv")
            image_result = pd.read_csv(out / "image_metrics_by_run.csv")
            self.assertTrue((out / "breast_metrics_summary.csv").exists())
        self.assertEqual(breast_result["threshold_source"].iloc[0], "youden_validation")
        self.assertEqual(list(breast_result["n_units"]), [6, 6])
        self.assertEqual(list(image_result["n_units"]), [12, 12])
        self.assertEqual(breast_result["auc"].iloc[0], 1.0)

    def test_youden_requires_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "b.csv"
            breast_rows().to_csv(path, index=False)
            with self.assertRaises(SystemExit):
                main([str(path), "--threshold", "youden"])


if __name__ == "__main__":
    unittest.main()
