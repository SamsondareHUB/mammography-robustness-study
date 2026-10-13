"""Tests use synthetic prediction records; no mammography data is included."""

import csv
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from mammography_robustness.predictions import (
    BREAST_COLUMNS, IMAGE_COLUMNS, BreastPrediction, ImagePrediction, SplitImage,
    aggregate_breast_predictions, export_prediction_tables, summarize_splits,
    write_breast_csv, write_image_csv, write_split_summary_json,
)


def image(view="CC", score=0.2, model="small", degradation="clean",
          severity="none", patient="p1", side="L", label=1, seed=1, image_id=None,
          case="c1"):
    return ImagePrediction(case, patient, side, image_id or f"{patient}-{side}-{view}",
                           view, label, score, model, degradation, severity, seed)


def set_of_views(model="small", patient="p1", side="L", label=1, seed=1):
    kwargs = dict(model=model, patient=patient, side=side, label=label, seed=seed)
    return [
        image("CC", 0.2, **kwargs), image("MLO", 0.6, **kwargs),
        image("CC", 0.35, degradation="noise", severity="low", **kwargs),
        image("MLO", 0.5, degradation="noise", severity="low", **kwargs),
    ]


class PredictionTests(unittest.TestCase):
    def test_fixed_max_aggregation(self):
        result = aggregate_breast_predictions(set_of_views())
        self.assertEqual(len(result), 2)
        self.assertEqual([r.score for r in result], [0.6, 0.5])
        self.assertEqual([r.degradation for r in result], ["clean", "noise"])

    def test_exact_headers_and_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp)/"images.csv", Path(temp)/"breasts.csv"
            export_prediction_tables(set_of_views(), a, b)
            with a.open(newline="") as f:
                img_rows = list(csv.DictReader(f))
            with b.open(newline="") as f:
                breast_rows = list(csv.DictReader(f))
            self.assertEqual(tuple(img_rows[0]), IMAGE_COLUMNS)
            self.assertEqual(tuple(breast_rows[0]), BREAST_COLUMNS)
            self.assertEqual(len(img_rows), 4)
            self.assertEqual(len(breast_rows), 2)
            self.assertEqual(float(breast_rows[0]["score"]), 0.6)
            self.assertEqual(img_rows[0]["view"], "CC")

    def test_model_runs_match(self):
        x = set_of_views()
        y = [replace(r, model="large") for r in x]
        self.assertEqual(len(aggregate_breast_predictions(x+y)), 4)

    def test_multiple_breasts_for_patient(self):
        left = set_of_views()
        right = [replace(r, breast_side="R", image_id="right-"+r.image_id) for r in left]
        result = aggregate_breast_predictions(left+right)
        self.assertEqual(len(result), 4)

    def test_case_id_not_required_unique_per_breast(self):
        left = set_of_views()
        right = [replace(r, breast_side="R", image_id="right-"+r.image_id) for r in left]
        self.assertEqual(len(aggregate_breast_predictions(left+right)), 4)

    def test_missing_mlo_rejected(self):
        with self.assertRaisesRegex(ValueError,"one CC and one MLO"):
            aggregate_breast_predictions([r for r in set_of_views() if r.view=="CC"])

    def test_duplicate_view_rejected(self):
        rows=set_of_views()
        rows.append(replace(rows[0], image_id="other"))
        with self.assertRaises(ValueError):
            aggregate_breast_predictions(rows)

    def test_duplicate_prediction_rejected(self):
        rows = set_of_views()
        with self.assertRaisesRegex(ValueError,"Duplicate image prediction"):
            aggregate_breast_predictions(rows+[rows[0]])

    def test_missing_clean_rejected(self):
        with self.assertRaisesRegex(ValueError,"Missing clean baseline"):
            aggregate_breast_predictions(set_of_views()[2:])

    def test_different_degraded_views_rejected(self):
        rows=set_of_views()
        rows[2]=replace(rows[2],image_id="replacement-cc")
        with self.assertRaisesRegex(ValueError,"Clean and degraded image views differ"):
            aggregate_breast_predictions(rows)

    def test_cross_model_views_must_match(self):
        a=set_of_views()
        b=[replace(r,model="large",image_id=("other-"+r.image_id if r.view=="MLO" else r.image_id)) for r in a]
        with self.assertRaisesRegex(ValueError,"Cross-model image views differ"):
            aggregate_breast_predictions(a+b)

    def test_cross_seed_views_must_match(self):
        a=set_of_views()
        b=[replace(r,seed=2,image_id="alt-"+r.image_id) for r in a]
        with self.assertRaisesRegex(ValueError,"Cross-model image views differ"):
            aggregate_breast_predictions(a+b)

    def test_inconsistent_breast_labels_rejected(self):
        a=set_of_views()
        a[1]=replace(a[1],label=0)
        with self.assertRaisesRegex(ValueError,"Conflicting breast labels"):
            aggregate_breast_predictions(a)

    def test_image_id_ownership_rejected(self):
        a=set_of_views()
        a[1]=replace(a[1],image_id=a[0].image_id)
        with self.assertRaisesRegex(ValueError,"Inconsistent image_id ownership"):
            aggregate_breast_predictions(a)

    def test_inconsistent_case_ownership_rejected(self):
        a=set_of_views()
        b=set_of_views(patient="p2")
        b=[replace(r,image_id="p2-"+r.image_id) for r in b]
        with self.assertRaisesRegex(ValueError,"case_id spans different patients"):
            aggregate_breast_predictions(a+b)

    def test_invalid_score_rejected(self):
        for score in [-0.1, 1.1, math.nan, math.inf, True, "0.2"]:
            with self.subTest(score=score), self.assertRaises(ValueError):
                aggregate_breast_predictions([replace(r,score=score) for r in set_of_views()])

    def test_invalid_label_rejected(self):
        for label in [2, -1, True, 1.0]:
            with self.subTest(label=label), self.assertRaises(ValueError):
                aggregate_breast_predictions([replace(r,label=label) for r in set_of_views()])

    def test_invalid_condition_rejected(self):
        rows=set_of_views()
        rows[0]=replace(rows[0],severity="medium")
        with self.assertRaises(ValueError):
            aggregate_breast_predictions(rows)

    def test_invalid_view_rejected(self):
        rows=set_of_views()
        rows[0]=replace(rows[0],view="LCC")
        with self.assertRaisesRegex(ValueError,"view must"):
            aggregate_breast_predictions(rows)

    def test_missing_expected_conditions_rejected(self):
        with self.assertRaisesRegex(ValueError,"missing conditions"):
            aggregate_breast_predictions(set_of_views(), expected_conditions=[("noise","low"),("blur","high")])

    def test_no_file_after_validation_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"images.csv"
            with self.assertRaises(ValueError):
                write_image_csv(set_of_views()[:1],path)
            self.assertFalse(path.exists())

    def test_breast_export_duplicate_rejected(self):
        x=aggregate_breast_predictions(set_of_views())
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError,"Duplicate breast prediction"):
                write_breast_csv(x+[x[0]],Path(temp)/"out.csv")

    def test_breast_export_missing_clean_rejected(self):
        x=aggregate_breast_predictions(set_of_views())
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError,"Missing clean baseline"):
                write_breast_csv(x[1:],Path(temp)/"out.csv")

    def test_stable_order(self):
        rows=set_of_views()
        with tempfile.TemporaryDirectory() as temp:
            a,b=Path(temp)/"a.csv",Path(temp)/"b.csv"
            write_image_csv(rows,a)
            write_image_csv(list(reversed(rows)),b)
            self.assertEqual(a.read_bytes(),b.read_bytes())
            breast=aggregate_breast_predictions(rows)
            write_breast_csv(breast,a)
            write_breast_csv(list(reversed(breast)),b)
            self.assertEqual(a.read_bytes(),b.read_bytes())

    def test_numpy_scalars_if_available(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy not installed")
        rows=[replace(r,score=np.float32(r.score),seed=np.int64(r.seed)) for r in set_of_views()]
        self.assertEqual(len(aggregate_breast_predictions(rows)),2)

    def test_split_summary_counts(self):
        rows=[SplitImage("p1","L","i1","train"),
              SplitImage("p1","L","i2","train"),
              SplitImage("p1","R","i3","train"),
              SplitImage("p2","L","i4","validation"),
              SplitImage("p3","R","i5","test")]
        out=summarize_splits(rows)
        self.assertEqual(out["train"],{"patients":1,"breasts":2,"images":3})
        self.assertEqual(out["test"]["patients"],1)
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/"summary.json"
            write_split_summary_json(rows,p)
            data=json.loads(p.read_text())
            self.assertTrue(data["patient_disjoint"])
            self.assertEqual(data["counts"]["train"]["images"],3)

    def test_patient_leakage_rejected(self):
        rows=[SplitImage("p1","L","i1","train"),SplitImage("p1","R","i2","test")]
        with self.assertRaisesRegex(ValueError,"Patient appears in multiple splits"):
            summarize_splits(rows)

    def test_split_image_assignment_rejected(self):
        rows=[SplitImage("p1","L","i1","train"),SplitImage("p2","L","i1","validation")]
        with self.assertRaises(ValueError):
            summarize_splits(rows)

    def test_no_partial_pair_for_missing_destination_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / "images.csv"
            breast_path = Path(temp) / "missing" / "breasts.csv"
            with self.assertRaises(FileNotFoundError):
                export_prediction_tables(set_of_views(), image_path, breast_path)
            self.assertFalse(image_path.exists())

    def test_export_does_not_leave_temporary_files(self):
        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / "images.csv"
            breast_path = Path(temp) / "breasts.csv"
            export_prediction_tables(set_of_views(), image_path, breast_path)
            self.assertEqual(sorted(p.name for p in Path(temp).iterdir()),
                             ["breasts.csv", "images.csv"])

    def test_same_path_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/"p.csv"
            with self.assertRaisesRegex(ValueError,"different paths"):
                export_prediction_tables(set_of_views(),p,p)


if __name__ == "__main__":
    unittest.main()
