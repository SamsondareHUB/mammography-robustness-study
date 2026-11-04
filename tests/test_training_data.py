"""Validation tests for image manifests and patient-level split files."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from mammography_robustness.training_data import (
    ImageRecord, index_images, read_manifest, read_patient_splits, validate_records,
)


FIELDS = ("patient_id", "case_id", "breast_side", "view", "label", "split")


def records():
    result = []
    for patient, split in (("1", "train"), ("2", "val"), ("3", "test")):
        for view in ("CC", "MLO"):
            result.append(ImageRecord(patient, f"P{patient}_L_DM_{view}", "L", view,
                                      int(patient == "1"), split))
    return result


def splits():
    return {"train": {"1"}, "val": {"2"}, "test": {"3"}}


class TrainingManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write_manifest(self, rows):
        path = self.root / "manifest.csv"
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow(row.__dict__)
        return path

    def write_splits(self, contents=None):
        path = self.root / "splits.json"
        payload = contents if contents is not None else {k: sorted(v) for k, v in splits().items()}
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_manifest_and_patient_counts(self):
        loaded = read_manifest(self.write_manifest(records()))
        summary = validate_records(loaded, read_patient_splits(self.write_splits()))
        self.assertEqual(summary.image_counts, {"train": 2, "val": 2, "test": 2})
        self.assertEqual(summary.patient_counts, {"train": 1, "val": 1, "test": 1})
        self.assertEqual(summary.breast_counts, {"train": 1, "val": 1, "test": 1})
        self.assertEqual(summary.incomplete_breasts, 0)
        self.assertEqual(summary.conflicting_label_breasts, 0)

    def test_csv_and_json_disagreement(self):
        wrong = {"train": ["2"], "val": ["1"], "test": ["3"]}
        with self.assertRaisesRegex(ValueError, "2 patient assignments"):
            validate_records(records(), read_patient_splits(self.write_splits(wrong)))

    def test_manifest_patient_leakage(self):
        image_rows = records()
        first = image_rows[0]
        image_rows[0] = ImageRecord(first.patient_id, first.case_id, "L", first.view, 1, "val")
        with self.assertRaisesRegex(ValueError, "multiple manifest splits"):
            validate_records(image_rows, splits())

    def test_json_patient_leakage(self):
        with self.assertRaisesRegex(ValueError, "multiple JSON splits"):
            read_patient_splits(self.write_splits({"train": [1], "val": [1, 2], "test": [3]}))

    def test_missing_patient_in_json(self):
        with self.assertRaisesRegex(ValueError, "different patients"):
            validate_records(records(), {"train": {"1"}, "val": {"2"}, "test": set()})

    def test_duplicate_case_id(self):
        image_rows = records()
        image_rows[1] = ImageRecord("1", image_rows[0].case_id, "L", "MLO", 1, "train")
        with self.assertRaisesRegex(ValueError, "Duplicate case_id"):
            read_manifest(self.write_manifest(image_rows))

    def test_reject_unknown_labels(self):
        path = self.write_manifest(records())
        text = path.read_text().replace(",1,train", ",malignant,train", 1)
        path.write_text(text)
        with self.assertRaisesRegex(ValueError, "Invalid binary label"):
            read_manifest(path)

    def test_reject_unknown_split(self):
        path = self.write_manifest(records())
        path.write_text(path.read_text().replace(",val\n", ",validation\n", 1))
        with self.assertRaisesRegex(ValueError, "Invalid split"):
            read_manifest(path)

    def test_reject_duplicate_breast_view(self):
        image_rows = records()
        row = image_rows[1]
        image_rows[1] = ImageRecord(row.patient_id, row.case_id, row.breast_side, "CC", row.label, row.split)
        with self.assertRaisesRegex(ValueError, "Duplicate view"):
            validate_records(image_rows, splits())

    def test_report_conflicting_breast_label_without_relabeling_images(self):
        image_rows = records()
        row = image_rows[1]
        image_rows[1] = ImageRecord(row.patient_id, row.case_id, row.breast_side, row.view, 0, row.split)
        summary = validate_records(image_rows, splits())
        self.assertEqual(summary.conflicting_label_breasts, 1)
        self.assertEqual([r.label for r in image_rows[:2]], [1, 0])

    def test_count_incomplete_view_pairs(self):
        summary = validate_records(records()[:-1], splits())
        self.assertEqual(summary.incomplete_breasts, 1)

    def test_image_files_resolved_by_stem(self):
        imgs = self.root / "images" / "nested"
        imgs.mkdir(parents=True)
        for row in records():
            (imgs / f"{row.case_id}.jpg").write_bytes(b"fixture")
        index = index_images(self.root / "images", records())
        self.assertEqual(len(index), 6)

    def test_missing_image_is_rejected(self):
        self.assertRaisesRegex(ValueError, "Missing 6 image files", index_images,
                               self.root, records())

    def test_duplicate_image_stems_are_rejected(self):
        (self.root / "P1_L_DM_CC.jpg").write_bytes(b"fixture")
        (self.root / "P1_L_DM_CC.png").write_bytes(b"fixture")
        with self.assertRaisesRegex(ValueError, "Multiple image files"):
            index_images(self.root, records())

    def test_reject_case_id_with_path_separator(self):
        image_rows = records()
        first = image_rows[0]
        image_rows[0] = ImageRecord(first.patient_id, "../outside", first.breast_side,
                                    first.view, first.label, first.split)
        with self.assertRaisesRegex(ValueError, "Invalid case_id"):
            read_manifest(self.write_manifest(image_rows))

    def test_bad_json_type_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "array of patient IDs"):
            read_patient_splits(self.write_splits({"train": "1", "val": [], "test": []}))


if __name__ == "__main__":
    unittest.main()
