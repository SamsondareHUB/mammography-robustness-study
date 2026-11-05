"""Regression tests for approved repeated-MLO selection in CDD-CESM."""

import unittest
from dataclasses import replace

from mammography_robustness.training_data import (
    ImageRecord, select_evaluation_images, validate_records,
)


def example_records():
    rows = [
        ImageRecord("99", "P99_L_DM_CC", "L", "CC", 0, "train"),
        ImageRecord("99", "P99_L_DM_MLO", "L", "MLO", 0, "train"),
        ImageRecord("99", "P99_L_DM_MLO_2", "L", "MLO", 0, "train"),
        ImageRecord("281", "P281_L_DM_CC", "L", "CC", 0, "val"),
        ImageRecord("281", "P281_L_DM_MLO", "L", "MLO", 0, "val"),
        ImageRecord("281", "P281_L_DM_MLO2", "L", "MLO", 0, "val"),
        ImageRecord("16", "P16_R_DM_CC", "R", "CC", 0, "test"),
        ImageRecord("16", "P16_R_DM_MLO", "R", "MLO", 1, "test"),
    ]
    splits = {"train": {"99"}, "val": {"281"}, "test": {"16"}}
    return rows, splits


class ApprovedAcquisitionTests(unittest.TestCase):
    def test_known_extra_mlos_are_valid_with_original_split_counts(self):
        rows, splits = example_records()
        summary = validate_records(rows, splits)
        self.assertEqual(summary.image_counts, {"train": 3, "val": 3, "test": 2})
        self.assertEqual(summary.conflicting_label_breasts, 1)
        self.assertEqual(summary.incomplete_breasts, 0)
        self.assertEqual(len(rows), 8)  # Source acquisitions remain unchanged.

    def test_fixed_evaluation_selection_keeps_unsuffixed_images(self):
        rows, _ = example_records()
        chosen = select_evaluation_images(rows)
        ids = {r.case_id for r in chosen}
        self.assertEqual(len(chosen), 6)
        self.assertIn("P99_L_DM_MLO", ids)
        self.assertIn("P281_L_DM_MLO", ids)
        self.assertNotIn("P99_L_DM_MLO_2", ids)
        self.assertNotIn("P281_L_DM_MLO2", ids)
        self.assertEqual(sum(r.split == "val" for r in chosen), 2)
        self.assertEqual(sum(r.split == "test" for r in chosen), 2)

    def test_train_retains_both_patient_99_acquisitions(self):
        rows, _ = example_records()
        training_rows = [r for r in rows if r.split == "train"]
        self.assertEqual(len(training_rows), 3)
        self.assertIn("P99_L_DM_MLO_2", {r.case_id for r in training_rows})
        self.assertNotIn("281", {r.patient_id for r in training_rows})

    def test_unapproved_extra_mlo_is_rejected(self):
        rows, splits = example_records()
        rows.append(ImageRecord("16", "P16_R_DM_MLO2", "R", "MLO", 1, "test"))
        with self.assertRaisesRegex(ValueError, "Duplicate view"):
            validate_records(rows, splits)

    def test_changed_extra_mlo_identifier_is_rejected(self):
        rows, splits = example_records()
        rows[2] = replace(rows[2], case_id="P99_L_DM_MLO3")
        with self.assertRaisesRegex(ValueError, "Duplicate view"):
            validate_records(rows, splits)

    def test_extra_mlo_in_wrong_split_is_rejected(self):
        rows, splits = example_records()
        rows[2] = replace(rows[2], split="val")
        with self.assertRaisesRegex(ValueError, "multiple manifest splits"):
            validate_records(rows, splits)

    def test_extra_mlo_label_mismatch_is_rejected(self):
        rows, splits = example_records()
        rows[2] = replace(rows[2], label=1)
        with self.assertRaisesRegex(ValueError, "Duplicate view"):
            validate_records(rows, splits)

    def test_selected_evaluation_acquisition_must_exist(self):
        rows, _ = example_records()
        rows = [r for r in rows if r.case_id != "P281_L_DM_MLO"]
        with self.assertRaisesRegex(ValueError, "Missing selected evaluation acquisition"):
            select_evaluation_images(rows)

    def test_no_source_mutation(self):
        rows, _ = example_records()
        original = list(rows)
        select_evaluation_images(rows)
        self.assertEqual(rows, original)


if __name__ == "__main__":
    unittest.main()
