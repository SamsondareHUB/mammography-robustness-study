"""Focused tests for approved breast-level checkpoint selection."""
import unittest

from mammography_robustness.training_protocol import (
    breast_auc, inverse_frequency_weights, should_replace_best,
)


class TrainingProtocolTests(unittest.TestCase):
    def test_auc_aggregates_max_available_views(self):
        self.assertAlmostEqual(
            breast_auc(["1", "1", "2", "2", "3"], ["L", "L", "R", "R", "L"],
                       [0, 0, 1, 1, 1], [.2, .7, .4, .8, .6]), 0.5,
        )

    def test_auc_with_tied_scores(self):
        self.assertAlmostEqual(
            breast_auc(["1", "2", "3", "4"], ["L"] * 4,
                       [0, 0, 1, 1], [.5] * 4), 0.5,
        )

    def test_single_class_not_silently_accepted(self):
        with self.assertRaisesRegex(ValueError, "both classes"):
            breast_auc(["1", "2"], ["L", "L"], [1, 1], [.4, .6])

    def test_checkpoint_tie_breaking(self):
        self.assertTrue(should_replace_best(.8, .4, 3, (.7, .3, 2)))
        self.assertTrue(should_replace_best(.8, .2, 3, (.8, .3, 2)))
        self.assertFalse(should_replace_best(.8, .3, 3, (.8, .3, 2)))
        self.assertFalse(should_replace_best(.7, .1, 3, (.8, .3, 2)))

    def test_weights_training_only(self):
        self.assertEqual(inverse_frequency_weights([0, 0, 0, 1]), (2 / 3, 2))
        with self.assertRaises(ValueError):
            inverse_frequency_weights([0, 0])


if __name__ == "__main__":
    unittest.main()
