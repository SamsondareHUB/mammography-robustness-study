"""Shared training and inference preprocessing regression tests."""
import unittest


class PreprocessingTests(unittest.TestCase):
    def test_eval_is_repeatable_and_normalized(self):
        from PIL import Image
        from mammography_robustness.preprocessing import make_transform
        img = Image.new("L", (50, 100), color=127)
        f = make_transform(64, training=False)
        a, b = f(img), f(img)
        self.assertEqual(tuple(a.shape), (3, 64, 64))
        self.assertTrue(a.equal(b))
        self.assertNotAlmostEqual(float(a[0, 32, 32]), 127 / 255)

    def test_legacy_checkpoint_uses_plain_tensor(self):
        from PIL import Image
        from mammography_robustness.preprocessing import make_transform
        img = Image.new("RGB", (64, 64), color=(255, 0, 0))
        v = make_transform(64, normalization="none")(img)
        self.assertEqual(float(v[0, 20, 20]), 1.0)
        self.assertEqual(float(v[1, 20, 20]), 0.0)

    def test_training_augmentation_can_be_disabled(self):
        from PIL import Image
        from mammography_robustness.preprocessing import make_transform
        img = Image.new("RGB", (64, 64), color=(140, 20, 90))
        aug = make_transform(64, training=True, flip_probability=0, rotation_degrees=0)
        evaluation = make_transform(64, training=False)
        self.assertTrue(aug(img).equal(evaluation(img)))


if __name__ == "__main__":
    unittest.main()
