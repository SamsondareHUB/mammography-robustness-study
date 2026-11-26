import hashlib
import unittest

import numpy as np
from PIL import Image
import torch

from mammography_robustness.degradations import (
    CONDITIONS, corrupted_input, content_box, input_hash, model_input, noise_seed,
)
from mammography_robustness.preprocessing import make_transform


class DegradationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        y, x = np.indices((93, 157))
        a = np.uint8((x * 11 + y * 7) % 256)
        self.image = Image.fromarray(a)
        self.case = 'synthetic_case'

    def test_condition_matrix(self):
        self.assertEqual(len(CONDITIONS), 13)
        self.assertEqual(len(set(CONDITIONS)), 13)

    def test_clean_matches_baseline(self):
        for im in (self.image, self.image.transpose(Image.Transpose.TRANSPOSE), self.image.convert('RGB')):
            actual, digest = model_input(im, self.case, 'clean', 'none')
            self.assertTrue(torch.equal(actual, make_transform(224, training=False)(im)))
            self.assertEqual(digest, input_hash(make_transform(224, training=False, normalization='none')(im)))

    def test_all_conditions_padding_range_and_repeatability(self):
        for im in (self.image, self.image.transpose(Image.Transpose.TRANSPOSE), Image.new('L',(31,31),128)):
            l,t,r,b = content_box(im)
            outside = torch.ones((224,224),dtype=torch.bool)
            outside[t:b,l:r] = False
            for condition in CONDITIONS:
                with self.subTest(size=im.size, condition=condition):
                    raw = corrupted_input(im, self.case, *condition)
                    self.assertEqual(tuple(raw.shape),(3,224,224))
                    self.assertEqual(raw.dtype,torch.float32)
                    self.assertTrue(torch.isfinite(raw).all())
                    self.assertTrue(((raw >= 0) & (raw <= 1)).all())
                    self.assertTrue((raw[:,outside] == 0).all())
                    hashes = [model_input(im,self.case,*condition)[1] for _ in range(6)]
                    self.assertEqual(len(set(hashes)),1)

    def test_noise_matches_protocol_and_varies_by_case(self):
        im = Image.new('RGB',(157,93),(128,128,128))
        l,t,r,b = content_box(im)
        seed = int.from_bytes(hashlib.sha256(b'314159:synthetic_case').digest(),'big')
        self.assertEqual(noise_seed(self.case),seed)
        expected = np.random.Generator(np.random.PCG64(seed)).standard_normal((b-t,r-l),dtype=np.float32)
        clean = corrupted_input(im,self.case,'clean','none')[:,t:b,l:r]
        for severity, sd in [('low',.02),('medium',.05),('high',.10)]:
            raw = corrupted_input(im,self.case,'noise',severity)[:,t:b,l:r]
            self.assertTrue(torch.equal(raw,(clean + torch.from_numpy(expected*np.float32(sd))).clamp(0,1)))
            self.assertTrue(torch.equal(raw[0],raw[1]))
            self.assertTrue(torch.equal(raw[1],raw[2]))
        self.assertNotEqual(input_hash(corrupted_input(im,self.case,'noise','low')),input_hash(corrupted_input(im,'another_case','noise','low')))

    def test_contrast_formula(self):
        l,t,r,b=content_box(self.image)
        clean=corrupted_input(self.image,self.case,'clean','none')[:,t:b,l:r]
        mean=clean.mean((1,2),keepdim=True)
        for severity, p in [('low',.8),('medium',.6),('high',.4)]:
            raw=corrupted_input(self.image,self.case,'contrast',severity)[:,t:b,l:r]
            self.assertTrue(torch.equal(raw,(mean+p*(clean-mean)).clamp(0,1)))

    def test_blur_and_resolution_reduce_high_frequency(self):
        im=Image.fromarray(np.uint8((np.indices((224,224)).sum(axis=0)%2)*255))
        clean=corrupted_input(im,self.case,'clean','none')
        def roughness(x): return (x[:,:,1:]-x[:,:,:-1]).abs().mean().item()
        for kind in ('blur','resolution'):
            raw=corrupted_input(im,self.case,kind,'high')
            self.assertLess(roughness(raw),roughness(clean))

    def test_invalid_conditions(self):
        for kind,severity in [('clean','low'),('noise','none'),('unknown','high')]:
            with self.assertRaises(ValueError): corrupted_input(self.image,self.case,kind,severity)
        with self.assertRaises(ValueError): corrupted_input(self.image,'','noise','low')
        with self.assertRaises(ValueError): corrupted_input(self.image,self.case,'noise','low',size=128)

if __name__ == '__main__': unittest.main()
