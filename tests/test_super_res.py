"""
tests/test_super_res.py - Tests for ESPCN Super-Resolution and CMYK Halftone Detector.
"""

import unittest
import numpy as np
import cv2
import torch

from core.super_res_halftone import (
    ESPCNSubPixelNet,
    SuperResolutionModule,
    HalftoneDotDetector
)


class TestSuperResHalftone(unittest.TestCase):

    def setUp(self):
        self.device = "cpu"
        self.sr_module = SuperResolutionModule(scale_factor=2, device=self.device)
        self.detector = HalftoneDotDetector()

    def test_espcn_network_tensor_shapes(self):
        """Verify ESPCN forward pass upscaling factor."""
        net = ESPCNSubPixelNet(scale_factor=2, in_channels=1)
        net.eval()
        x = torch.randn(2, 1, 32, 32)
        with torch.inference_mode():
            out = net(x)
        self.assertEqual(out.shape, (2, 1, 64, 64))

    def test_super_resolution_roi_upscaling(self):
        """Verify BGR ROI upscaling produces 2x magnified resolution."""
        roi = np.random.randint(0, 256, (48, 48, 3), dtype=np.uint8)
        sr_roi = self.sr_module.upscale_roi(roi)

        self.assertEqual(sr_roi.shape, (96, 96, 3))
        self.assertEqual(sr_roi.dtype, np.uint8)

    def test_cmyk_decomposition(self):
        """Verify CMYK separation channels."""
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        # Red patch in BGR
        img[:, :] = (0, 0, 255)
        cmyk = self.detector.rgb_to_cmyk(img)

        self.assertIn("C", cmyk)
        self.assertIn("M", cmyk)
        self.assertIn("Y", cmyk)
        self.assertIn("K", cmyk)

        # Pure red in CMYK has zero cyan, full magenta and yellow
        self.assertEqual(cmyk["C"].shape, (64, 64))
        self.assertLess(np.mean(cmyk["C"]), 5.0)
        self.assertGreater(np.mean(cmyk["M"]), 200.0)
        self.assertGreater(np.mean(cmyk["Y"]), 200.0)

    def test_fft_spectrum_and_halftone_discrimination(self):
        """
        Verify that periodic halftone screens produce high FFT harmonic peak ratios,
        while smooth continuous line patterns do not.
        """
        h, w = 128, 128
        # 1. Authentic Intaglio simulated fine continuous wave
        intaglio_patch = np.full((h, w, 3), 150, dtype=np.uint8)
        for y in range(h):
            intaglio_patch[y, :, :] = int(150 + 30 * np.sin(y * 0.2))

        # 2. Counterfeit Halftone simulated periodic dot screen (15 degree screen angle)
        halftone_patch = np.full((h, w, 3), 150, dtype=np.uint8)
        y_c, x_c = np.ogrid[:h, :w]
        theta = np.radians(15.0)
        u = x_c * np.cos(theta) + y_c * np.sin(theta)
        v = -x_c * np.sin(theta) + y_c * np.cos(theta)
        dots = (np.sin(u * 0.75) * np.sin(v * 0.75)) > 0.4
        halftone_patch[dots, 0] = 235
        halftone_patch[dots, 1] = 210
        halftone_patch[dots, 2] = 20

        # Analyze both patches
        res_intaglio = self.detector.analyze_microtexture(intaglio_patch)
        res_halftone = self.detector.analyze_microtexture(halftone_patch)

        # Verification asserts
        self.assertIn("halftone_score", res_halftone)
        self.assertIn("peak_ratio", res_halftone)
        self.assertIn("vis_spectrum", res_halftone)

        # Counterfeit halftone must trigger higher halftone score and peak prominence
        self.assertGreater(res_halftone["halftone_score"], res_intaglio["halftone_score"])
        self.assertTrue(res_halftone["is_halftone_detected"])


if __name__ == "__main__":
    unittest.main()

