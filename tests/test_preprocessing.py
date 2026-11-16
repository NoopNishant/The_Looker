"""
tests/test_preprocessing.py - Tests for CLAHE enhancement and Screen-Flash Differencing.
"""

import unittest
import numpy as np
import cv2

from core.preprocessing import CLAHEEnhancer, ScreenFlashDifferencing


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.clahe = CLAHEEnhancer(clip_limit=2.5, tile_grid_size=(8, 8))
        self.differencer = ScreenFlashDifferencing()

    def test_clahe_bgr_enhancement(self):
        """Test CLAHE contrast enhancement on BGR color image."""
        # Generate low contrast synthetic texture
        low_contrast = np.full((128, 128, 3), 120, dtype=np.int32)
        # Add subtle wave pattern
        for y in range(128):
            low_contrast[y, :, 0] += int(10 * np.sin(y * 0.1))
            low_contrast[y, :, 1] += int(10 * np.cos(y * 0.1))
        low_contrast = np.clip(low_contrast, 0, 255).astype(np.uint8)

        enhanced = self.clahe.enhance_bgr(low_contrast)

        self.assertEqual(enhanced.shape, low_contrast.shape)
        self.assertEqual(enhanced.dtype, np.uint8)
        # CLAHE should increase standard deviation (contrast)
        std_orig = np.std(cv2.cvtColor(low_contrast, cv2.COLOR_BGR2GRAY))
        std_enh = np.std(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY))
        self.assertGreater(std_enh, std_orig)

    def test_clahe_gray_and_topology(self):
        """Test grayscale enhancement and topological gradient generation."""
        img = np.random.randint(60, 180, (120, 120, 3), dtype=np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        enh_gray = self.clahe.enhance_gray(gray)
        self.assertEqual(enh_gray.shape, (120, 120))
        self.assertEqual(enh_gray.dtype, np.uint8)

        topo = self.clahe.extract_topological_gradients(img)
        self.assertEqual(topo.shape, (120, 120))
        self.assertEqual(topo.dtype, np.uint8)
        self.assertGreater(np.max(topo), 0)

    def test_screen_flash_simulation_and_difference(self):
        """Test Screen-Flash Differencing module on simulated watermark."""
        base_frame = np.full((240, 360, 3), 100, dtype=np.uint8)
        ambient, flash = ScreenFlashDifferencing.simulate_screen_flash(base_frame)

        self.assertEqual(ambient.shape, base_frame.shape)
        self.assertEqual(flash.shape, base_frame.shape)

        # Flash must have higher luminance due to front-light screen reflection
        self.assertGreater(np.mean(flash), np.mean(ambient))

        # Compute difference
        diff_results = self.differencer.compute_difference(ambient, flash)

        self.assertIn("delta_bgr", diff_results)
        self.assertIn("delta_gray", diff_results)
        self.assertIn("delta_enhanced", diff_results)
        self.assertIn("watermark_mask", diff_results)
        self.assertIn("transmission_density", diff_results)

        # Delta shape matches
        self.assertEqual(diff_results["delta_bgr"].shape, base_frame.shape)
        self.assertEqual(diff_results["watermark_mask"].shape, (240, 360))

        # Transmission density should detect simulated watermark region
        density = diff_results["transmission_density"]
        self.assertGreater(density, 0.0)
        self.assertLess(density, 1.0)


if __name__ == "__main__":
    unittest.main()
