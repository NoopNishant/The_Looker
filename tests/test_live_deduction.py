"""
tests/test_live_deduction.py - Unit tests for BanknoteDetector and DenominationDeductionEngine.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2

from core.detector import BanknoteDetector
from core.denomination_engine import DenominationDeductionEngine
from core.pipeline import CurrencyInspectionPipeline
from synthetic_data import SyntheticCurrencyGenerator


class TestLiveDeduction(unittest.TestCase):

    def setUp(self):
        self.detector = BanknoteDetector()
        self.pipeline = CurrencyInspectionPipeline(device="cpu", enable_audio=False)

    def test_banknote_detector_on_synthetic_note(self):
        """Verify BanknoteDetector locates a currency note against a background."""
        bg = np.full((720, 1280, 3), 210, dtype=np.uint8) # Desk background
        note = SyntheticCurrencyGenerator.generate_banknote("500", is_counterfeit=False, width=640, height=320)
        
        # Place note in center of desk
        bg[200:520, 320:960] = note

        detection = self.detector.detect(bg)
        self.assertTrue(detection["detected"])
        self.assertIsNotNone(detection["bbox"])
        x, y, w, h = detection["bbox"]
        
        # Detected box should tightly enclose the note region
        self.assertAlmostEqual(x + w / 2, 640, delta=80)
        self.assertAlmostEqual(y + h / 2, 360, delta=60)
        self.assertGreater(detection["confidence"], 0.60)

    def test_banknote_detector_rejects_empty_frame(self):
        """Verify detector reports detected=False when no currency is present."""
        empty_desk = np.full((480, 640, 3), 180, dtype=np.uint8)
        detection = self.detector.detect(empty_desk)
        self.assertFalse(detection["detected"])

    def test_denomination_deduction_across_denominations(self):
        """Verify DenominationDeductionEngine accurately deduces multiple currency classes."""
        for denom in ["10", "20", "50", "100", "200", "500", "2000"]:
            note = SyntheticCurrencyGenerator.generate_banknote(denom, is_counterfeit=False, width=640, height=320)
            res = DenominationDeductionEngine.deduce_denomination(note)
            
            self.assertTrue(res["is_valid_currency"])
            self.assertEqual(res["denomination"], denom)
            self.assertGreaterEqual(res["confidence"], 0.70)

    def test_full_pipeline_deduction_and_authenticity(self):
        """Verify end-to-end detection, denomination deduction, and real vs counterfeit verdict."""
        # 1. Genuine Note
        genuine_note = SyntheticCurrencyGenerator.generate_banknote("500", is_counterfeit=False, width=640, height=320)
        res_gen = self.pipeline.process_frame(genuine_note)

        self.assertEqual(res_gen["denomination"], "500")
        self.assertEqual(res_gen["verdict"], "GENUINE")
        self.assertFalse(res_gen["is_counterfeit"])
        self.assertIn("Intaglio", res_gen["authenticity_reason"])

        # 2. Counterfeit Note (CMYK Rosettes)
        counterfeit_note = SyntheticCurrencyGenerator.generate_banknote("500", is_counterfeit=True, width=640, height=320)
        res_sus = self.pipeline.process_frame(counterfeit_note)

        self.assertEqual(res_sus["denomination"], "500")
        self.assertEqual(res_sus["verdict"], "SUSPECT / COUNTERFEIT")
        self.assertTrue(res_sus["is_counterfeit"])
        self.assertIn("CMYK", res_sus["authenticity_reason"])


if __name__ == "__main__":
    unittest.main()

