"""
tests/test_web_api.py - Verification tests for FastAPI web endpoints and serializers.
"""

import os
import sys
import unittest

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from fastapi.testclient import TestClient

from web.app import app, image_to_base64
from synthetic_data import SyntheticCurrencyGenerator


class TestWebAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_get_root_serves_html(self):
        """Verify GET / returns 200 and loads the minimalist studio interface."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("The Looker", resp.text)
        self.assertIn("Currency Forensics", resp.text)

    def test_generate_demo_genuine(self):
        """Verify synthetic demo endpoint for authentic note."""
        resp = self.client.post("/api/generate-demo", json={"denomination": "500", "is_counterfeit": False})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertIn("verdict", data)
        self.assertIn("composite_counterfeit_score", data)
        self.assertIn("plates", data)
        self.assertIn("sample_image", data)

        self.assertEqual(data["verdict"], "GENUINE")
        self.assertLess(data["composite_counterfeit_score"], 0.50)
        self.assertTrue(data["sample_image"].startswith("data:image/"))

    def test_generate_demo_counterfeit(self):
        """Verify synthetic demo endpoint for counterfeit note."""
        resp = self.client.post("/api/generate-demo", json={"denomination": "500", "is_counterfeit": True})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertEqual(data["verdict"], "SUSPECT / COUNTERFEIT")
        self.assertGreaterEqual(data["composite_counterfeit_score"], 0.50)
        self.assertEqual(data["audio_tag"], "COUNTERFEIT_ALARM")

    def test_inspect_uploaded_photo(self):
        """Verify POST /api/inspect/image endpoint with binary payload."""
        test_img = SyntheticCurrencyGenerator.generate_banknote("500", is_counterfeit=False, width=640, height=320)
        _, buffer = cv2.imencode(".png", test_img)
        file_bytes = buffer.tobytes()

        resp = self.client.post(
            "/api/inspect/image",
            content=file_bytes,
            headers={"Content-Type": "image/png", "X-Filename": "test_500.png"}
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertIn("verdict", data)
        self.assertIn("plates", data)
        self.assertIn("clahe", data["plates"])
        self.assertIn("super_res", data["plates"])
        self.assertIn("fft_spectrum", data["plates"])

    def test_flash_differencing_endpoint(self):
        """Verify POST /api/flash-diff for paired client frame analysis."""
        base_img = np.full((120, 160, 3), 100, dtype=np.uint8)
        amb_b64 = image_to_base64(base_img)
        # Create brighter frame with watermark oval
        flash_img = base_img.copy() + 40
        cv2.circle(flash_img, (80, 60), 20, (255, 255, 255), -1)
        flash_b64 = image_to_base64(flash_img)

        resp = self.client.post(
            "/api/flash-diff",
            json={"ambient_base64": amb_b64, "flash_base64": flash_b64}
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertIn("transmission_density", data)
        self.assertIn("watermark_mask", data)
        self.assertGreater(data["transmission_density"], 0.0)


if __name__ == "__main__":
    unittest.main()
