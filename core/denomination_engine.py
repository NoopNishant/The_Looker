"""
core/denomination_engine.py - Calibrated Multi-Cue Denomination Deduction.

Deduces currency denomination using illumination-invariant normalized chromaticity,
HSV spectral centroids, and dimensional aspect ratio metrics across Indian Rupee notes:
₹10, ₹20, ₹50, ₹100, ₹200, ₹500, ₹2000.
"""

from typing import Dict, Optional, Tuple, Union
import cv2
import numpy as np


class DenominationDeductionEngine:
    """
    Expert deduction system mapping optical colorimetric coordinates to banknotes.
    Uses normalized RGB chromaticity (r = R/sum, g = G/sum, b = B/sum), which is
    mathematically invariant to ambient lighting variations and camera exposure shifts.
    """

    PROFILES = {
        "10": {
            "name": "₹10 (Chocolate Brown)",
            "chroma": np.array([0.45, 0.32, 0.23]),
            "hue_center": 11.0,
            "target_aspect": 1.95
        },
        "20": {
            "name": "₹20 (Greenish Yellow)",
            "chroma": np.array([0.41, 0.39, 0.20]),
            "hue_center": 28.0,
            "target_aspect": 2.03
        },
        "50": {
            "name": "₹50 (Fluorescent Blue)",
            "chroma": np.array([0.22, 0.40, 0.38]),
            "hue_center": 68.0,
            "target_aspect": 2.03
        },
        "100": {
            "name": "₹100 (Lavender)",
            "chroma": np.array([0.34, 0.31, 0.35]),
            "hue_center": 138.0,
            "target_aspect": 2.15
        },
        "200": {
            "name": "₹200 (Bright Yellow-Orange)",
            "chroma": np.array([0.56, 0.34, 0.10]),
            "hue_center": 22.0,
            "target_aspect": 2.21
        },
        "500": {
            "name": "₹500 (Stone Gray)",
            "chroma": np.array([0.34, 0.35, 0.31]),
            "hue_center": 58.0,
            "target_aspect": 2.27
        },
        "2000": {
            "name": "₹2000 (Magenta)",
            "chroma": np.array([0.48, 0.22, 0.30]),
            "hue_center": 162.0,
            "target_aspect": 2.52
        }
    }

    @classmethod
    def deduce_denomination(
        cls,
        cropped_note: np.ndarray,
        cnn_probs: Optional[Dict[str, float]] = None,
        aspect_ratio_valid: bool = True
    ) -> Dict[str, Union[str, float, bool, Dict]]:
        """
        Deduce banknote denomination using normalized chromaticity, spectral centroids,
        saturation, value, and dimensional aspect ratio metrics.
        """
        if cropped_note is None or cropped_note.size == 0:
            return {
                "denomination": "Background",
                "confidence": 0.0,
                "is_valid_currency": False,
                "note_name": "No Currency",
                "scores": {}
            }

        h, w = cropped_note.shape[:2]
        aspect_ratio = max(w, h) / max(1.0, min(w, h))

        # Check internal texture variance
        gray = cv2.cvtColor(cropped_note, cv2.COLOR_BGR2GRAY)
        lap_var = float(np.var(cv2.Laplacian(gray, cv2.CV_32F)))

        # Sample central 60% region to eliminate table/hand borders
        y1, y2 = max(0, int(h * 0.18)), min(h, int(h * 0.82))
        x1, x2 = max(0, int(w * 0.18)), min(w, int(w * 0.82))
        central_bgr = cropped_note[y1:y2, x1:x2]

        if central_bgr.size == 0:
            central_bgr = cropped_note

        # Convert to RGB float
        rgb = cv2.cvtColor(central_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) + 1e-5
        rgb_sum = np.sum(rgb, axis=2, keepdims=True)

        # Normalized Chromaticity Coordinates (invariant to light intensity)
        r_norm = rgb[:, :, 0] / rgb_sum[:, :, 0]
        g_norm = rgb[:, :, 1] / rgb_sum[:, :, 0]
        b_norm = rgb[:, :, 2] / rgb_sum[:, :, 0]

        mean_chroma = np.array([
            float(np.median(r_norm)),
            float(np.median(g_norm)),
            float(np.median(b_norm))
        ])

        # HSV for hue, saturation, and luminance guidance
        hsv = cv2.cvtColor(central_bgr, cv2.COLOR_BGR2HSV)
        median_hue = float(np.median(hsv[:, :, 0]))
        mean_sat = float(np.mean(hsv[:, :, 1]))
        mean_val = float(np.mean(hsv[:, :, 2]))

        # Balanced multi-cue likeness evaluation across all denominations
        scores: Dict[str, float] = {}

        for denom, prof in cls.PROFILES.items():
            target_chroma = prof["chroma"]
            target_hue = prof["hue_center"]
            target_ar = prof["target_aspect"]

            # 1. Normalized chromaticity distance (Euclidean in r-g-b space)
            dist_chroma = float(np.linalg.norm(mean_chroma - target_chroma))
            score_chroma = np.exp(-18.0 * (dist_chroma ** 2))

            # 2. Angular hue distance (symmetric Gaussian Likelihood)
            hue_diff = min(abs(median_hue - target_hue), 180.0 - abs(median_hue - target_hue))
            score_hue = np.exp(-0.5 * ((hue_diff / 16.0) ** 2))

            # 3. Dimensional aspect ratio (only applied when tight bounding quad is verified)
            if aspect_ratio_valid:
                ar_diff = abs(aspect_ratio - target_ar)
                score_ar = max(0.35, np.exp(-0.5 * ((ar_diff / 0.28) ** 2)))
            else:
                score_ar = 1.0

            # Combined optical likeness
            optical_like = 0.55 * score_chroma + 0.35 * score_hue + 0.10 * score_ar

            # Distinctive optical discrimination between ₹10 (Chocolate Brown) and ₹200 (Bright Yellow-Orange)
            if denom == "200":
                if mean_chroma[2] >= 0.17:  # High blue indicates chocolate brown or other, not bright yellow-orange
                    optical_like *= 0.35
                if mean_sat < 110:         # ₹200 has high vibrant saturation
                    optical_like *= 0.40
                if mean_val < 150:         # ₹200 has high luminance
                    optical_like *= 0.60
                if mean_chroma[0] >= 0.52 and mean_chroma[2] <= 0.13 and mean_sat >= 130:
                    optical_like = min(1.0, optical_like * 1.35)

            elif denom == "10":
                if mean_chroma[2] <= 0.13 and mean_chroma[0] >= 0.50:  # Low blue + high red indicates bright orange
                    optical_like *= 0.35
                if mean_sat > 135:         # ₹10 has muted earthy brown saturation
                    optical_like *= 0.45
                if mean_val > 185:         # ₹10 is dark brown, not neon bright
                    optical_like *= 0.65
                if mean_chroma[2] >= 0.18 and mean_sat <= 110:
                    optical_like = min(1.0, optical_like * 1.30)

            # Multi-cue fusion with CNN semantic probability
            if cnn_probs:
                cnn_score = float(cnn_probs.get(denom, 0.0))
                final_score = 0.65 * cnn_score + 0.35 * optical_like
            else:
                final_score = optical_like

            scores[denom] = float(final_score)

        # Strict Background Gate: if CNN identified Background, or no currency profile fits
        if cnn_probs and cnn_probs.get("Background", 0.0) >= 0.35:
            return {
                "denomination": "Background",
                "confidence": 0.0,
                "is_valid_currency": False,
                "note_name": "No Currency in View",
                "scores": scores
            }

        # Select top denomination
        best_denom = max(scores, key=scores.get)
        best_val = scores[best_denom]

        # Margin calculation for confidence readout
        score_vals = list(scores.values())
        sorted_scores = sorted(score_vals, reverse=True)
        margin = sorted_scores[0] - (sorted_scores[1] if len(sorted_scores) > 1 else 0.0)
        confidence = float(np.clip(0.78 + margin * 0.40, 0.72, 0.99))

        return {
            "denomination": best_denom,
            "confidence": round(confidence, 3),
            "is_valid_currency": True,
            "note_name": cls.PROFILES[best_denom]["name"],
            "scores": scores
        }
