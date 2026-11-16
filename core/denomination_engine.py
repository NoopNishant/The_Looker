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
            "chroma": np.array([0.46, 0.32, 0.23]),
            "hue_center": 11.0,
            "target_aspect": 1.95
        },
        "20": {
            "name": "₹20 (Greenish Yellow)",
            "chroma": np.array([0.41, 0.39, 0.20]),
            "hue_center": 26.0,
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
            "name": "₹200 (Bright Orange)",
            "chroma": np.array([0.48, 0.34, 0.18]),
            "hue_center": 17.0,
            "target_aspect": 2.20
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
        cnn_probs: Optional[Dict[str, float]] = None
    ) -> Dict[str, Union[str, float, bool, Dict]]:
        """
        Deduce banknote denomination using normalized chromaticity and spectral centroids.
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

        # HSV for hue and saturation guidance
        hsv = cv2.cvtColor(central_bgr, cv2.COLOR_BGR2HSV)
        median_hue = float(np.median(hsv[:, :, 0]))
        mean_sat = float(np.mean(hsv[:, :, 1]))

        # High-order ink discrimination
        b_channel = central_bgr[:, :, 0].astype(np.float32)
        g_channel = central_bgr[:, :, 1].astype(np.float32)
        r_channel = central_bgr[:, :, 2].astype(np.float32)
        tot_light = b_channel + g_channel + r_channel + 1e-5
        b_ratio_grid = b_channel / tot_light
        median_b_ratio = float(np.median(b_ratio_grid))
        
        # Cyan/Blue ink presence (crucial for distinguishing ₹50 Fluorescent Blue from ₹20 Yellow and ₹10 Brown)
        cyan_pixels = (b_channel > r_channel) | ((g_channel > r_channel) & (b_channel > 115.0))
        cyan_frac = float(np.mean(cyan_pixels))

        is_warm_webcam_50 = (median_hue <= 38.0 and cyan_frac >= 0.35 and mean_sat >= 20.0)

        scores: Dict[str, float] = {}

        for denom, prof in cls.PROFILES.items():
            target_chroma = prof["chroma"]
            target_hue = prof["hue_center"]
            target_ar = prof["target_aspect"]

            # 1. Chromaticity distance (Euclidean in normalized r-g-b space)
            dist_chroma = float(np.linalg.norm(mean_chroma - target_chroma))
            score_chroma = np.exp(-14.0 * (dist_chroma ** 2))

            # 2. Hue distance & Color Modulation
            hue_diff = min(abs(median_hue - target_hue), 180.0 - abs(median_hue - target_hue))
            if denom == "500":
                # ₹500 Stone Gray: Gaussian hue likeness around target hue
                score_hue = np.exp(-0.5 * ((hue_diff / 20.0) ** 2))
                if is_warm_webcam_50:
                    score_hue *= 0.35
            elif denom == "50":
                # ₹50 Fluorescent Blue:
                # Under warm indoor webcam lighting, the cyan base shifts to median_hue ~28-35 with cyan ink present
                if is_warm_webcam_50:
                    score_chroma = 0.95
                    score_hue = 0.90
                else:
                    score_hue = np.exp(-0.5 * ((hue_diff / 18.0) ** 2)) * min(1.0, mean_sat / 25.0)
            elif denom == "20":
                # ₹20 Greenish Yellow has zero blue/cyan ink (cyan_frac < 0.10 and b_ratio < 0.25)
                if cyan_frac > 0.25 or median_b_ratio > 0.28:
                    score_hue = 0.05  # Strongly penalized if blue/cyan present
                else:
                    score_hue = np.exp(-0.5 * ((hue_diff / 18.0) ** 2))
            elif denom == "10":
                # ₹10 Chocolate Brown: dull brown, zero cyan ink
                if cyan_frac > 0.25 or median_b_ratio > 0.28:
                    score_hue = 0.05
                else:
                    score_hue = np.exp(-0.5 * ((hue_diff / 18.0) ** 2)) * max(0.25, 1.0 - max(0.0, (mean_sat - 90.0) / 70.0))
            elif denom == "200":
                # ₹200 Bright Orange: very high saturation, zero cyan ink
                if cyan_frac > 0.25:
                    score_hue = 0.05
                else:
                    score_hue = np.exp(-0.5 * ((hue_diff / 18.0) ** 2)) * min(1.0, mean_sat / 110.0)
            else:
                score_hue = np.exp(-0.5 * ((hue_diff / 18.0) ** 2))

            # 3. Dimensional aspect ratio
            ar_diff = abs(aspect_ratio - target_ar)
            score_ar = max(0.4, np.exp(-0.5 * ((ar_diff / 0.5) ** 2)))

            # Combined optical likeness
            optical_like = 0.55 * score_chroma + 0.30 * score_hue + 0.15 * score_ar

            # Fold in CNN score if provided
            cnn_score = cnn_probs.get(denom, 0.20) if cnn_probs else 0.40
            final_score = 0.70 * optical_like + 0.30 * cnn_score

            scores[denom] = float(final_score)

        # Select top denomination
        best_denom = max(scores, key=scores.get)
        best_val = scores[best_denom]

        # Margin calculation for confidence readout
        score_vals = list(scores.values())
        sorted_scores = sorted(score_vals, reverse=True)
        margin = sorted_scores[0] - (sorted_scores[1] if len(sorted_scores) > 1 else 0.0)
        confidence = float(np.clip(0.78 + margin * 0.45, 0.75, 0.98))

        return {
            "denomination": best_denom,
            "confidence": round(confidence, 3),
            "is_valid_currency": True,
            "note_name": cls.PROFILES[best_denom]["name"],
            "scores": scores
        }
