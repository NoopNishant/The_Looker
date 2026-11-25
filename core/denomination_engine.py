"""
core/denomination_engine.py - Calibrated Multi-Cue Denomination Deduction.

Deduces currency denomination using illumination-invariant normalized chromaticity,
HSV spectral centroids, and dimensional aspect ratio metrics across Indian Rupee notes:
₹10, ₹20, ₹50, ₹100, ₹200, ₹500, ₹2000.

Revision notes (accuracy):
- Colour statistics are taken from well-exposed pixels only (glare and near-black ink excluded).
- Hue is a saturation-weighted *circular* mean (the old median broke across the 0/180 red wrap),
  and its influence is damped only for practically neutral (near-zero saturation) crops.
- Profiles may carry alternate chroma/hue references (camera + lighting variants of one note).
- CNN probabilities are renormalised over the 7 denominations (mass assigned to "Background" no
  longer leaks into the denomination scores) and the CNN/optical blend adapts to how sure the CNN is.
- Confidence is now derived from the winning margin and CNN agreement instead of being floored at
  0.72, so the downstream "confidence >= x" gates actually mean something.
"""

from typing import Dict, List, Optional, Union
import cv2
import numpy as np


class DenominationDeductionEngine:
    """
    Expert deduction system mapping optical colorimetric coordinates to banknotes.
    Uses normalized RGB chromaticity (r = R/sum, g = G/sum, b = B/sum), which is
    invariant to lighting *intensity* / camera exposure (not to lighting colour; the CNN,
    trained with colour-temperature augmentation, covers that part).
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
            # Cooler / daylight-balanced cyan-blue variant of the same note.
            "chroma_alt": [np.array([0.11, 0.42, 0.47])],
            "hue_alt": [92.0],
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

    DENOMS = tuple(PROFILES.keys())

    # Blend / confidence tuning
    CNN_WEIGHT_MIN = 0.45   # weight of CNN when it is unsure
    CNN_WEIGHT_MAX = 0.85   # weight of CNN when it is very sure
    BACKGROUND_GATE = 0.35
    _CAL_MARGIN = (0.0, 0.025, 0.075, 0.125, 0.175, 0.25, 0.375, 0.45)
    _CAL_CONF = (0.50, 0.56, 0.62, 0.69, 0.82, 0.89, 0.98, 0.99)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _circular_hue(hue180: np.ndarray, weights: np.ndarray) -> float:
        """Saturation-weighted circular mean of OpenCV hue (0-180). Returns value in 0-180."""
        ang = hue180.astype(np.float32) * (2.0 * np.pi / 180.0)
        wsum = float(np.sum(weights))
        if wsum < 1e-6:
            return float(np.median(hue180))
        x = float(np.sum(weights * np.cos(ang)) / wsum)
        y = float(np.sum(weights * np.sin(ang)) / wsum)
        if np.hypot(x, y) < 1e-4:
            return float(np.median(hue180))
        return float((np.degrees(np.arctan2(y, x)) % 360.0) / 2.0)

    @staticmethod
    def _hue_diff(a: float, b: float) -> float:
        d = abs(a - b)
        return min(d, 180.0 - d)

    @classmethod
    def _references(cls, prof: Dict) -> Dict[str, List]:
        chromas = [prof["chroma"]] + list(prof.get("chroma_alt", []))
        hues = [prof["hue_center"]] + list(prof.get("hue_alt", []))
        return {"chroma": chromas, "hue": hues}

    @classmethod
    def _optical_scores(
        cls,
        mean_chroma: np.ndarray,
        median_hue: float,
        mean_sat: float,
        mean_val: float,
        aspect_ratio: float,
        aspect_ratio_valid: bool,
    ) -> Dict[str, float]:
        # Hue becomes noisy and shifts strongly with ambient colour temperature when
        # saturation is low (e.g. stone-grey ₹500 under warm indoor lighting).
        # Ramp hue trust smoothly with saturation instead of trusting low-sat hue fully.
        hue_trust = float(np.clip((mean_sat - 16.0) / 52.0, 0.08, 1.0))
        chroma_spread = float(np.max(mean_chroma) - np.min(mean_chroma))
        scores: Dict[str, float] = {}

        for denom, prof in cls.PROFILES.items():
            refs = cls._references(prof)

            # 1. Normalized chromaticity distance (best of the reference variants)
            dist_chroma = min(float(np.linalg.norm(mean_chroma - c)) for c in refs["chroma"])
            score_chroma = float(np.exp(-48.0 * (dist_chroma ** 2)))

            # 2. Chroma purity / spread compatibility (derived uniformly from profile chroma)
            spread_diff = min(abs(chroma_spread - float(np.max(c) - np.min(c))) for c in refs["chroma"])
            score_spread = float(np.exp(-0.5 * ((spread_diff / 0.11) ** 2)))

            # 3. Angular hue distance (symmetric Gaussian likelihood), trusted by saturation
            hue_diff = min(cls._hue_diff(median_hue, hc) for hc in refs["hue"])
            score_hue = float(np.exp(-0.5 * ((hue_diff / 18.0) ** 2)))

            # 4. Dimensional aspect ratio (only when a tight, rectified quad is verified)
            if aspect_ratio_valid:
                ar_diff = abs(aspect_ratio - prof["target_aspect"])
                score_ar = float(max(0.35, np.exp(-0.5 * ((ar_diff / 0.28) ** 2))))
            else:
                score_ar = 1.0

            w_c, w_s, w_h, w_ar = 0.45, 0.20, 0.25 * hue_trust, 0.10
            total = w_c + w_s + w_h + w_ar
            optical_like = (w_c * score_chroma + w_s * score_spread + w_h * score_hue + w_ar * score_ar) / total

            # Distinctive discrimination between ₹10 (Chocolate Brown) and ₹200 (Bright Yellow-Orange)
            if denom == "200":
                if mean_chroma[2] >= 0.17:      # High blue -> chocolate brown or other
                    optical_like *= 0.35
                if mean_sat < 110:              # ₹200 is vibrant
                    optical_like *= 0.40
                if mean_val < 150:              # ₹200 is bright
                    optical_like *= 0.60
                if mean_chroma[0] >= 0.52 and mean_chroma[2] <= 0.13 and mean_sat >= 130:
                    optical_like = min(1.0, optical_like * 1.35)

            elif denom == "10":
                if mean_chroma[2] <= 0.13 and mean_chroma[0] >= 0.50:   # low blue + high red = bright orange
                    optical_like *= 0.35
                if mean_sat > 135:              # ₹10 is muted earthy brown
                    optical_like *= 0.45
                if mean_val > 185:              # ₹10 is dark brown, not neon
                    optical_like *= 0.65
                if mean_chroma[2] >= 0.18 and mean_sat <= 110:
                    optical_like = min(1.0, optical_like * 1.30)

            scores[denom] = float(optical_like)
        return scores

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @classmethod
    def deduce_denomination(
        cls,
        cropped_note: np.ndarray,
        cnn_probs: Optional[Dict[str, float]] = None,
        aspect_ratio_valid: bool = True
    ) -> Dict[str, Union[str, float, bool, Dict]]:
        """
        Deduce banknote denomination using normalized chromaticity, spectral centroids,
        saturation, value, and dimensional aspect ratio metrics, fused with CNN probabilities.

        Returns the same keys as before (denomination, confidence, is_valid_currency, note_name,
        scores) plus: optical_scores, second_best, margin, ambiguous.
        """
        if cropped_note is None or cropped_note.size == 0:
            return {
                "denomination": "Background",
                "confidence": 0.0,
                "is_valid_currency": False,
                "note_name": "No Currency",
                "scores": {},
                "optical_scores": {},
                "second_best": "",
                "margin": 0.0,
                "ambiguous": True,
            }

        h, w = cropped_note.shape[:2]
        aspect_ratio = max(w, h) / max(1.0, min(w, h))

        # Sample central region to eliminate table/hand borders
        y1, y2 = max(0, int(h * 0.18)), min(h, int(h * 0.82))
        x1, x2 = max(0, int(w * 0.18)), min(w, int(w * 0.82))
        central_bgr = cropped_note[y1:y2, x1:x2]
        if central_bgr.size == 0:
            central_bgr = cropped_note

        hsv = cv2.cvtColor(central_bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        # Well-exposed pixel mask: drop deep-black ink and blown-out glare
        good = (val >= 45) & ~((val >= 240) & (sat < 30))
        if float(np.mean(good)) < 0.08:
            good = np.ones_like(good, dtype=bool)

        rgb = cv2.cvtColor(central_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) + 1e-5
        rgb_sum = np.sum(rgb, axis=2)
        r_norm = (rgb[:, :, 0] / rgb_sum)[good]
        g_norm = (rgb[:, :, 1] / rgb_sum)[good]
        b_norm = (rgb[:, :, 2] / rgb_sum)[good]
        mean_chroma = np.array([
            float(np.median(r_norm)),
            float(np.median(g_norm)),
            float(np.median(b_norm))
        ])

        median_hue = cls._circular_hue(hsv[:, :, 0][good], sat[good].astype(np.float32))
        # Saturation / value gates keep their original definition (all central pixels)
        mean_sat = float(np.mean(sat))
        mean_val = float(np.mean(val))

        optical = cls._optical_scores(
            mean_chroma, median_hue, mean_sat, mean_val, aspect_ratio, aspect_ratio_valid
        )

        # ---- Fusion with CNN ------------------------------------------------
        bg_prob = float(cnn_probs.get("Background", 0.0)) if cnn_probs else 0.0
        cnn_norm: Optional[Dict[str, float]] = None
        if cnn_probs:
            mass = sum(float(cnn_probs.get(d, 0.0)) for d in cls.DENOMS)
            if mass > 1e-6:
                cnn_norm = {d: float(cnn_probs.get(d, 0.0)) / mass for d in cls.DENOMS}

        scores: Dict[str, float] = {}
        opt_pow = {d: float(optical[d]) ** 5.8 for d in cls.DENOMS}
        opt_mass = sum(opt_pow.values())
        opt_norm = {d: opt_pow[d] / opt_mass for d in cls.DENOMS} if opt_mass > 1e-6 else dict(optical)
        if cnn_norm is not None:
            cnn_conf = max(cnn_norm.values())
            w_cnn = cls.CNN_WEIGHT_MIN + (cls.CNN_WEIGHT_MAX - cls.CNN_WEIGHT_MIN) * float(
                np.clip((cnn_conf - 0.35) / 0.50, 0.0, 1.0)
            )
            # When part of the CNN's belief already went to "Background", trust it a bit less.
            w_cnn = float(np.clip(w_cnn * (1.0 - 0.5 * bg_prob), 0.35, cls.CNN_WEIGHT_MAX))
            for d in cls.DENOMS:
                scores[d] = w_cnn * cnn_norm[d] + (1.0 - w_cnn) * opt_norm[d]
        else:
            scores = dict(opt_norm)

        # Strict Background Gate: if CNN identified Background
        if cnn_probs and bg_prob >= cls.BACKGROUND_GATE:
            return {
                "denomination": "Background",
                "confidence": 0.0,
                "is_valid_currency": False,
                "note_name": "No Currency in View",
                "scores": scores,
                "optical_scores": optical,
                "second_best": "",
                "margin": 0.0,
                "ambiguous": True,
            }

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_denom, best_val = ranked[0]
        second_denom, second_val = ranked[1] if len(ranked) > 1 else ("", 0.0)
        margin = float(best_val - second_val)

        # Confidence from margin + agreement between the two evidence sources.
        # The margin -> confidence curve was measured on 700 randomised synthetic camera scenes
        # (optical cues only): accuracy was ~56% at margin 0.03, ~69% at 0.12, ~82% at 0.17,
        # ~89% at 0.25, ~98% at 0.38 and 100% above 0.45.  Re-measure it on real photos.
        confidence = float(np.interp(margin, cls._CAL_MARGIN, cls._CAL_CONF))
        if cnn_norm is not None:
            cnn_top = max(cnn_norm, key=cnn_norm.get)
            optical_top = max(optical, key=optical.get)
            if cnn_top == best_denom and optical_top == best_denom:
                confidence += 0.08
            elif cnn_top != optical_top:
                confidence -= 0.10
        confidence = float(np.clip(confidence, 0.05, 0.99))

        return {
            "denomination": best_denom,
            "confidence": round(confidence, 3),
            "is_valid_currency": True,
            "note_name": cls.PROFILES[best_denom]["name"],
            "scores": scores,
            "optical_scores": optical,
            "second_best": second_denom,
            "margin": round(margin, 4),
            "ambiguous": bool(margin < 0.08),
        }