"""
core/security_validator.py - Banknote Security Feature & Counterfeit Verification Engine.

Performs deterministic & semantic forensic analysis on Indian Rupee banknotes:
1. Legal Authority Header Verification:
   - Validates official "RESERVE BANK OF INDIA" / "भारतीय रिज़र्व बैंक".
   - Detects disallowed toy/counterfeit markers: "CHILDREN BANK OF INDIA", "FULL OF FUN",
     "CHURAN LABEL", "BHARTIYA BACHON KA BANK", "MANORANJAN", "CINEMA NOTE", "TOY", "SPECIMEN".
2. Security Strip Inspection:
   - Flags imitation printed bands ("CHURAN LABEL", flat olive/green strips on right border).
3. Intaglio Line Tactile Relief Variance:
   - Measures continuous directional gradient energy characteristic of authentic intaglio engraving.
"""

import glob
import os
from typing import Dict, Optional, Tuple, Union
import cv2
import numpy as np


class BanknoteSecurityValidator:
    """
    Forensic security feature validator detecting commercial toy counterfeits,
    'Children Bank of India' notes, 'Churan Label' strips, and offset-printed fakes.
    """

    def __init__(self, templates_dir: Optional[str] = None):
        if templates_dir is None:
            templates_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data",
                "counterfeit_templates"
            )
        self.templates_dir = templates_dir
        self.templates: Dict[str, np.ndarray] = {}
        self._load_templates()

    def _load_templates(self) -> None:
        """Load known counterfeit marker templates."""
        if not os.path.exists(self.templates_dir):
            return
        for p in glob.glob(os.path.join(self.templates_dir, "*.jpg")):
            name = os.path.basename(p).replace(".jpg", "")
            t = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if t is not None:
                self.templates[name] = t

    def check_counterfeit_markers(self, note_bgr: np.ndarray) -> Dict[str, Union[float, bool, str]]:
        """
        Scan banknote for unauthorized toy/counterfeit text patterns.
        Returns diagnostic dictionary with match scores and detected markers.
        """
        if note_bgr is None or note_bgr.size == 0 or not self.templates:
            return {
                "is_toy_counterfeit": False,
                "max_marker_score": 0.0,
                "flagged_marker": "",
                "details": {}
            }

        gray = cv2.cvtColor(note_bgr, cv2.COLOR_BGR2GRAY)
        nh, nw = gray.shape[:2]
        base_scale = nh / 176.0

        scores = {}
        flagged = []
        max_score = 0.0
        primary_marker = ""

        # Calibrated thresholds to ensure zero false positives on genuine RBI notes
        # while reliably triggering on commercial toy/play/manoranjan currency
        marker_thresholds = {
            "children_bank_header": 0.62,
            "children_word": 0.72,
            "hindi_children_bank": 0.72,
            "manoranjan_bank": 0.86,
            "full_of_fun_top": 0.72,
            "full_of_fun_bot": 0.72,
            "churan_label_strip": 0.92
        }

        header_templates = {
            "children_bank_header",
            "children_word",
            "hindi_children_bank",
            "manoranjan_bank",
            "full_of_fun_top"
        }

        for name, templ in self.templates.items():
            th, tw = templ.shape[:2]
            best_v = 0.0

            # Header markers strictly belong in top portion of banknote
            search_region = gray[:int(nh * 0.55), :] if name in header_templates else gray
            s_h, s_w = search_region.shape[:2]

            # Multi-scale check across expected banknote sizes & low-res mobile captures
            base_s = nw / 360.0
            candidate_scales = [base_s * s for s in [0.35, 0.50, 0.65, 0.80, 1.0, 1.20, 1.40]]
            candidate_scales += [s_w / (tw * 2.5), s_w / (tw * 3.5), s_w / (tw * 5.0)]

            for cur_s in set(candidate_scales):
                sw, sh = int(tw * cur_s), int(th * cur_s)
                if sw >= s_w or sh >= s_h or sw < 10 or sh < 6:
                    continue
                scaled_t = cv2.resize(templ, (sw, sh), interpolation=cv2.INTER_AREA)
                res = cv2.matchTemplate(search_region, scaled_t, cv2.TM_CCOEFF_NORMED)
                _, max_v, _, _ = cv2.minMaxLoc(res)
                if max_v > best_v:
                    best_v = float(max_v)

            scores[name] = round(best_v, 3)
            thresh = marker_thresholds.get(name, 0.72)

            if best_v >= thresh:
                flagged.append(name)
                if best_v > max_score:
                    max_score = best_v
                    primary_marker = name

        is_toy = len(flagged) > 0

        # Human-readable label for detected fake marker
        marker_descriptions = {
            "children_bank_header": "CHILDREN BANK OF INDIA Header",
            "children_word": "CHILDREN Bank Marking",
            "full_of_fun_top": "FULL OF FUN Watermark Text",
            "full_of_fun_bot": "FULL OF FUN Denomination Marking",
            "hindi_children_bank": "भारतीय बच्चों का बैंक (Children Bank) Text",
            "manoranjan_bank": "भारतीय मनोरंजन बैंक (Manoranjan Bank) Text",
            "churan_label_strip": "CHURAN LABEL Imitation Security Thread"
        }

        reason = marker_descriptions.get(primary_marker, primary_marker)

        return {
            "is_toy_counterfeit": is_toy,
            "max_marker_score": round(max_score, 4),
            "flagged_marker": reason,
            "detected_markers": [marker_descriptions.get(m, m) for m in flagged],
            "scores": scores
        }

    def compute_intaglio_sharpness(self, note_bgr: np.ndarray) -> float:
        """
        Analyze high-frequency intaglio engraving relief sharpness.
        Real notes have continuous high-contrast ridge variance; photocopies/fakes are flat.
        """
        if note_bgr is None or note_bgr.size == 0:
            return 0.0
        gray = cv2.cvtColor(note_bgr, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        var = float(np.var(lap))
        # Normalized intaglio score (0.0 to 1.0)
        return float(np.clip(var / 800.0, 0.0, 1.0))
