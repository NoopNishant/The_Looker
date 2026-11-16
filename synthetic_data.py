"""
synthetic_data.py - Synthetic Currency & Counterfeit Generator.

Generates realistic simulated currency samples for automated testing,
verification, and offline demonstrations:
- Genuine Currency: Intaglio guilloche lines, microprint, embedded watermark, continuous tone.
- Counterfeit Currency: Inkjet/laser CMYK halftone dot screens (rosettes), surface blur, missing watermark.
"""

from typing import Tuple
import numpy as np
import cv2


class SyntheticCurrencyGenerator:
    """
    Synthesizes authentic-style and counterfeit currency samples.
    """

    PALETTES = {
        "10": (80, 110, 160),     # Chocolate Brown
        "20": (60, 205, 165),     # Greenish Yellow (distinctly green-tinted yellow)
        "50": (195, 175, 45),     # Fluorescent Blue / Cyan
        "100": (175, 120, 140),   # Lavender
        "200": (30, 125, 245),    # Bright Orange (distinct high red, low green/blue)
        "500": (130, 140, 130),   # Stone Gray
        "2000": (155, 45, 215),   # Magenta (Deep Pink, distinct from lavender)
    }

    @classmethod
    def generate_banknote(
        cls,
        denomination: str = "500",
        is_counterfeit: bool = False,
        width: int = 640,
        height: int = 320
    ) -> np.ndarray:
        """
        Generate a synthetic currency image.
        Args:
            denomination: One of "10", "20", "50", "100", "200", "500", "2000".
            is_counterfeit: If True, injects CMYK halftone printer dots and suppresses intaglio.
        Returns:
            BGR image.
        """
        base_color = cls.PALETTES.get(denomination, (120, 120, 120))
        img = np.full((height, width, 3), base_color, dtype=np.uint8)

        # Subtle organic paper substrate texture
        raw_noise = np.random.normal(0, 10, (height, width, 3)).astype(np.float32)
        smooth_grain = cv2.GaussianBlur(raw_noise, (5, 5), 1.5)
        img = np.clip(img.astype(np.float32) + smooth_grain, 0, 255).astype(np.uint8)

        # Draw decorative security borders
        cv2.rectangle(img, (12, 12), (width - 12, height - 12), (50, 50, 50), 2)
        cv2.rectangle(img, (16, 16), (width - 16, height - 16), (80, 80, 80), 1)

        # Draw Large Denomination Numeral
        cv2.putText(
            img, str(denomination), (int(width * 0.72), int(height * 0.38)),
            cv2.FONT_HERSHEY_DUPLEX, 1.8, (30, 30, 30), 4, cv2.LINE_AA
        )
        cv2.putText(
            img, f"RESERVE BANK", (int(width * 0.35), int(height * 0.22)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2, cv2.LINE_AA
        )

        # Left Watermark Window Oval
        wm_center = (int(width * 0.20), int(height * 0.52))
        cv2.ellipse(img, wm_center, (int(width * 0.10), int(height * 0.32)), 0, 0, 360, (235, 235, 240), -1)
        cv2.ellipse(img, wm_center, (int(width * 0.10), int(height * 0.32)), 0, 0, 360, (180, 180, 180), 1)

        # Embedded security thread
        thread_x = int(width * 0.42)
        cv2.line(img, (thread_x, 12), (thread_x, height - 12), (40, 40, 40), 3)

        if not is_counterfeit:
            # GENUINE CURRENCY FEATURES:
            # 1. Fine-line Guilloche Intaglio Lathe Waves across center
            for i in range(12):
                pts = []
                freq = 0.045 + i * 0.005
                amp = 14 + (i % 4) * 5
                y_offset = int(height * 0.50) + (i - 6) * 10
                for x in range(int(width * 0.45), int(width * 0.88), 2):
                    y = int(y_offset + amp * np.sin(x * freq))
                    pts.append([x, y])
                pts = np.array(pts, np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [pts], False, (45, 45, 45), 1, cv2.LINE_AA)

            # 2. Microprinting continuous ribbon
            micro_text = "RBI " * 60
            cv2.putText(
                img, micro_text, (int(width * 0.36), int(height * 0.88)),
                cv2.FONT_HERSHEY_PLAIN, 0.55, (25, 25, 25), 1, cv2.LINE_AA
            )

        else:
            # COUNTERFEIT CURRENCY FEATURES:
            # Inject 4-color CMYK Halftone Screen Rosettes (periodic dither grid at 15°, 45°, 75°)
            y_coords, x_coords = np.ogrid[:height, :width]

            # Cyan (15°), Magenta (75°), Black (45°)
            th_c = np.radians(15.0)
            u_c = x_coords * np.cos(th_c) + y_coords * np.sin(th_c)
            v_c = -x_coords * np.sin(th_c) + y_coords * np.cos(th_c)
            halftone_c = (np.sin(u_c * 0.75) * np.sin(v_c * 0.75)) > 0.42

            th_m = np.radians(75.0)
            u_m = x_coords * np.cos(th_m) + y_coords * np.sin(th_m)
            v_m = -x_coords * np.sin(th_m) + y_coords * np.cos(th_m)
            halftone_m = (np.sin(u_m * 0.75) * np.sin(v_m * 0.75)) > 0.42

            th_k = np.radians(45.0)
            u_k = x_coords * np.cos(th_k) + y_coords * np.sin(th_k)
            v_k = -x_coords * np.sin(th_k) + y_coords * np.cos(th_k)
            halftone_k = (np.sin(u_k * 0.75) * np.sin(v_k * 0.75)) > 0.45

            # Apply CMYK halftone ink dots across the active printing zone
            roi_y1, roi_y2 = int(height * 0.20), int(height * 0.80)
            roi_x1, roi_x2 = int(width * 0.32), int(width * 0.88)

            dot_c = halftone_c[roi_y1:roi_y2, roi_x1:roi_x2]
            dot_m = halftone_m[roi_y1:roi_y2, roi_x1:roi_x2]
            dot_k = halftone_k[roi_y1:roi_y2, roi_x1:roi_x2]

            sub_zone = img[roi_y1:roi_y2, roi_x1:roi_x2]
            sub_zone[dot_c] = (235, 210, 20)   # Cyan ink dots
            sub_zone[dot_m] = (180, 25, 225)   # Magenta ink dots
            sub_zone[dot_k] = (25, 25, 25)     # Black carbon toner dots

            # Non-intaglio flat print
            cv2.putText(
                img, "RBI RBI RBI", (int(width * 0.36), int(height * 0.88)),
                cv2.FONT_HERSHEY_PLAIN, 0.6, (90, 90, 90), 1, cv2.LINE_AA
            )
            img = cv2.GaussianBlur(img, (3, 3), 0.6)

        return img
