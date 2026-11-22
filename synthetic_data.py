"""
synthetic_data.py - Synthetic Currency & Counterfeit Generator.

Generates realistic simulated currency samples for automated testing,
verification, and offline demonstrations:
- Genuine Currency: Intaglio guilloche lines, microprint, embedded watermark, continuous tone.
- Counterfeit Currency: Inkjet/laser CMYK halftone dot screens (rosettes), surface blur, missing watermark.

New in this revision:
- ASPECT_RATIOS: real-world length/width ratio per denomination.
- generate_scene(): places a note into a cluttered camera frame with rotation,
  perspective keystone, colour-temperature cast, vignette, shadow and sensor noise,
  and returns the ground-truth corner quad. Used for detector tests and for
  training-time realism.  generate_banknote() is unchanged (tests rely on it).
"""

from typing import Optional, Tuple
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

    # Real Mahatma Gandhi (New) series length / width ratios.
    ASPECT_RATIOS = {
        "10": 1.95,
        "20": 2.03,
        "50": 2.03,
        "100": 2.15,
        "200": 2.21,
        "500": 2.27,
        "2000": 2.52,
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

        if not is_counterfeit:
            header_text = "RESERVE BANK OF INDIA"
            cv2.putText(
                img, header_text, (int(width * 0.30), int(height * 0.22)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA
            )
        else:
            header_text = "CHILDREN BANK OF INDIA"
            cv2.putText(
                img, header_text, (int(width * 0.28), int(height * 0.22)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2, cv2.LINE_AA
            )
            cv2.putText(
                img, "FULL OF FUN", (int(width * 0.08), int(height * 0.25)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 25, 25), 2, cv2.LINE_AA
            )
            cv2.putText(
                img, "FULL OF FUN", (int(width * 0.68), int(height * 0.90)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 25, 25), 2, cv2.LINE_AA
            )

        # Left Watermark Window Oval
        wm_center = (int(width * 0.20), int(height * 0.52))
        cv2.ellipse(img, wm_center, (int(width * 0.10), int(height * 0.32)), 0, 0, 360, (235, 235, 240), -1)
        cv2.ellipse(img, wm_center, (int(width * 0.10), int(height * 0.32)), 0, 0, 360, (180, 180, 180), 1)

        # Embedded security thread / strip
        thread_x = int(width * 0.42)
        if not is_counterfeit:
            cv2.line(img, (thread_x, 12), (thread_x, height - 12), (40, 40, 40), 3)
        else:
            # Imitation strip on right side with CHURAN LABEL
            strip_x = int(width * 0.82)
            cv2.rectangle(img, (strip_x - 12, 12), (strip_x + 12, height - 12), (60, 140, 90), -1)
            cv2.putText(
                img, "CHURAN", (strip_x - 8, int(height * 0.45)),
                cv2.FONT_HERSHEY_PLAIN, 0.9, (20, 20, 20), 1, cv2.LINE_AA
            )
            cv2.putText(
                img, "LABEL", (strip_x - 8, int(height * 0.60)),
                cv2.FONT_HERSHEY_PLAIN, 0.9, (20, 20, 20), 1, cv2.LINE_AA
            )

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

    @classmethod
    def generate_scene(
        cls,
        denomination: str = "500",
        is_counterfeit: bool = False,
        frame_size: Tuple[int, int] = (1280, 720),
        note_width_frac: float = 0.42,
        rotation_deg: float = 0.0,
        perspective: float = 0.0,
        color_cast: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        brightness: float = 1.0,
        background: Optional[Tuple[int, int, int]] = None,
        clutter: bool = True,
        noise_sigma: float = 3.0,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Render a note inside a realistic camera frame.

        Args:
            frame_size: (width, height) of the output frame.
            note_width_frac: note length as a fraction of frame width (before rotation).
            rotation_deg: in-plane rotation of the note.
            perspective: 0..0.3 keystone strength (top edge shorter than bottom edge).
            color_cast: (R, G, B) gains emulating incandescent / cool-white lighting.
            brightness: global exposure gain.
            background: BGR desk colour; random mid-tone if None.
            clutter: draw random rectangles / lines on the desk.
        Returns:
            (bgr_frame, gt_corners) with gt_corners a (4, 2) float32 quad in frame pixels.
        """
        rng = np.random.default_rng(seed)
        W, H = frame_size
        ar = cls.ASPECT_RATIOS.get(denomination, 2.15)

        note_w = 640
        note = cls.generate_banknote(
            denomination, is_counterfeit, width=note_w, height=int(round(note_w / ar))
        )
        nh = note.shape[0]

        base = np.array(background if background is not None else rng.integers(90, 200, 3), dtype=np.float32)
        bg = np.empty((H, W, 3), np.float32)
        bg[:] = base
        bg *= np.linspace(0.92, 1.08, W, dtype=np.float32)[None, :, None]
        bg = np.clip(bg, 0, 255).astype(np.uint8)
        if clutter:
            for _ in range(int(rng.integers(6, 14))):
                x1, y1 = int(rng.integers(0, W)), int(rng.integers(0, H))
                x2, y2 = x1 + int(rng.integers(40, 300)), y1 + int(rng.integers(20, 160))
                col = tuple(int(c) for c in np.clip(base + rng.integers(-45, 46, 3), 0, 255))
                thickness = -1 if rng.random() < 0.5 else 2
                cv2.rectangle(bg, (x1, y1), (x2, y2), col, thickness)

        tw = W * note_width_frac
        th = tw / ar
        cx = W / 2 + rng.uniform(-0.04, 0.04) * W
        cy = H / 2 + rng.uniform(-0.04, 0.04) * H
        pts = np.array(
            [[-tw / 2, -th / 2], [tw / 2, -th / 2], [tw / 2, th / 2], [-tw / 2, th / 2]],
            dtype=np.float32,
        )
        if perspective > 0:
            pts[0, 0] *= (1.0 - perspective)
            pts[1, 0] *= (1.0 - perspective)
            pts[:2, 1] *= (1.0 - 0.3 * perspective)
        t = np.radians(rotation_deg)
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]], dtype=np.float32)
        dst = pts @ R.T + np.array([cx, cy], dtype=np.float32)

        src = np.array([[0, 0], [note_w - 1, 0], [note_w - 1, nh - 1], [0, nh - 1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(note, M, (W, H))
        mask = cv2.warpPerspective(np.full((nh, note_w), 255, np.uint8), M, (W, H))

        # Soft contact shadow
        shadow = cv2.GaussianBlur(np.roll(mask, (6, 8), axis=(0, 1)), (0, 0), 7).astype(np.float32) / 255.0
        comp = bg.astype(np.float32) * (1.0 - 0.28 * shadow)[:, :, None]
        m3 = (mask > 127)[:, :, None]
        comp = np.where(m3, warped.astype(np.float32), comp)

        # Lighting: colour cast (RGB gains applied to BGR), exposure, vignette
        r, g, b = color_cast
        comp *= np.array([b, g, r], dtype=np.float32)[None, None, :] * brightness
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        vig = 1.0 - 0.18 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / 2.0
        comp *= vig[:, :, None]
        if noise_sigma > 0:
            comp += rng.normal(0, noise_sigma, comp.shape).astype(np.float32)
        return np.clip(comp, 0, 255).astype(np.uint8), dst.astype(np.float32)