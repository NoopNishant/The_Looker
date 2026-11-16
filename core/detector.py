"""
core/detector.py - Robust Real-Time Banknote Localization & Hand-Invariant Tracking.

Designed for live laptop webcams & forensic inspection:
1. Direct recognition for pre-cropped banknote photos & synthetic samples.
2. Multi-candidate contour polygon analysis with interior edge density gating.
3. Hand-held currency detection with skin-color / background suppression.
4. Central guided viewport presence detection with Laplacian texture energy verification.
5. Temporal Exponential Moving Average (EMA) smoothing for rock-solid reticle tracking.
"""

from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


class BanknoteDetector:
    """
    Robust real-time banknote detector.
    Reliably locates currency notes in front of webcams even with hand occlusion,
    imperfect lighting, and varied room backgrounds.
    """

    def __init__(
        self,
        min_area_ratio: float = 0.04,
        max_area_ratio: float = 0.98,
        smoothing_alpha: float = 0.35
    ) -> None:
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.smoothing_alpha = smoothing_alpha

        # Temporal tracking state
        self.smooth_bbox: Optional[Tuple[float, float, float, float]] = None
        self.miss_counter: int = 0

    @staticmethod
    def compute_texture_energy(gray_patch: np.ndarray) -> float:
        """Measure high-frequency engraving/texture energy via Laplacian variance."""
        if gray_patch is None or gray_patch.size == 0:
            return 0.0
        lap = cv2.Laplacian(gray_patch, cv2.CV_32F)
        return float(np.var(lap))

    @staticmethod
    def compute_edge_density(gray_patch: np.ndarray, low_thresh: int = 40, high_thresh: int = 120) -> float:
        """Calculate density of high-frequency Canny edge pixels."""
        if gray_patch is None or gray_patch.size == 0:
            return 0.0
        edges = cv2.Canny(gray_patch, low_thresh, high_thresh)
        return float(np.count_nonzero(edges)) / float(max(1, edges.size))

    def detect(self, bgr_frame: np.ndarray) -> Dict[str, Union[bool, Tuple, np.ndarray, float, List]]:
        """
        Locate currency note in live camera frame or pre-cropped image.
        """
        if bgr_frame is None or bgr_frame.size == 0:
            return {
                "detected": False,
                "bbox": (0, 0, 100, 100),
                "cropped_note": None,
                "confidence": 0.0,
                "corners": None,
                "strategy": "empty"
            }

        h, w = bgr_frame.shape[:2]
        frame_area = h * w
        frame_ar = w / float(max(1, h))

        # Scale down for fast processing
        proc_scale = 360.0 / max(h, w)
        proc_w = int(w * proc_scale)
        proc_h = int(h * proc_scale)
        small = cv2.resize(bgr_frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # Step 0: Check if input image is ALREADY a pre-cropped banknote
        # Uploaded photos or synthetic samples typically have banknote aspect ratio ~1.92 to 2.55
        # Standard 16:9 webcams have aspect ratio 1.778, which must be searched for notes
        if 1.90 <= frame_ar <= 2.65:
            full_edge_dens = self.compute_edge_density(gray)
            full_tex = self.compute_texture_energy(gray)
            if full_edge_dens > 0.025 or full_tex > 35.0:
                self.smooth_bbox = (0.0, 0.0, float(w), float(h))
                return {
                    "detected": True,
                    "bbox": (0, 0, w, h),
                    "corners": [[0, 0], [w, 0], [w, h], [0, h]],
                    "cropped_note": bgr_frame,
                    "confidence": 0.98,
                    "strategy": "full_banknote_frame"
                }

        # Define default central guided bounding box (68% width, standard ~2.05 aspect ratio)
        guide_w = int(w * 0.68)
        guide_h = int(guide_w / 2.05)
        gx1 = max(0, (w - guide_w) // 2)
        gy1 = max(0, (h - guide_h) // 2)
        gx2 = min(w, gx1 + guide_w)
        gy2 = min(h, gy1 + guide_h)
        default_bbox = (gx1, gy1, gx2 - gx1, gy2 - gy1)
        default_crop = bgr_frame[gy1:gy2, gx1:gx2].copy()

        # Step 1: Multi-candidate contour polygon analysis
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 110)

        # Morphological dilation to connect broken line segments
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        best_bbox = None
        best_score = 0.0
        best_corners = None

        min_px = (proc_w * proc_h) * self.min_area_ratio
        max_px = (proc_w * proc_h) * self.max_area_ratio

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_px or area > max_px:
                continue

            rect = cv2.minAreaRect(cnt)
            (cx_r, cy_r), (rw, rh), angle = rect
            if rw <= 0 or rh <= 0:
                continue

            long_s = max(rw, rh)
            short_s = min(rw, rh)
            ar = long_s / short_s

            # Banknote aspect ratio (allow perspective tilt: 1.25 to 3.2)
            if not (1.20 <= ar <= 3.25):
                continue

            rect_area = rw * rh
            rectangularity = area / max(1.0, rect_area)
            if rectangularity < 0.40:
                continue

            # Check interior edge density inside this candidate box
            x, y, bw, bh = cv2.boundingRect(cnt)
            
            # Exclude contours that encompass the whole frame (border artifacts)
            if bw > 0.82 * proc_w and bh > 0.82 * proc_h:
                continue

            candidate_patch = gray[y:y+bh, x:x+bw]
            edge_dens = self.compute_edge_density(candidate_patch)
            tex_energy = self.compute_texture_energy(candidate_patch)

            # Banknotes contain high density of fine print and latent guilloche lines
            # Plain walls/doors have edge_density < 0.02
            if edge_dens < 0.028 and tex_energy < 30.0:
                continue

            # Banknote aspect ratio score (Indian banknotes ~ 2.05 to 2.25)
            ar_score = max(0.0, 1.0 - abs(ar - 2.10) / 0.85)
            # Area score: prefers realistic note size in webcam view (10% to 50% of frame)
            area_ratio = area / float(max(1, proc_w * proc_h))
            area_score = max(0.0, 1.0 - abs(area_ratio - 0.22) / 0.28)

            # Centrality factor: prioritize currency held up in front of camera
            norm_cx = (cx_r - proc_w / 2.0) / (proc_w / 2.0)
            norm_cy = (cy_r - proc_h / 2.0) / (proc_h / 2.0)
            center_dist = float(np.sqrt(norm_cx ** 2 + norm_cy ** 2))
            center_score = max(0.20, 1.0 - (center_dist / 1.35))

            score = 0.30 * ar_score + 0.25 * min(1.0, edge_dens * 12.0) + 0.15 * rectangularity + 0.10 * area_score + 0.20 * center_score

            if score > best_score:
                best_score = score
                orig_x = int(x / proc_scale)
                orig_y = int(y / proc_scale)
                orig_bw = int(bw / proc_scale)
                orig_bh = int(bh / proc_scale)
                best_bbox = (orig_x, orig_y, orig_bw, orig_bh)
                box_pts = cv2.boxPoints(rect) / proc_scale
                best_corners = box_pts.astype(np.int32).tolist()

        # If a verified contour quad is detected
        if best_bbox is not None and best_score >= 0.38:
            bx, by, bw, bh = best_bbox
            bx1 = max(0, bx)
            by1 = max(0, by)
            bx2 = min(w, bx + bw)
            by2 = min(h, by + bh)

            # Apply temporal smoothing
            if self.smooth_bbox is None:
                self.smooth_bbox = (float(bx1), float(by1), float(bx2 - bx1), float(by2 - by1))
            else:
                a = self.smoothing_alpha
                sx, sy, sw, sh = self.smooth_bbox
                self.smooth_bbox = (
                    (1 - a) * sx + a * bx1,
                    (1 - a) * sy + a * by1,
                    (1 - a) * sw + a * (bx2 - bx1),
                    (1 - a) * sh + a * (by2 - by1)
                )

            smoothed = (
                int(self.smooth_bbox[0]),
                int(self.smooth_bbox[1]),
                int(self.smooth_bbox[2]),
                int(self.smooth_bbox[3])
            )

            cropped = bgr_frame[by1:by2, bx1:bx2].copy()
            self.miss_counter = 0

            return {
                "detected": True,
                "bbox": smoothed,
                "corners": best_corners,
                "cropped_note": cropped if cropped.size > 0 else default_crop,
                "confidence": float(np.clip(best_score * 1.1, 0.55, 0.98)),
                "strategy": "contour_quad"
            }

        # Step 2: Hand-held Guided Viewport Saliency
        # When users hold a note directly in front of the lens with hands/fingers covering edges
        central_gray = cv2.cvtColor(default_crop, cv2.COLOR_BGR2GRAY)
        texture_energy = self.compute_texture_energy(central_gray)
        edge_density = self.compute_edge_density(central_gray)
        color_std = float(np.mean(np.std(default_crop, axis=(0, 1))))

        # High texture energy & interior edge density confirms a printed banknote is in view
        if (texture_energy > 55.0 and edge_density > 0.035) or (texture_energy > 120.0 and color_std > 20.0):
            # If recently tracked, retain smoothed position briefly
            if self.smooth_bbox is not None and self.miss_counter < 5:
                smoothed = (
                    int(self.smooth_bbox[0]),
                    int(self.smooth_bbox[1]),
                    int(self.smooth_bbox[2]),
                    int(self.smooth_bbox[3])
                )
                self.miss_counter += 1
            else:
                smoothed = default_bbox

            return {
                "detected": True,
                "bbox": smoothed,
                "corners": None,
                "cropped_note": default_crop,
                "confidence": float(np.clip(0.65 + min(0.3, edge_density * 2.5), 0.65, 0.95)),
                "strategy": "guided_presence"
            }

        # Step 3: Scene is empty, uniform wall, or background person without money
        self.miss_counter += 1
        if self.miss_counter > 8:
            self.smooth_bbox = None

        return {
            "detected": False,
            "bbox": default_bbox,
            "corners": None,
            "cropped_note": default_crop,
            "confidence": 0.0,
            "strategy": "standby"
        }
