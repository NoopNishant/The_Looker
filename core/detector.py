"""
core/detector.py - Robust Real-Time Banknote Localization & Face-Suppressed Tracking.

Designed for live laptop webcams & forensic inspection:
1. Built-in Haar Cascade Face Detection to actively suppress human faces from false banknote detection.
2. Geometric aspect-ratio verification: Indian banknotes strictly range from 1.65 to 2.85.
3. Multi-candidate contour polygon analysis with rectangularity & corner angle gating.
4. YCrCb/HSV skin chrominance suppression to eliminate hands, skin, and face false positives.
5. Temporal Exponential Moving Average (EMA) smoothing for rock-solid reticle tracking.
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


class BanknoteDetector:
    """
    Robust real-time banknote detector with active human face suppression.
    Reliably isolates currency notes while strictly ignoring human faces, skin, and room clutter.
    """

    def __init__(
        self,
        min_area_ratio: float = 0.04,
        max_area_ratio: float = 0.90,
        smoothing_alpha: float = 0.35
    ) -> None:
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.smoothing_alpha = smoothing_alpha

        # Temporal tracking state
        self.smooth_bbox: Optional[Tuple[float, float, float, float]] = None
        self.miss_counter: int = 0

        # Initialize OpenCV Haar Face Cascade for active face suppression
        self.face_cascade = None
        try:
            cascade_dir = cv2.data.haarcascades
            alt2_path = os.path.join(cascade_dir, "haarcascade_frontalface_alt2.xml")
            default_path = os.path.join(cascade_dir, "haarcascade_frontalface_default.xml")
            if os.path.exists(alt2_path):
                self.face_cascade = cv2.CascadeClassifier(alt2_path)
            elif os.path.exists(default_path):
                self.face_cascade = cv2.CascadeClassifier(default_path)
        except Exception:
            self.face_cascade = None

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

    @staticmethod
    def compute_skin_ratio(bgr_patch: np.ndarray) -> float:
        """Calculate fraction of pixels falling within human skin chrominance cluster."""
        if bgr_patch is None or bgr_patch.size == 0:
            return 0.0
        ycrcb = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        skin_mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
        return float(np.mean(skin_mask))

    def detect_faces(self, gray_small: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect human faces in downscaled grayscale frame."""
        if self.face_cascade is None or self.face_cascade.empty():
            return []
        try:
            faces = self.face_cascade.detectMultiScale(
                gray_small,
                scaleFactor=1.12,
                minNeighbors=3,
                minSize=(28, 28)
            )
            return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
        except Exception:
            return []

    def detect(
        self,
        bgr_frame: np.ndarray,
        is_pre_cropped: bool = False
    ) -> Dict[str, Union[bool, Tuple, np.ndarray, float, List]]:
        """
        Locate currency note in live camera frame while strictly rejecting faces.
        """
        if bgr_frame is None or bgr_frame.size == 0:
            return {
                "detected": False,
                "bbox": (0, 0, 100, 100),
                "cropped_note": None,
                "confidence": 0.0,
                "corners": None,
                "strategy": "empty",
                "default_crop": None,
                "default_bbox": (0, 0, 100, 100)
            }

        h, w = bgr_frame.shape[:2]
        frame_ar = w / float(max(1, h))

        # Define default central guided bounding box (75% width, standard ~2.15 aspect ratio)
        guide_w = int(w * 0.75)
        guide_h = min(int(h * 0.82), int(guide_w / 2.15))
        gx1 = max(0, (w - guide_w) // 2)
        gy1 = max(0, (h - guide_h) // 2)
        gx2 = min(w, gx1 + guide_w)
        gy2 = min(h, gy1 + guide_h)
        default_bbox = (gx1, gy1, gx2 - gx1, gy2 - gy1)
        default_crop = bgr_frame[gy1:gy2, gx1:gx2].copy()

        # Scale down for fast processing (<5ms)
        proc_scale = 360.0 / max(h, w)
        proc_w = int(w * proc_scale)
        proc_h = int(h * proc_scale)
        small = cv2.resize(bgr_frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # 1. Run Active Human Face Detection
        detected_faces = self.detect_faces(gray)
        face_area_ratio = sum(fw * fh for fx, fy, fw, fh in detected_faces) / float(max(1, proc_w * proc_h))

        # Explicit pre-cropped override (e.g. synthetic bench, dataset crop, or uploaded note photo)
        if is_pre_cropped:
            self.smooth_bbox = (0.0, 0.0, float(w), float(h))
            return {
                "detected": True,
                "bbox": (0, 0, w, h),
                "corners": [[0, 0], [w, 0], [w, h], [0, h]],
                "cropped_note": bgr_frame,
                "confidence": 0.98,
                "strategy": "full_banknote_frame",
                "default_crop": default_crop,
                "default_bbox": default_bbox,
                "aspect_ratio_valid": True
            }

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

            # Banknote aspect ratio constraint:
            # Indian banknotes strictly range from 1.95 to 2.52.
            # Allowing reasonable perspective tilt: 1.62 to 2.85.
            # Human faces (0.8 - 1.35) are strictly eliminated!
            if not (1.62 <= ar <= 2.85):
                continue

            rect_area = rw * rh
            rectangularity = area / max(1.0, rect_area)
            if rectangularity < 0.50:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)

            # Exclude contours that encompass the whole frame (border artifacts)
            if bw > 0.85 * proc_w and bh > 0.85 * proc_h:
                continue

            # Anti-Face Verification: Candidate bounding box must NOT be dominated by a human face
            candidate_box_area = float(max(1, bw * bh))
            is_face_box = False
            for (fx, fy, fw, fh) in detected_faces:
                inter_x1 = max(x, fx)
                inter_y1 = max(y, fy)
                inter_x2 = min(x + bw, fx + fw)
                inter_y2 = min(y + bh, fy + fh)
                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    face_box_area = float(max(1, fw * fh))
                    # Face box if the face occupies >45% of the candidate box, or candidate covers >70% of the face
                    if (inter_area / candidate_box_area > 0.45) or (inter_area / face_box_area > 0.70):
                        is_face_box = True
                        break
            if is_face_box:
                continue

            candidate_patch = small[y:y+bh, x:x+bw]
            candidate_gray = gray[y:y+bh, x:x+bw]

            edge_dens = self.compute_edge_density(candidate_gray)
            tex_energy = self.compute_texture_energy(candidate_gray)

            # Banknotes contain high density of fine print and latent guilloche lines
            if edge_dens < 0.025 and tex_energy < 25.0:
                continue

            # Bare skin filter: smooth skin has very low edge density
            skin_frac = self.compute_skin_ratio(candidate_patch)
            if skin_frac > 0.70 and edge_dens < 0.032:
                continue

            # Banknote aspect ratio score (target ~2.15)
            ar_score = max(0.0, 1.0 - abs(ar - 2.15) / 0.70)
            area_ratio = area / float(max(1, proc_w * proc_h))
            area_score = max(0.0, 1.0 - abs(area_ratio - 0.22) / 0.28)

            # Centrality factor: prioritize currency held up in camera view
            norm_cx = (cx_r - proc_w / 2.0) / (proc_w / 2.0)
            norm_cy = (cy_r - proc_h / 2.0) / (proc_h / 2.0)
            center_dist = float(np.sqrt(norm_cx ** 2 + norm_cy ** 2))
            center_score = max(0.05, 1.0 - (center_dist / 1.15))

            # Overlap with guided central reticle
            pgx1 = int(gx1 * proc_scale)
            pgy1 = int(gy1 * proc_scale)
            pgx2 = int(gx2 * proc_scale)
            pgy2 = int(gy2 * proc_scale)
            ix1 = max(x, pgx1)
            iy1 = max(y, pgy1)
            ix2 = min(x + bw, pgx2)
            iy2 = min(y + bh, pgy2)
            if ix2 > ix1 and iy2 > iy1:
                overlap_ratio = float((ix2 - ix1) * (iy2 - iy1)) / float(max(1, bw * bh))
            else:
                overlap_ratio = 0.0

            score = (
                0.25 * ar_score +
                0.20 * min(1.0, edge_dens * 12.0) +
                0.15 * rectangularity +
                0.10 * area_score +
                0.15 * center_score +
                0.15 * overlap_ratio
            )

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
        if best_bbox is not None and best_score >= 0.44:
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
                "confidence": float(np.clip(best_score * 1.1, 0.60, 0.98)),
                "strategy": "contour_quad",
                "default_crop": default_crop,
                "default_bbox": default_bbox,
                "aspect_ratio_valid": True
            }

        # Step 2: Guided Central Reticle Presence
        # When user aligns currency inside the visual reticle (even if fingers interrupt outer contour)
        gw = gx2 - gx1
        gh = gy2 - gy1
        reticle_area = float(max(1, gw * gh))
        reticle_has_face = False
        for (fx, fy, fw, fh) in detected_faces:
            orig_fx = int(fx / proc_scale)
            orig_fy = int(fy / proc_scale)
            orig_fw = int(fw / proc_scale)
            orig_fh = int(fh / proc_scale)
            ix1 = max(gx1, orig_fx)
            iy1 = max(gy1, orig_fy)
            ix2 = min(gx2, orig_fx + orig_fw)
            iy2 = min(gy2, orig_fy + orig_fh)
            if ix2 > ix1 and iy2 > iy1:
                inter_a = (ix2 - ix1) * (iy2 - iy1)
                if inter_a / reticle_area > 0.25 or inter_a / float(max(1, orig_fw * orig_fh)) > 0.35:
                    reticle_has_face = True
                    break

        if not reticle_has_face and default_crop is not None and default_crop.size > 0:
            gray_crop = cv2.cvtColor(default_crop, cv2.COLOR_BGR2GRAY)
            crop_edge = self.compute_edge_density(gray_crop)
            crop_tex = self.compute_texture_energy(gray_crop)
            crop_skin = self.compute_skin_ratio(default_crop)

            # High print/intaglio density in reticle without being bare skin
            if (crop_edge >= 0.024 or crop_tex >= 30.0) and not (crop_skin > 0.75 and crop_edge < 0.025):
                self.miss_counter = 0
                return {
                    "detected": True,
                    "bbox": default_bbox,
                    "corners": [[gx1, gy1], [gx2, gy1], [gx2, gy2], [gx1, gy2]],
                    "cropped_note": default_crop,
                    "confidence": 0.88,
                    "strategy": "guided_reticle",
                    "default_crop": default_crop,
                    "default_bbox": default_bbox,
                    "aspect_ratio_valid": False
                }

        # Step 3: Standby (empty frame, user face, or smooth clothes)
        self.miss_counter += 1
        if self.miss_counter > 6:
            self.smooth_bbox = None

        return {
            "detected": False,
            "bbox": default_bbox,
            "corners": None,
            "cropped_note": default_crop,
            "confidence": 0.0,
            "strategy": "standby",
            "default_crop": default_crop,
            "default_bbox": default_bbox,
            "aspect_ratio_valid": False
        }
