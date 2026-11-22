"""
core/detector.py - Robust Real-Time Banknote Localization & Face-Suppressed Tracking.

Designed for live laptop webcams & forensic inspection:
1. Built-in Haar Cascade Face Detection to actively suppress human faces from false banknote detection.
2. Geometric aspect-ratio verification measured on the *quadrilateral* (perspective-aware):
   Indian banknotes range from 1.95 to 2.52, tolerance 1.55 - 2.95 for tilt.
3. Two complementary candidate generators:
     - auto-thresholded Canny edges (adapts to scene brightness), and
     - background-difference mask (Lab distance from the frame border colour), which finds
       low-contrast notes (e.g. grey Rs.500 on a grey desk) that have weak outer edges.
4. Quad fitting (approxPolyDP) with a perspective-rectified crop: the returned
   `cropped_note` is a clean, upright, background-free note image whose width/height equals the
   measured aspect ratio.  This makes the CNN input, the colour statistics and the aspect-ratio cue
   in the denomination engine all consistent (the old axis-aligned crop shrank the aspect ratio and
   contaminated colours with desk pixels whenever the note was tilted).
5. YCrCb skin chrominance suppression to eliminate hands, skin, and face false positives.
6. Corner-level exponential smoothing that resets on large jumps, plus a short track-hold so a
   single missed frame does not make the reticle flicker.
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def order_corners(pts: np.ndarray) -> np.ndarray:
    """Return corners ordered top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()  # y - x
    ordered = np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )
    if len({(float(p[0]), float(p[1])) for p in ordered}) < 4:
        # Degenerate (e.g. ~45 degree tilt): fall back to angular sort around the centroid.
        c = pts.mean(axis=0)
        ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        ordered = pts[np.argsort(ang)]
        start = int(np.argmin(ordered.sum(axis=1)))
        ordered = np.roll(ordered, -start, axis=0)
    return ordered.astype(np.float32)


def quad_side_lengths(quad: np.ndarray) -> Tuple[float, float]:
    """Mean (top/bottom) width and mean (left/right) height of an ordered quad."""
    tl, tr, br, bl = quad
    width = 0.5 * (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl))
    height = 0.5 * (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr))
    return float(width), float(height)


def inset_quad(quad: np.ndarray, d: float) -> np.ndarray:
    """
    Move every side of an ordered quad inward by `d` pixels (along the quad's own axes).
    Compensates for the outline inflation caused by dilating the edge map.
    """
    if d <= 0:
        return quad
    tl, tr, br, bl = quad
    u = (tr - tl) + (br - bl)
    v = (bl - tl) + (br - tr)
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-6 or nv < 1e-6:
        return quad
    u, v = u / nu, v / nv
    w, h = quad_side_lengths(quad)
    if w <= 2.5 * d or h <= 2.5 * d:
        return quad
    out = np.array([tl + d * u + d * v, tr - d * u + d * v, br - d * u - d * v, bl + d * u - d * v], dtype=np.float32)
    return out


def _align_corners(prev: np.ndarray, new: np.ndarray) -> np.ndarray:
    """Roll `new` (4,2) so corner i corresponds to prev corner i (avoids smoothing across a relabel)."""
    best_roll, best_cost = 0, float("inf")
    for k in range(4):
        cost = float(np.sum(np.linalg.norm(np.roll(new, -k, axis=0) - prev, axis=1)))
        if cost < best_cost:
            best_cost, best_roll = cost, k
    return np.roll(new, -best_roll, axis=0)


def _bbox_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax1 + aw, bx1 + bw), min(ay1 + ah, by1 + bh)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return float(inter / max(1e-6, union))


class BanknoteDetector:
    """
    Robust real-time banknote detector with active human face suppression.
    Reliably isolates currency notes while strictly ignoring human faces, skin, and room clutter.
    """

    MIN_AR = 1.55
    MAX_AR = 2.95
    ACCEPT_SCORE = 0.44

    def __init__(
        self,
        min_area_ratio: float = 0.04,
        max_area_ratio: float = 0.90,
        smoothing_alpha: float = 0.40,
        proc_size: float = 400.0,
        hold_frames: int = 3,
    ) -> None:
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.smoothing_alpha = smoothing_alpha
        self.proc_size = float(proc_size)
        self.hold_frames = hold_frames

        # Temporal tracking state
        self.smooth_bbox: Optional[Tuple[float, float, float, float]] = None
        self.smooth_corners: Optional[np.ndarray] = None
        self.last_raw_corners: Optional[np.ndarray] = None
        self.hold_counter: int = 0
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

    # ------------------------------------------------------------------ #
    # Image statistics
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # Perspective rectification
    # ------------------------------------------------------------------ #
    @staticmethod
    def warp_note(
        frame: np.ndarray,
        corners: np.ndarray,
        max_long_side: int = 640
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Perspective-rectify the quad to an upright, landscape rectangle.
        Returns (warped_bgr, measured_aspect_ratio) or (None, 0.0) if the quad is degenerate.
        The long side is always horizontal (a note held portrait is rotated 90 degrees).
        """
        quad = order_corners(corners)
        width, height = quad_side_lengths(quad)
        if height > width:
            quad = quad[[1, 2, 3, 0]]
            width, height = height, width
        if width < 16 or height < 8:
            return None, 0.0

        scale = min(1.0, max_long_side / width)
        out_w = max(16, int(round(width * scale)))
        out_h = max(8, int(round(height * scale)))
        dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
        warped = cv2.warpPerspective(
            frame, M, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        return warped, float(width / max(height, 1e-6))

    # ------------------------------------------------------------------ #
    # Candidate generation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _auto_canny(blurred_gray: np.ndarray) -> np.ndarray:
        """Canny with thresholds adapted to scene brightness (clamped to sane limits)."""
        med = float(np.median(blurred_gray))
        low = int(np.clip(0.67 * med, 20, 60))
        high = int(np.clip(1.33 * med, 70, 160))
        return cv2.Canny(blurred_gray, low, high)

    @staticmethod
    def _background_diff_mask(small_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Foreground mask = Lab distance from the median colour of the frame border ring.
        Finds low-contrast notes whose outline is too weak for Canny.
        """
        h, w = small_bgr.shape[:2]
        m = max(2, int(0.06 * min(h, w)))
        lab = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        ring = np.concatenate([
            lab[:m].reshape(-1, 3), lab[-m:].reshape(-1, 3),
            lab[:, :m].reshape(-1, 3), lab[:, -m:].reshape(-1, 3)
        ])
        bg = np.median(ring, axis=0)
        diff = np.linalg.norm(lab - bg, axis=2)
        if float(diff.max()) < 18.0:
            return None
        diff8 = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        diff8 = cv2.GaussianBlur(diff8, (5, 5), 0)
        _, mask = cv2.threshold(diff8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        return mask

    @staticmethod
    def _overlaps_face(bbox: Tuple[int, int, int, int], faces: List[Tuple[int, int, int, int]]) -> bool:
        x, y, bw, bh = bbox
        cand_area = float(max(1, bw * bh))
        for (fx, fy, fw, fh) in faces:
            ix1, iy1 = max(x, fx), max(y, fy)
            ix2, iy2 = min(x + bw, fx + fw), min(y + bh, fy + fh)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                if (inter / cand_area > 0.45) or (inter / float(max(1, fw * fh)) > 0.70):
                    return True
        return False

    def _evaluate_contour(
        self,
        cnt: np.ndarray,
        small: np.ndarray,
        gray: np.ndarray,
        faces: List[Tuple[int, int, int, int]],
        guide_proc: Tuple[int, int, int, int],
        inset_px: float = 0.0,
    ) -> Optional[Dict]:
        proc_h, proc_w = gray.shape[:2]
        area = cv2.contourArea(cnt)
        if area < proc_w * proc_h * self.min_area_ratio or area > proc_w * proc_h * self.max_area_ratio:
            return None

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            return None
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.025 * peri, True)
        is_quad = len(approx) == 4 and cv2.isContourConvex(approx)
        if is_quad:
            quad = approx.reshape(4, 2).astype(np.float32)
        else:
            quad = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)
        quad_raw = order_corners(quad)

        # Shape test on the RAW quad (before the dilation-compensation inset, which would shrink the
        # box and inflate the apparent fill). A real note's hull fills ~95%+ of its quad; an ellipse
        # (e.g. the oval watermark window) fills only pi/4 = 78%. Symmetric ratio because an
        # approximated quad is inscribed in the hull.
        raw_area = float(cv2.contourArea(quad_raw))
        if raw_area <= 1:
            return None
        rectangularity = min(hull_area, raw_area) / max(hull_area, raw_area)
        if rectangularity < 0.84:
            return None

        quad = inset_quad(quad_raw, inset_px)
        qw, qh = quad_side_lengths(quad)
        if qw <= 1 or qh <= 1:
            return None
        ar = max(qw, qh) / min(qw, qh)
        # Human faces (0.8 - 1.35) and most clutter are eliminated here.
        if not (self.MIN_AR <= ar <= self.MAX_AR):
            return None

        quad_area = float(cv2.contourArea(quad))
        if quad_area <= 1:
            return None

        x, y, bw, bh = cv2.boundingRect(quad.astype(np.int32))
        x, y = max(0, x), max(0, y)
        bw, bh = min(bw, proc_w - x), min(bh, proc_h - y)
        if bw <= 2 or bh <= 2:
            return None
        # Exclude contours that encompass the whole frame (border artifacts)
        if bw > 0.85 * proc_w and bh > 0.85 * proc_h:
            return None
        if self._overlaps_face((x, y, bw, bh), faces):
            return None

        cand_gray = gray[y:y + bh, x:x + bw]
        cand_patch = small[y:y + bh, x:x + bw]
        edge_dens = self.compute_edge_density(cand_gray)
        tex_energy = self.compute_texture_energy(cand_gray)
        # Banknotes contain high density of fine print and latent guilloche lines
        if edge_dens < 0.025 and tex_energy < 25.0:
            return None
        # Bare skin filter: smooth skin has very low edge density
        skin_frac = self.compute_skin_ratio(cand_patch)
        if skin_frac > 0.70 and edge_dens < 0.032:
            return None

        ar_score = max(0.0, 1.0 - abs(ar - 2.15) / 0.70)
        area_ratio = quad_area / float(max(1, proc_w * proc_h))
        area_score = max(0.0, 1.0 - abs(area_ratio - 0.22) / 0.28)

        cxq, cyq = quad.mean(axis=0)
        norm_cx = (cxq - proc_w / 2.0) / (proc_w / 2.0)
        norm_cy = (cyq - proc_h / 2.0) / (proc_h / 2.0)
        center_dist = float(np.sqrt(norm_cx ** 2 + norm_cy ** 2))
        center_score = max(0.05, 1.0 - (center_dist / 1.15))

        pgx1, pgy1, pgx2, pgy2 = guide_proc
        ix1, iy1 = max(x, pgx1), max(y, pgy1)
        ix2, iy2 = min(x + bw, pgx2), min(y + bh, pgy2)
        overlap_ratio = (
            float((ix2 - ix1) * (iy2 - iy1)) / float(max(1, bw * bh))
            if (ix2 > ix1 and iy2 > iy1) else 0.0
        )

        score = (
            0.25 * ar_score +
            0.20 * min(1.0, edge_dens * 12.0) +
            0.15 * rectangularity +
            0.10 * area_score +
            0.15 * center_score +
            0.15 * overlap_ratio +
            (0.06 if is_quad else 0.0)
        )
        return {"score": float(score), "quad_proc": quad, "is_quad": is_quad, "ar": float(ar)}

    # ------------------------------------------------------------------ #
    # Tracking
    # ------------------------------------------------------------------ #
    def _update_track(self, corners: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """EMA-smooth corners; reset when the new detection is a jump (different note / new scene)."""
        corners = corners.astype(np.float32)
        if self.smooth_corners is not None:
            corners = _align_corners(self.smooth_corners, corners)
            old_bb = cv2.boundingRect(self.smooth_corners.astype(np.int32))
            new_bb = cv2.boundingRect(corners.astype(np.int32))
            # Genuine hand motion between consecutive frames keeps IoU > ~0.7; anything lower
            # (different note, note moved/zoomed abruptly) must restart the filter, not be blended.
            if _bbox_iou(tuple(map(float, old_bb)), tuple(map(float, new_bb))) < 0.50:
                self.smooth_corners = None
        if self.smooth_corners is None:
            self.smooth_corners = corners.copy()
        else:
            a = self.smoothing_alpha
            self.smooth_corners = (1.0 - a) * self.smooth_corners + a * corners
        x, y, w, h = cv2.boundingRect(self.smooth_corners.astype(np.int32))
        self.smooth_bbox = (float(x), float(y), float(w), float(h))
        return self.smooth_corners, (x, y, w, h)

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #
    def detect(
        self,
        bgr_frame: np.ndarray,
        is_pre_cropped: bool = False
    ) -> Dict[str, Union[bool, Tuple, np.ndarray, float, List]]:
        """
        Locate currency note in live camera frame while strictly rejecting faces.

        Returns a dict with (unchanged keys) detected, bbox, corners, cropped_note, confidence,
        strategy, default_crop, default_bbox, aspect_ratio_valid; plus
        `measured_aspect_ratio` (float, 0.0 when unknown).
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
                "default_bbox": (0, 0, 100, 100),
                "aspect_ratio_valid": False,
                "measured_aspect_ratio": 0.0,
            }

        h, w = bgr_frame.shape[:2]

        # Default central guided bounding box (75% width, standard ~2.15 aspect ratio)
        guide_w = int(w * 0.75)
        guide_h = min(int(h * 0.82), int(guide_w / 2.15))
        gx1 = max(0, (w - guide_w) // 2)
        gy1 = max(0, (h - guide_h) // 2)
        gx2 = min(w, gx1 + guide_w)
        gy2 = min(h, gy1 + guide_h)
        default_bbox = (gx1, gy1, gx2 - gx1, gy2 - gy1)
        default_crop = bgr_frame[gy1:gy2, gx1:gx2].copy()

        # Explicit pre-cropped override (e.g. synthetic bench, dataset crop, or uploaded note photo)
        if is_pre_cropped:
            self.smooth_bbox = (0.0, 0.0, float(w), float(h))
            self.smooth_corners = None
            self.last_raw_corners = None
            return {
                "detected": True,
                "bbox": (0, 0, w, h),
                "corners": [[0, 0], [w, 0], [w, h], [0, h]],
                "cropped_note": bgr_frame,
                "confidence": 0.98,
                "strategy": "full_banknote_frame",
                "default_crop": default_crop,
                "default_bbox": default_bbox,
                "aspect_ratio_valid": True,
                "measured_aspect_ratio": max(w, h) / float(max(1, min(w, h))),
            }

        # Scale down for fast processing
        proc_scale = self.proc_size / max(h, w)
        proc_w = max(8, int(w * proc_scale))
        proc_h = max(8, int(h * proc_scale))
        small = cv2.resize(bgr_frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # 1. Active human face detection
        detected_faces = self.detect_faces(gray)

        guide_proc = (
            int(gx1 * proc_scale), int(gy1 * proc_scale),
            int(gx2 * proc_scale), int(gy2 * proc_scale)
        )

        # 2. Candidate masks: adaptive edges + background difference
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = self._auto_canny(blurred)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilate_iters = 2
        edge_mask = cv2.dilate(edges, kernel, iterations=dilate_iters)
        # Each 5x5 dilation iteration grows the outline by 2 px; undo that when fitting the quad.
        edge_inset = 2.0 * dilate_iters
        masks = [("edges", edge_mask, cv2.RETR_TREE, edge_inset)]
        bg_mask = self._background_diff_mask(small)
        if bg_mask is not None:
            masks.append(("bgdiff", bg_mask, cv2.RETR_EXTERNAL, 0.0))

        best: Optional[Dict] = None
        for source, mask, mode, inset_px in masks:
            contours, _ = cv2.findContours(mask, mode, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:25]
            for cnt in contours:
                cand = self._evaluate_contour(cnt, small, gray, detected_faces, guide_proc, inset_px)
                if cand is not None and (best is None or cand["score"] > best["score"]):
                    best = cand
                    best["source"] = source

        # 3. Verified quad found
        if best is not None and best["score"] >= self.ACCEPT_SCORE:
            raw_corners = best["quad_proc"] / proc_scale
            raw_corners[:, 0] = np.clip(raw_corners[:, 0], 0, w - 1)
            raw_corners[:, 1] = np.clip(raw_corners[:, 1], 0, h - 1)
            self.last_raw_corners = raw_corners.copy()
            self.hold_counter = 0
            self.miss_counter = 0

            smooth, bbox = self._update_track(raw_corners)
            warped, meas_ar = self.warp_note(bgr_frame, raw_corners)
            if warped is None or warped.size == 0:
                bx, by, bw_, bh_ = bbox
                warped = bgr_frame[max(0, by):min(h, by + bh_), max(0, bx):min(w, bx + bw_)].copy()
                meas_ar = float(best["ar"])
            if warped is None or warped.size == 0:
                warped = default_crop

            return {
                "detected": True,
                "bbox": bbox,
                "corners": smooth.astype(np.int32).tolist(),
                "cropped_note": warped,
                "confidence": float(np.clip(best["score"] * 1.1, 0.60, 0.98)),
                "strategy": "contour_quad",
                "default_crop": default_crop,
                "default_bbox": default_bbox,
                "aspect_ratio_valid": True,
                "measured_aspect_ratio": float(meas_ar),
                "candidate_source": best.get("source", ""),
            }

        # 4. Short track-hold: ride through 1-3 dropped frames (motion blur, finger over edge)
        if self.last_raw_corners is not None and self.hold_counter < self.hold_frames:
            self.hold_counter += 1
            warped, meas_ar = self.warp_note(bgr_frame, self.last_raw_corners)
            if warped is not None and warped.size > 0 and self.smooth_corners is not None:
                x, y, bw_, bh_ = cv2.boundingRect(self.smooth_corners.astype(np.int32))
                return {
                    "detected": True,
                    "bbox": (x, y, bw_, bh_),
                    "corners": self.smooth_corners.astype(np.int32).tolist(),
                    "cropped_note": warped,
                    "confidence": 0.50,
                    "strategy": "track_hold",
                    "default_crop": default_crop,
                    "default_bbox": default_bbox,
                    "aspect_ratio_valid": True,
                    "measured_aspect_ratio": float(meas_ar),
                }
        elif self.hold_counter >= self.hold_frames:
            self.last_raw_corners = None
            self.smooth_corners = None
            self.hold_counter = 0

        # 5. Guided central reticle presence
        # (user aligned the note inside the visual reticle; fingers may break the outer contour)
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
                    "confidence": 0.62,
                    "strategy": "guided_reticle",
                    "default_crop": default_crop,
                    "default_bbox": default_bbox,
                    "aspect_ratio_valid": False,
                    "measured_aspect_ratio": 0.0,
                }

        # 6. Standby (empty frame, user face, or smooth clothes)
        self.miss_counter += 1
        if self.miss_counter > 6:
            self.smooth_bbox = None
            self.smooth_corners = None

        return {
            "detected": False,
            "bbox": default_bbox,
            "corners": None,
            "cropped_note": default_crop,
            "confidence": 0.0,
            "strategy": "standby",
            "default_crop": default_crop,
            "default_bbox": default_bbox,
            "aspect_ratio_valid": False,
            "measured_aspect_ratio": 0.0,
        }