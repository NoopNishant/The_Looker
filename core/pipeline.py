"""
core/pipeline.py - Unified Real-Time Currency Recognition & Counterfeit Detection Engine.

Integrates:
- BanknoteDetector for real-time automatic banknote contour localization in live video
  (now returns a perspective-rectified note crop and its measured aspect ratio).
- DenominationDeductionEngine for multi-cue optical & deep feature denomination deduction.
- MultiTaskMobileNetV3 for deep semantic classification and counterfeit probability.
- CLAHE for optical topology enhancement.
- ScreenFlashDifferencing for active optical transmission inspection.
- ESPCN SuperResolution + HalftoneDotDetector for micro-texture CMYK dot discrimination.
- DiscreetAudioEngine for non-visual acoustic feedback.
- TemporalStabilizer (opt-in via process_frame(..., live=True)) that votes the denomination over
  recent frames and adds hysteresis to the counterfeit verdict, so a live feed does not flicker.
"""

import os
import time
from collections import deque
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import cv2

try:
    import torch
    import torchvision.transforms as T
    from .model import MultiTaskMobileNetV3
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    T = None
    MultiTaskMobileNetV3 = None

from .model import OpenCVDNNMobileNetV3, DEFAULT_DENOMINATIONS
from .preprocessing import CLAHEEnhancer, ScreenFlashDifferencing
from .super_res_halftone import SuperResolutionModule, HalftoneDotDetector
from .audio_feedback import DiscreetAudioEngine
from .detector import BanknoteDetector
from .denomination_engine import DenominationDeductionEngine
from .security_validator import BanknoteSecurityValidator


class TemporalStabilizer:
    """
    Frame-to-frame stabiliser for live video.

    * Denomination: recency-weighted, confidence-weighted vote over the last `window` frames.
      The incumbent is only replaced when a challenger's vote beats it by `switch_ratio`.
    * Counterfeit flag: EMA of the per-frame boolean with on/off hysteresis. A hard marker hit
      (known toy-note text) latches immediately.
    * A note that disappears for more than `miss_tolerance` frames resets all state.
    """

    def __init__(
        self,
        window: int = 8,
        switch_ratio: float = 1.25,
        miss_tolerance: int = 4,
        flag_alpha: float = 0.5,
        flag_on: float = 0.6,
        flag_off: float = 0.3,
    ) -> None:
        self.window = window
        self.switch_ratio = switch_ratio
        self.miss_tolerance = miss_tolerance
        self.flag_alpha = flag_alpha
        self.flag_on = flag_on
        self.flag_off = flag_off
        self.reset()

    def reset(self) -> None:
        self.history: deque = deque(maxlen=self.window)
        self.stable_denom: Optional[str] = None
        self.stable_frames: int = 0
        self.flag_ema: float = 0.0
        self.cf_state: bool = False
        self.last_reason: str = ""
        self.misses: int = 0

    def miss(self) -> None:
        self.misses += 1
        if self.misses > self.miss_tolerance:
            self.reset()

    def update(
        self,
        denom: str,
        conf: float,
        cf_flag: bool,
        hard_cf: bool = False,
        reason: str = ""
    ) -> Dict[str, Union[str, float, bool, int]]:
        self.misses = 0
        self.history.append((denom, max(float(conf), 0.05)))
        n = len(self.history)

        totals: Dict[str, float] = {}
        for i, (d, c) in enumerate(self.history):
            totals[d] = totals.get(d, 0.0) + c * (0.6 + 0.4 * (i + 1) / n)
        leader = max(totals, key=totals.get)

        if self.stable_denom is None or self.stable_denom not in totals:
            self.stable_denom, self.stable_frames = leader, 1
        elif leader == self.stable_denom:
            self.stable_frames += 1
        elif totals[leader] >= self.switch_ratio * totals[self.stable_denom]:
            self.stable_denom, self.stable_frames = leader, 1
        else:
            self.stable_frames += 1  # incumbent holds

        agree = [c for d, c in self.history if d == self.stable_denom]
        share = totals[self.stable_denom] / max(1e-9, sum(totals.values()))
        conf_out = float(np.clip(np.mean(agree) * (0.6 + 0.4 * share), 0.0, 0.99))

        if hard_cf:
            self.cf_state = True
            self.flag_ema = 1.0
        else:
            self.flag_ema = self.flag_alpha * float(cf_flag) + (1.0 - self.flag_alpha) * self.flag_ema
            if self.cf_state and self.flag_ema <= self.flag_off:
                self.cf_state = False
            elif (not self.cf_state) and self.flag_ema >= self.flag_on:
                self.cf_state = True
        if cf_flag and reason:
            self.last_reason = reason

        return {
            "denomination": self.stable_denom,
            "confidence": conf_out,
            "is_counterfeit": self.cf_state,
            "stable_frames": self.stable_frames,
        }


class CurrencyInspectionPipeline:
    """
    End-to-end CPU-optimized currency recognition and multi-layer counterfeit detection pipeline.
    Supports ultra-lightweight OpenCV DNN runtime (<65MB bundle) with seamless PyTorch fallback.
    """

    BACKGROUND_GATE = 0.35

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        enable_audio: bool = True,
        counterfeit_threshold: float = 0.50,
        roi_size: int = 96,
        tta: bool = False
    ) -> None:
        self.device = torch.device(device) if (TORCH_AVAILABLE and torch) else "cpu"
        self.counterfeit_threshold = counterfeit_threshold
        self.roi_size = roi_size
        self.tta = tta  # average CNN output over the image and its mirror (2x CNN cost)

        # 1. Initialize Banknote Detector (Contour & Boundary Segmentation)
        self.detector = BanknoteDetector()

        # 2. Initialize Denomination Deduction Engine
        self.denom_engine = DenominationDeductionEngine()

        # 3. Check for ONNX model first (pure OpenCV DNN runtime, zero PyTorch overhead)
        # NOTE: if an ONNX file exists it takes precedence over the .pth weights. After retraining,
        # make sure the ONNX is re-exported (train_currency_model.py now does this automatically).
        onnx_candidates = [
            "models/currency_mobilenet_v3.onnx",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "currency_mobilenet_v3.onnx")
        ]
        onnx_path = next((p for p in onnx_candidates if os.path.exists(p)), None)

        self.dnn_model: Optional[OpenCVDNNMobileNetV3] = None
        self.use_onnx = False

        if onnx_path:
            try:
                self.dnn_model = OpenCVDNNMobileNetV3(onnx_path)
                self.use_onnx = True
                print(f"[Pipeline] Loaded ONNX model via OpenCV DNN: {onnx_path} (Zero PyTorch overhead)")
            except Exception as e:
                print(f"[Pipeline] Warning: Could not load ONNX model via OpenCV DNN: {e}")

        # Fallback to PyTorch model if ONNX was not loaded and PyTorch is available
        self.model = None
        if not self.use_onnx and TORCH_AVAILABLE and MultiTaskMobileNetV3 is not None:
            try:
                self.model = MultiTaskMobileNetV3(
                    num_denominations=len(DEFAULT_DENOMINATIONS),
                    denominations=DEFAULT_DENOMINATIONS,
                    pretrained=False
                ).to(self.device)
                self.model.eval()

                if model_path is None:
                    default_weights = "models/currency_mobilenet_v3.pth"
                    if os.path.exists(default_weights):
                        model_path = default_weights
                    else:
                        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                        root_weights = os.path.join(repo_root, "models", "currency_mobilenet_v3.pth")
                        if os.path.exists(root_weights):
                            model_path = root_weights

                if model_path and os.path.exists(model_path):
                    state_dict = torch.load(model_path, map_location=self.device)
                    self.model.load_state_dict(state_dict)
                    print(f"[Pipeline] Successfully loaded PyTorch weights from {model_path}")
            except Exception as e:
                print(f"[Pipeline] Warning: Could not load PyTorch model from {model_path}: {e}")

        # Preprocessing transform for PyTorch
        if TORCH_AVAILABLE and T is not None:
            self.tensor_transform = T.Compose([
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.tensor_transform = None

        # 4. Optical Preprocessing Modules
        self.clahe_enhancer = CLAHEEnhancer(clip_limit=2.5, tile_grid_size=(8, 8))
        self.flash_differencer = ScreenFlashDifferencing(flash_duration_ms=180)

        # 5. Super-Resolution & Micro-Texture Halftone Detector
        self.super_res = SuperResolutionModule(scale_factor=2, device=device)
        self.halftone_detector = HalftoneDotDetector(fft_peak_threshold=16.0, dot_density_threshold=1.0)
        self.security_validator = BanknoteSecurityValidator()

        # 6. Discreet Audio Engine
        self.audio_engine = DiscreetAudioEngine(debounce_interval_sec=1.5, enabled=enable_audio)

        # 7. Temporal stabiliser (used only when process_frame(..., live=True))
        self.stabilizer = TemporalStabilizer()

        # Performance telemetry
        self.fps_rolling: float = 0.0
        self.frame_count: int = 0
        self.prev_time: float = time.time()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_preview(img: np.ndarray, max_h: int = 360, max_w: int = 640) -> np.ndarray:
        """Downscale (never upscale) so the image fits within max_w x max_h, keeping aspect ratio."""
        nh, nw = img.shape[:2]
        scale = min(1.0, max_h / float(max(1, nh)), max_w / float(max(1, nw)))
        if scale >= 1.0:
            return img
        return cv2.resize(img, (max(1, int(nw * scale)), max(1, int(nh * scale))), interpolation=cv2.INTER_AREA)

    def _cnn_once(self, bgr_img: np.ndarray) -> Tuple[Optional[Dict[str, float]], float]:
        if self.use_onnx and self.dnn_model is not None:
            out = self.dnn_model.predict(bgr_img)
            return out["probs"], float(out["counterfeit_prob"])
        if self.model is not None and TORCH_AVAILABLE and self.tensor_transform is not None:
            rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            tensor_in = self.tensor_transform(rgb).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                nn_output = self.model.forward(tensor_in)
                probs = torch.softmax(nn_output["denomination_logits"][0], dim=-1)
                probs_dict = {
                    self.model.denominations[i]: probs[i].item()
                    for i in range(min(len(self.model.denominations), probs.shape[0]))
                }
                return probs_dict, float(nn_output["counterfeit_prob"][0].item())
        return None, 0.0

    def _run_cnn(self, bgr_img: np.ndarray) -> Tuple[Optional[Dict[str, float]], float]:
        """Run the CNN (ONNX or PyTorch), optionally averaged with its mirror image (TTA)."""
        probs, cf = self._cnn_once(bgr_img)
        if probs is None or not self.tta:
            return probs, cf
        probs_f, cf_f = self._cnn_once(cv2.flip(bgr_img, 1))
        if probs_f is None:
            return probs, cf
        merged = {k: 0.5 * (probs[k] + probs_f.get(k, probs[k])) for k in probs}
        return merged, 0.5 * (cf + cf_f)

    def extract_central_roi(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Extract high-resolution region of interest (ROI) from center of currency note
        for micro-texture CMYK dot inspection.
        Returns:
            cropped_roi, (x1, y1, x2, y2)
        """
        h, w = frame.shape[:2]
        size = min(self.roi_size, h, w)
        cy, cx = h // 2, w // 2
        x1 = max(0, cx - size // 2)
        y1 = max(0, cy - size // 2)
        x2 = min(w, x1 + size)
        y2 = min(h, y1 + size)
        roi = frame[y1:y2, x1:x2].copy()
        return roi, (x1, y1, x2, y2)

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #
    def process_frame(
        self,
        bgr_frame: np.ndarray,
        cached_flash_delta: Optional[Dict[str, np.ndarray]] = None,
        roi_bbox: Optional[Tuple[int, int, int, int]] = None,
        is_pre_cropped: bool = False,
        live: bool = False
    ) -> Dict[str, Union[str, float, bool, np.ndarray, Dict]]:
        """
        Execute full inspection pass on incoming camera frame.

        Args:
            bgr_frame: Raw 1080p or preview BGR frame.
            cached_flash_delta: Optional pre-computed screen-flash difference result.
            roi_bbox: Optional custom ROI coordinates (x1, y1, x2, y2).
            is_pre_cropped: If True, treats frame as pre-cropped banknote image.
            live: Set True from the continuous camera loop / WebSocket stream to enable temporal
                  voting + verdict hysteresis. Leave False for single images (uploads, tests),
                  where state from a previous, unrelated image must not leak in.
        Returns:
            Comprehensive diagnostic dictionary.
        """
        t_start = time.perf_counter()

        h, w = bgr_frame.shape[:2]

        # Step 0: Auto-Detect Banknote (stateless for single photos, smoothed for live video)
        if not live:
            self.detector.reset()
        detection = self.detector.detect(bgr_frame, is_pre_cropped=is_pre_cropped)
        is_note_detected = detection["detected"]
        detected_bbox = detection["bbox"]
        corners = detection.get("corners")
        note_crop = detection["cropped_note"] if (is_note_detected and detection["cropped_note"] is not None) else bgr_frame

        # Step 1: Extract micro-texture ROI from the segmented banknote crop
        if roi_bbox is not None:
            x1, y1, x2, y2 = roi_bbox
            roi = bgr_frame[y1:y2, x1:x2].copy()
        else:
            roi, _ = self.extract_central_roi(note_crop)

        # Step 2: Optical CLAHE Contrast Enhancement
        preview_frame = self._to_preview(note_crop)
        clahe_bgr = self.clahe_enhancer.enhance_bgr(preview_frame)
        topological_grad = self.clahe_enhancer.extract_topological_gradients(preview_frame)

        # Step 3: Multi-Task CNN Inference (OpenCV DNN or PyTorch)
        cnn_probs_dict, counterfeit_cnn_prob = self._run_cnn(preview_frame)

        # Step 4: Calibrated Multi-Cue Denomination Deduction
        ar_valid = bool(detection.get("aspect_ratio_valid", detection.get("strategy") == "contour_quad"))
        deduction = self.denom_engine.deduce_denomination(note_crop, cnn_probs=cnn_probs_dict, aspect_ratio_valid=ar_valid)
        denomination = deduction["denomination"]
        confidence = deduction["confidence"]
        is_currency = deduction["is_valid_currency"]
        note_name = deduction.get("note_name", denomination)

        # Strict Background Gate with Guided Reticle & Full-Frame Fallbacks:
        bg_prob = cnn_probs_dict.get("Background", 0.0) if cnn_probs_dict else 0.0
        if bg_prob >= self.BACKGROUND_GATE:
            # 1) If a false contour was picked up, check the central guided reticle
            # 2) If guided_reticle cropped too tightly into a pre-cropped banknote, check full frame
            fallback_candidates = []
            reticle_crop = detection.get("default_crop")
            if (
                detection.get("strategy") in ("contour_quad", "track_hold")
                and reticle_crop is not None and reticle_crop.size > 0
            ):
                fallback_candidates.append((reticle_crop, detection.get("default_bbox", (0, 0, w, h)), False))
            if not is_pre_cropped and detection.get("strategy") in ("guided_reticle", "contour_quad"):
                frame_ar = max(w, h) / max(1.0, min(w, h))
                if 1.65 <= frame_ar <= 2.75:
                    fallback_candidates.append((bgr_frame, (0, 0, w, h), True))

            for fb_crop, fb_bbox, fb_ar_valid in fallback_candidates:
                cnn_probs_r, cf_cnn_r = self._run_cnn(self._to_preview(fb_crop))
                if cnn_probs_r and cnn_probs_r.get("Background", 0.0) < self.BACKGROUND_GATE:
                    note_crop = fb_crop
                    is_note_detected = True
                    detected_bbox = fb_bbox
                    corners = [
                        [fb_bbox[0], fb_bbox[1]],
                        [fb_bbox[0] + fb_bbox[2], fb_bbox[1]],
                        [fb_bbox[0] + fb_bbox[2], fb_bbox[1] + fb_bbox[3]],
                        [fb_bbox[0], fb_bbox[1] + fb_bbox[3]]
                    ]
                    cnn_probs_dict = cnn_probs_r
                    counterfeit_cnn_prob = float(cf_cnn_r)
                    deduction = self.denom_engine.deduce_denomination(note_crop, cnn_probs=cnn_probs_dict, aspect_ratio_valid=fb_ar_valid)
                    denomination = deduction["denomination"]
                    confidence = deduction["confidence"]
                    is_currency = deduction["is_valid_currency"]
                    note_name = deduction.get("note_name", denomination)
                    bg_prob = cnn_probs_r.get("Background", 0.0)
                    roi, _ = self.extract_central_roi(note_crop)
                    break

        if bg_prob >= self.BACKGROUND_GATE:
            is_currency = False
            denomination = "Background"
            note_name = "No Currency in View"

        # Step 5: Super-Resolution & CMYK Halftone Analysis
        sr_roi = self.super_res.upscale_roi(roi)
        halftone_metrics = self.halftone_detector.analyze_microtexture(sr_roi)
        halftone_score = halftone_metrics["halftone_score"]
        is_halftone = halftone_metrics["is_halftone_detected"]

        # Step 6: Banknote Security Feature & Counterfeit Marker Validation
        sec_results = self.security_validator.check_counterfeit_markers(note_crop)
        is_toy_counterfeit = sec_results["is_toy_counterfeit"]
        flagged_marker = sec_results["flagged_marker"]
        marker_score = sec_results["max_marker_score"]

        # Multi-Signal Counterfeit & Authenticity Deduction
        watermark_penalty = 0.0
        if cached_flash_delta is not None and "transmission_density" in cached_flash_delta:
            density = cached_flash_delta["transmission_density"]
            if density < 0.015:
                watermark_penalty = 0.35

        # Composite score calculation
        if is_toy_counterfeit:
            composite_counterfeit_score = float(max(0.96, marker_score))
            counterfeit_cnn_prob = max(counterfeit_cnn_prob, 0.95)
        else:
            composite_counterfeit_score = float(
                0.30 * counterfeit_cnn_prob +
                0.55 * halftone_score +
                0.15 * min(1.0, watermark_penalty)
            )

        # Determine Verdict & Detailed Reason
        note_valid = bool(is_note_detected and is_currency and denomination != "Background")
        stable_frames = 0

        if not note_valid:
            if live:
                self.stabilizer.miss()
            is_note_detected = False
            verdict = "AWAITING CURRENCY NOTE"
            is_counterfeit = False
            denomination = "Background"
            note_name = "No Currency in View"
            authenticity_reason = "Position banknote within the central camera reticle"
            confidence = 0.0
            composite_counterfeit_score = 0.0
            halftone_score = 0.0
            counterfeit_cnn_prob = 0.0
        else:
            raw_flag = False
            hard_flag = False
            raw_reason = ""
            if is_toy_counterfeit:
                raw_flag = hard_flag = True
                raw_reason = f"Counterfeit Marker Detected ({flagged_marker})"
            elif is_halftone:
                raw_flag = True
                raw_reason = "CMYK Halftone Printer Rosettes Detected (Dither Dots)"
            elif composite_counterfeit_score >= self.counterfeit_threshold:
                raw_flag = True
                raw_reason = "Unnatural Surface Relief & Optical Spectrum Mismatch"

            if live:
                st = self.stabilizer.update(denomination, confidence, raw_flag, hard_flag, raw_reason)
                denomination = str(st["denomination"])
                confidence = float(st["confidence"])
                is_counterfeit = bool(st["is_counterfeit"])
                stable_frames = int(st["stable_frames"])
                note_name = self.denom_engine.PROFILES.get(denomination, {}).get("name", denomination)
                authenticity_reason = raw_reason or self.stabilizer.last_reason
            else:
                self.stabilizer.reset()
                is_counterfeit = raw_flag
                authenticity_reason = raw_reason

            if is_counterfeit:
                verdict = "SUSPECT / COUNTERFEIT"
                if not authenticity_reason:
                    authenticity_reason = "Unnatural Surface Relief & Optical Spectrum Mismatch"
            else:
                verdict = "GENUINE"
                authenticity_reason = "Authentic Continuous-Line Intaglio Engraving Verified"

        # Step 7: Discreet Audio Feedback Trigger
        # (in live mode, wait for a stable read so a one-frame mis-read never beeps the wrong note)
        audio_played = False
        if (
            note_valid and denomination != "Background" and confidence >= 0.45
            and (not live or stable_frames >= 3)
        ):
            audio_played = self.audio_engine.trigger_feedback(
                denomination=denomination,
                is_counterfeit=is_counterfeit
            )

        # Telemetry & Frame Rate computation
        t_elapsed = time.perf_counter() - t_start
        fps_instant = 1.0 / max(1e-5, t_elapsed)
        self.fps_rolling = 0.9 * self.fps_rolling + 0.1 * fps_instant if self.fps_rolling > 0 else fps_instant
        self.frame_count += 1

        return {
            "is_note_detected": is_note_detected,
            "denomination": denomination,
            "note_name": note_name,
            "confidence": confidence,
            "denomination_uncertain": bool(note_valid and (confidence < 0.30 or deduction.get("ambiguous", False))),
            "denomination_scores": deduction.get("scores", {}),
            "counterfeit_cnn_prob": counterfeit_cnn_prob,
            "halftone_score": halftone_score,
            "composite_counterfeit_score": composite_counterfeit_score,
            "is_counterfeit": is_counterfeit,
            "verdict": verdict,
            "authenticity_reason": authenticity_reason,
            "stable_frames": stable_frames,
            "detection_strategy": detection.get("strategy", ""),
            "aspect_ratio": float(detection.get("measured_aspect_ratio", 0.0)),
            "fps": self.fps_rolling,
            "latency_ms": t_elapsed * 1000.0,
            "roi_bbox": detected_bbox if is_note_detected else (w//4, h//4, w//2, h//2),
            "corners": corners,
            "raw_roi": roi,
            "sr_roi": sr_roi,
            "clahe_bgr": clahe_bgr,
            "topological_grad": topological_grad,
            "halftone_metrics": halftone_metrics,
            "watermark_delta": cached_flash_delta,
            "audio_played": audio_played
        }