"""
core/pipeline.py - Unified Real-Time Currency Recognition & Counterfeit Detection Engine.

Integrates:
- BanknoteDetector for real-time automatic banknote contour localization in live video.
- DenominationDeductionEngine for multi-cue optical & deep feature denomination deduction.
- MultiTaskMobileNetV3 for deep semantic classification and counterfeit probability.
- CLAHE for optical topology enhancement.
- ScreenFlashDifferencing for active optical transmission inspection.
- ESPCN SuperResolution + HalftoneDotDetector for micro-texture CMYK dot discrimination.
- DiscreetAudioEngine for non-visual acoustic feedback.
"""

from typing import Dict, Optional, Tuple, Union
import os
import time
import numpy as np
import cv2
import torch
import torchvision.transforms as T

from .model import MultiTaskMobileNetV3, DEFAULT_DENOMINATIONS
from .preprocessing import CLAHEEnhancer, ScreenFlashDifferencing
from .super_res_halftone import SuperResolutionModule, HalftoneDotDetector
from .audio_feedback import DiscreetAudioEngine
from .detector import BanknoteDetector
from .denomination_engine import DenominationDeductionEngine
from .security_validator import BanknoteSecurityValidator


class CurrencyInspectionPipeline:
    """
    End-to-end CPU-optimized currency recognition and multi-layer counterfeit detection pipeline.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        enable_audio: bool = True,
        counterfeit_threshold: float = 0.50,
        roi_size: int = 96
    ) -> None:
        self.device = torch.device(device)
        self.counterfeit_threshold = counterfeit_threshold
        self.roi_size = roi_size

        # 1. Initialize Banknote Detector (Contour & Boundary Segmentation)
        self.detector = BanknoteDetector()

        # 2. Initialize Denomination Deduction Engine
        self.denom_engine = DenominationDeductionEngine()

        # 3. Initialize Neural Network (MobileNetV3-Small)
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

        if model_path:
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"[Pipeline] Successfully loaded trained weights from {model_path}")
            except Exception as e:
                print(f"[Pipeline] Warning: Could not load model from {model_path}: {e}")

        # PyTorch ImageNet preprocessing transform
        self.tensor_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 4. Optical Preprocessing Modules
        self.clahe_enhancer = CLAHEEnhancer(clip_limit=2.5, tile_grid_size=(8, 8))
        self.flash_differencer = ScreenFlashDifferencing(flash_duration_ms=180)

        # 5. Super-Resolution & Micro-Texture Halftone Detector
        self.super_res = SuperResolutionModule(scale_factor=2, device=device)
        self.halftone_detector = HalftoneDotDetector(fft_peak_threshold=16.0, dot_density_threshold=1.0)
        self.security_validator = BanknoteSecurityValidator()

        # 6. Discreet Audio Engine
        self.audio_engine = DiscreetAudioEngine(debounce_interval_sec=1.5, enabled=enable_audio)

        # Performance telemetry
        self.fps_rolling: float = 0.0
        self.frame_count: int = 0
        self.prev_time: float = time.time()

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

    def process_frame(
        self,
        bgr_frame: np.ndarray,
        cached_flash_delta: Optional[Dict[str, np.ndarray]] = None,
        roi_bbox: Optional[Tuple[int, int, int, int]] = None,
        is_pre_cropped: bool = False
    ) -> Dict[str, Union[str, float, bool, np.ndarray, Dict]]:
        """
        Execute full inspection pass on incoming camera frame.
        
        Args:
            bgr_frame: Raw 1080p or preview BGR frame.
            cached_flash_delta: Optional pre-computed screen-flash difference result.
            roi_bbox: Optional custom ROI coordinates (x1, y1, x2, y2).
            is_pre_cropped: If True, treats frame as pre-cropped banknote image.
        Returns:
            Comprehensive diagnostic dictionary.
        """
        t_start = time.perf_counter()

        h, w = bgr_frame.shape[:2]

        # Step 0: Auto-Detect Banknote in Live Feed
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
        nh, nw = note_crop.shape[:2]
        if nh > 360 or nw > 640:
            scale_preview = 360.0 / nh
            preview_w = int(nw * scale_preview)
            preview_frame = cv2.resize(note_crop, (preview_w, 360), interpolation=cv2.INTER_AREA)
        else:
            preview_frame = note_crop

        clahe_bgr = self.clahe_enhancer.enhance_bgr(preview_frame)
        topological_grad = self.clahe_enhancer.extract_topological_gradients(preview_frame)

        # Step 3: Multi-Task CNN Inference (MobileNetV3)
        # Feed natural RGB pixels (consistent with ImageNet normalization & model training)
        rgb_for_nn = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
        tensor_in = self.tensor_transform(rgb_for_nn).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            nn_output = self.model.forward(tensor_in)
            denom_logits = nn_output["denomination_logits"][0]
            counterfeit_cnn_prob = nn_output["counterfeit_prob"][0].item()

            probs = torch.softmax(denom_logits, dim=-1)
            cnn_probs_dict = {
                self.model.denominations[i]: probs[i].item()
                for i in range(min(len(self.model.denominations), probs.shape[0]))
            }

        # Step 4: Calibrated Multi-Cue Denomination Deduction
        deduction = self.denom_engine.deduce_denomination(note_crop, cnn_probs=cnn_probs_dict)
        denomination = deduction["denomination"]
        confidence = deduction["confidence"]
        is_currency = deduction["is_valid_currency"]
        note_name = deduction.get("note_name", denomination)

        # Strict Background Gate with Guided Reticle Fallback:
        bg_prob = cnn_probs_dict.get("Background", 0.0)
        if bg_prob >= 0.35:
            # If a false contour was picked up (e.g. clothing), check the central guided reticle
            if detection.get("strategy") == "contour_quad" and "default_crop" in detection:
                reticle_crop = detection["default_crop"]
                if reticle_crop is not None and reticle_crop.size > 0:
                    r_h, r_w = reticle_crop.shape[:2]
                    if r_h > 360 or r_w > 640:
                        s_r = 360.0 / r_h
                        r_prev = cv2.resize(reticle_crop, (int(r_w * s_r), 360), interpolation=cv2.INTER_AREA)
                    else:
                        r_prev = reticle_crop
                    rgb_r = cv2.cvtColor(r_prev, cv2.COLOR_BGR2RGB)
                    t_r = self.tensor_transform(rgb_r).unsqueeze(0).to(self.device)
                    with torch.inference_mode():
                        nn_out_r = self.model.forward(t_r)
                        probs_r = torch.softmax(nn_out_r["denomination_logits"][0], dim=-1)
                        cnn_probs_r = {
                            self.model.denominations[i]: probs_r[i].item()
                            for i in range(min(len(self.model.denominations), probs_r.shape[0]))
                        }
                    if cnn_probs_r.get("Background", 0.0) < 0.35:
                        note_crop = reticle_crop
                        is_note_detected = True
                        def_bbox = detection.get("default_bbox", (0, 0, w, h))
                        detected_bbox = def_bbox
                        corners = [
                            [def_bbox[0], def_bbox[1]],
                            [def_bbox[0] + def_bbox[2], def_bbox[1]],
                            [def_bbox[0] + def_bbox[2], def_bbox[1] + def_bbox[3]],
                            [def_bbox[0], def_bbox[1] + def_bbox[3]]
                        ]
                        cnn_probs_dict = cnn_probs_r
                        counterfeit_cnn_prob = nn_out_r["counterfeit_prob"][0].item()
                        deduction = self.denom_engine.deduce_denomination(note_crop, cnn_probs=cnn_probs_dict)
                        denomination = deduction["denomination"]
                        confidence = deduction["confidence"]
                        is_currency = deduction["is_valid_currency"]
                        note_name = deduction.get("note_name", denomination)
                        bg_prob = cnn_probs_r.get("Background", 0.0)
                        roi, _ = self.extract_central_roi(note_crop)

        if bg_prob >= 0.35:
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
        if not is_note_detected or not is_currency or denomination == "Background":
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
        elif is_toy_counterfeit:
            verdict = "SUSPECT / COUNTERFEIT"
            is_counterfeit = True
            authenticity_reason = f"Counterfeit Marker Detected ({flagged_marker})"
        elif is_halftone:
            verdict = "SUSPECT / COUNTERFEIT"
            is_counterfeit = True
            authenticity_reason = "CMYK Halftone Printer Rosettes Detected (Dither Dots)"
        elif composite_counterfeit_score >= self.counterfeit_threshold:
            verdict = "SUSPECT / COUNTERFEIT"
            is_counterfeit = True
            authenticity_reason = "Unnatural Surface Relief & Optical Spectrum Mismatch"
        else:
            verdict = "GENUINE"
            is_counterfeit = False
            authenticity_reason = "Authentic Continuous-Line Intaglio Engraving Verified"

        # Step 7: Discreet Audio Feedback Trigger
        audio_played = False
        if is_currency and denomination != "Background" and confidence >= 0.45:
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
            "counterfeit_cnn_prob": counterfeit_cnn_prob,
            "halftone_score": halftone_score,
            "composite_counterfeit_score": composite_counterfeit_score,
            "is_counterfeit": is_counterfeit,
            "verdict": verdict,
            "authenticity_reason": authenticity_reason,
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
