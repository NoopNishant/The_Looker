"""
utils/visualizer.py - HUD, Telemetry, and Multi-Sensor Diagnostic Dashboard.

Renders:
1. Live camera viewport with bounding box and HUD status ribbons.
2. Dynamic verdict banner (GENUINE [Green] vs SUSPECT/COUNTERFEIT [Red]).
3. Diagnostic telemetry grid:
   - CLAHE micro-contrast preview
   - ESPCN Super-Resolved micro-texture inspection window
   - 2D FFT Power Spectrum rosette harmonic display
   - Optical screen-flash watermark transmission mask
   - CMYK ink separation thumbnail tiles
"""

from typing import Dict, Optional, Tuple, Union
import cv2
import numpy as np


class HUDVisualizer:
    """
    Renders telemetry overlays, ROI targets, and diagnostic multi-spectral tiles.
    """

    COLOR_GENUINE = (46, 204, 113)     # Vibrant Green
    COLOR_SUSPECT = (41, 41, 231)      # Vibrant Crimson Red
    COLOR_NEUTRAL = (220, 220, 220)    # Light gray
    COLOR_BG = (24, 24, 28)            # Dark slate panel background
    COLOR_ACCENT = (255, 170, 0)       # Cyan/Amber accent

    def __init__(self, panel_width: int = 340) -> None:
        self.panel_width = panel_width

    def draw_hud(
        self,
        frame: np.ndarray,
        results: Dict[str, Union[str, float, bool, np.ndarray, Dict]]
    ) -> np.ndarray:
        """
        Render real-time HUD and composite diagnostic sidebar onto the frame.
        """
        h, w = frame.shape[:2]

        # Optimize display scaling for high-resolution feeds (e.g. 1080p -> 720p display canvas)
        coord_scale_x = 1.0
        coord_scale_y = 1.0
        if h > 720 or w > 1280:
            scale_factor = 720.0 / h
            target_w = int(w * scale_factor)
            canvas = cv2.resize(frame, (target_w, 720), interpolation=cv2.INTER_AREA)
            coord_scale_x = target_w / float(w)
            coord_scale_y = 720.0 / float(h)
            h, w = canvas.shape[:2]
        else:
            canvas = frame.copy()

        # 1. Draw central ROI targeting reticle
        roi_bbox = results.get("roi_bbox")
        if roi_bbox:
            x1, y1, x2, y2 = roi_bbox
            x1 = int(x1 * coord_scale_x)
            y1 = int(y1 * coord_scale_y)
            x2 = int(x2 * coord_scale_x)
            y2 = int(y2 * coord_scale_y)
            is_counterfeit = results.get("is_counterfeit", False)
            target_color = self.COLOR_SUSPECT if is_counterfeit else self.COLOR_GENUINE

            # Draw corner brackets
            bracket_len = min(20, (x2 - x1) // 4)
            t = 2
            # Top-left
            cv2.line(canvas, (x1, y1), (x1 + bracket_len, y1), target_color, t)
            cv2.line(canvas, (x1, y1), (x1, y1 + bracket_len), target_color, t)
            # Top-right
            cv2.line(canvas, (x2, y1), (x2 - bracket_len, y1), target_color, t)
            cv2.line(canvas, (x2, y1), (x2, y1 + bracket_len), target_color, t)
            # Bottom-left
            cv2.line(canvas, (x1, y2), (x1 + bracket_len, y2), target_color, t)
            cv2.line(canvas, (x1, y2), (x1, y2 - bracket_len), target_color, t)
            # Bottom-right
            cv2.line(canvas, (x2, y2), (x2 - bracket_len, y2), target_color, t)
            cv2.line(canvas, (x2, y2), (x2, y2 - bracket_len), target_color, t)

            cv2.putText(
                canvas, "MICRO-TEXTURE ROI",
                (x1, max(15, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, target_color, 1, cv2.LINE_AA
            )

        # 2. Main Status Banner at the top of camera viewport
        denom = results.get("denomination", "---")
        conf = results.get("confidence", 0.0) * 100.0
        verdict = results.get("verdict", "INSPECTING")
        is_counterfeit = results.get("is_counterfeit", False)
        fps = results.get("fps", 0.0)
        latency = results.get("latency_ms", 0.0)

        # Semi-transparent header banner
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, 64), (16, 16, 20), -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

        # Denomination text
        cv2.putText(
            canvas, f"DENOMINATION: {denom} ({conf:.1f}%)",
            (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA
        )

        # Verdict text
        v_color = self.COLOR_SUSPECT if is_counterfeit else self.COLOR_GENUINE
        cv2.putText(
            canvas, f"VERDICT: {verdict}",
            (16, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.60, v_color, 2, cv2.LINE_AA
        )

        # Telemetry info on top-right
        cv2.putText(
            canvas, f"FPS: {fps:.1f} | Latency: {latency:.1f} ms",
            (w - 240, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA
        )

        audio_played = results.get("audio_played", False)
        audio_text = "AUDIO: [PULSE]" if audio_played else "AUDIO: [ACTIVE]"
        audio_col = (0, 220, 255) if audio_played else (140, 140, 140)
        cv2.putText(
            canvas, audio_text,
            (w - 240, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, audio_col, 1, cv2.LINE_AA
        )

        # 3. Create Diagnostic Inspection Sidebar
        panel = self._build_diagnostic_panel(canvas.shape[0], results)
        composite = np.hstack([canvas, panel])

        return composite

    def _build_diagnostic_panel(
        self,
        height: int,
        results: Dict[str, Union[str, float, bool, np.ndarray, Dict]]
    ) -> np.ndarray:
        """
        Build the dedicated multi-sensor telemetry panel.
        """
        panel = np.full((height, self.panel_width, 3), self.COLOR_BG, dtype=np.uint8)
        y = 20

        # Title
        cv2.putText(
            panel, "DIAGNOSTIC TELEMETRY", (14, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 200, 0), 2, cv2.LINE_AA
        )
        y += 18
        cv2.line(panel, (12, y), (self.panel_width - 12, y), (60, 60, 70), 1)
        y += 16

        # Metric gauges
        c_prob = results.get("composite_counterfeit_score", 0.0)
        h_score = results.get("halftone_score", 0.0)
        cnn_p = results.get("counterfeit_cnn_prob", 0.0)

        # Draw metrics
        y = self._draw_metric_bar(panel, "Counterfeit Score", c_prob, y, threshold=0.50)
        y = self._draw_metric_bar(panel, "CMYK Halftone Score", h_score, y, threshold=0.50)
        y = self._draw_metric_bar(panel, "CNN Counterfeit Prob", cnn_p, y, threshold=0.50)

        # Tiles Section:
        # Tile 1: ESPCN Super-Resolved Micro-Texture
        sr_roi = results.get("sr_roi")
        if sr_roi is not None and sr_roi.size > 0:
            tile_sr = cv2.resize(sr_roi, (130, 95), interpolation=cv2.INTER_AREA)
        else:
            tile_sr = np.zeros((95, 130, 3), dtype=np.uint8)

        # Tile 2: 2D FFT Rosette Spectrum
        halftone_metrics = results.get("halftone_metrics", {})
        vis_spec = halftone_metrics.get("vis_spectrum") if isinstance(halftone_metrics, dict) else None
        if vis_spec is not None and vis_spec.size > 0:
            spec_color = cv2.applyColorMap(vis_spec, cv2.COLORMAP_MAGMA)
            tile_fft = cv2.resize(spec_color, (130, 95), interpolation=cv2.INTER_AREA)
        else:
            tile_fft = np.zeros((95, 130, 3), dtype=np.uint8)

        # Render Tile Row 1
        y += 8
        cv2.putText(panel, "SUPER-RES ROI", (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
        cv2.putText(panel, "2D FFT SPECTRUM", (165, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
        y += 6

        if y + 95 < height:
            panel[y:y + 95, 14:144] = tile_sr
            panel[y:y + 95, 165:295] = tile_fft
            y += 105

        # Tile 3: CLAHE Topography
        clahe_bgr = results.get("clahe_bgr")
        if clahe_bgr is not None and clahe_bgr.size > 0:
            tile_clahe = cv2.resize(clahe_bgr, (130, 95), interpolation=cv2.INTER_AREA)
        else:
            tile_clahe = np.zeros((95, 130, 3), dtype=np.uint8)

        # Tile 4: Watermark Delta / Transmission Mask
        wm_data = results.get("watermark_delta")
        if wm_data and isinstance(wm_data, dict) and "watermark_mask" in wm_data:
            wm_mask = wm_data["watermark_mask"]
            wm_color = cv2.applyColorMap(wm_mask, cv2.COLORMAP_VIRIDIS)
            tile_wm = cv2.resize(wm_color, (130, 95), interpolation=cv2.INTER_AREA)
        else:
            tile_wm = np.zeros((95, 130, 3), dtype=np.uint8)
            cv2.putText(tile_wm, "NO FLASH CAP", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1)

        # Render Tile Row 2
        if y + 115 < height:
            cv2.putText(panel, "CLAHE TOPOGRAPHY", (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
            cv2.putText(panel, "WATERMARK DELTA", (165, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
            y += 6
            panel[y:y + 95, 14:144] = tile_clahe
            panel[y:y + 95, 165:295] = tile_wm
            y += 105

        # Hotkeys / Status Guide at bottom
        if y + 40 < height:
            y = height - 42
            cv2.putText(
                panel, "[SPACE]: Screen Flash | [Q]: Quit",
                (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1, cv2.LINE_AA
            )
            cv2.putText(
                panel, "[M]: Mute Audio | [S]: Snapshot",
                (14, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1, cv2.LINE_AA
            )

        return panel

    def _draw_metric_bar(
        self,
        panel: np.ndarray,
        label: str,
        value: float,
        y: int,
        threshold: float = 0.50
    ) -> int:
        """Helper to draw a styled progress gauge."""
        cv2.putText(
            panel, f"{label}: {value:.2f}",
            (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA
        )
        y += 5
        bar_w = self.panel_width - 32
        bar_h = 7
        cv2.rectangle(panel, (14, y), (14 + bar_w, y + bar_h), (45, 45, 55), -1)

        fill_w = int(bar_w * np.clip(value, 0.0, 1.0))
        fill_col = self.COLOR_SUSPECT if value >= threshold else self.COLOR_GENUINE
        if fill_w > 0:
            cv2.rectangle(panel, (14, y), (14 + fill_w, y + bar_h), fill_col, -1)

        # Draw threshold marker
        thresh_x = 14 + int(bar_w * threshold)
        cv2.line(panel, (thresh_x, y - 2), (thresh_x, y + bar_h + 2), (255, 255, 255), 1)

        return y + bar_h + 12
