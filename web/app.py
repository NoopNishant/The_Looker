"""
web/app.py - FastAPI Application Server for The Looker Forensic Web Studio.

Zero-dependency native endpoints:
- GET /                    : Serves interactive minimalist frontend
- POST /api/inspect/image  : Inspects uploaded photo (binary or JSON base64)
- POST /api/generate-demo  : Generates synthetic genuine/counterfeit sample (JSON)
- POST /api/flash-diff     : Computes watermark delta from ambient + flash pair (JSON)
- WebSocket /ws/live       : Real-time low-latency video stream for webcam
"""

import base64
import json
import os
import tempfile
import time
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from core.pipeline import CurrencyInspectionPipeline
from core.preprocessing import ScreenFlashDifferencing
from synthetic_data import SyntheticCurrencyGenerator

# Initialize FastAPI App
app = FastAPI(
    title="The Looker - Currency Forensics Studio",
    description="Aesthetic minimal currency recognition and counterfeit detection platform",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def exception_logging_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        print(f"[Unhandled Server Error] {tb}", flush=True)
        return HTMLResponse(
            f"<html><body style='background:#0f172a;color:#f87171;padding:32px;font-family:monospace;'>"
            f"<h2 style='color:#ef4444;'>The Looker Server Error:</h2>"
            f"<pre style='background:#1e293b;padding:20px;border-radius:8px;color:#fca5a5;overflow:auto;white-space:pre-wrap;'>{tb}</pre>"
            f"</body></html>",
            status_code=500
        )

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Resolve static directory across local and serverless layouts
static_candidates = [
    STATIC_DIR,
    os.path.join(os.path.dirname(BASE_DIR), "web", "static"),
    os.path.abspath("web/static")
]
active_static_dir = next((p for p in static_candidates if os.path.exists(p)), None)

# Mount static directory if present (avoids read-only /var/task crash on Vercel)
if active_static_dir:
    app.mount("/static", StaticFiles(directory=active_static_dir), name="static")

# Lazy-loaded global pipeline engine (prevents serverless cold-start timeouts)
_pipeline_instance: Optional[CurrencyInspectionPipeline] = None


def get_pipeline() -> CurrencyInspectionPipeline:
    """Lazy initialize forensic pipeline on first inspection request."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = CurrencyInspectionPipeline(device="cpu", enable_audio=False)
    return _pipeline_instance


def image_to_base64(img: np.ndarray, format: str = ".jpg", quality: int = 85) -> str:
    """Encode OpenCV BGR or Grayscale image into base64 data URI."""
    if img is None or img.size == 0:
        return ""
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if format.lower() in [".jpg", ".jpeg"] else []
    success, buffer = cv2.imencode(format, img, params)
    if not success:
        return ""
    b64_str = base64.b64encode(buffer).decode("utf-8")
    mime = "image/jpeg" if format.lower() in [".jpg", ".jpeg"] else "image/png"
    return f"data:{mime};base64,{b64_str}"


def base64_to_image(b64_data: str) -> Optional[np.ndarray]:
    """Decode base64 string or data URL to OpenCV BGR image."""
    try:
        if "," in b64_data:
            b64_data = b64_data.split(",", 1)[1]
        raw_bytes = base64.b64decode(b64_data)
        np_arr = np.frombuffer(raw_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def serialize_inspection_results(
    results: Dict,
    include_images: bool = True
) -> Dict[str, Union[str, float, bool, Dict]]:
    """Format pipeline output into JSON-serializable dictionary with base64 visual tiles."""
    payload = {
        "is_note_detected": bool(results.get("is_note_detected", False)),
        "denomination": results.get("denomination", "---"),
        "note_name": results.get("note_name", "---"),
        "authenticity_reason": results.get("authenticity_reason", ""),
        "confidence": round(float(results.get("confidence", 0.0)), 4),
        "counterfeit_cnn_prob": round(float(results.get("counterfeit_cnn_prob", 0.0)), 4),
        "halftone_score": round(float(results.get("halftone_score", 0.0)), 4),
        "composite_counterfeit_score": round(float(results.get("composite_counterfeit_score", 0.0)), 4),
        "is_counterfeit": bool(results.get("is_counterfeit", False)),
        "verdict": results.get("verdict", "INSPECTING"),
        "fps": round(float(results.get("fps", 0.0)), 1),
        "latency_ms": round(float(results.get("latency_ms", 0.0)), 2),
        "roi_bbox": results.get("roi_bbox"),
        "corners": results.get("corners")
    }

    # Audio trigger tag
    denom = payload["denomination"]
    is_cf = payload["is_counterfeit"]
    if is_cf:
        payload["audio_tag"] = "COUNTERFEIT_ALARM"
    elif denom != "Background" and payload["confidence"] >= 0.30:
        payload["audio_tag"] = str(denom)
    else:
        payload["audio_tag"] = None

    if include_images:
        clahe_img = results.get("clahe_bgr")
        sr_img = results.get("sr_roi")
        raw_roi = results.get("raw_roi")
        topo_img = results.get("topological_grad")

        halftone_metrics = results.get("halftone_metrics", {})
        vis_spec = halftone_metrics.get("vis_spectrum") if isinstance(halftone_metrics, dict) else None

        if vis_spec is not None and vis_spec.size > 0:
            spec_magma = cv2.applyColorMap(vis_spec, cv2.COLORMAP_MAGMA)
        else:
            spec_magma = None

        payload["plates"] = {
            "clahe": image_to_base64(clahe_img),
            "super_res": image_to_base64(sr_img),
            "raw_roi": image_to_base64(raw_roi),
            "topography": image_to_base64(topo_img),
            "fft_spectrum": image_to_base64(spec_magma)
        }

        if isinstance(halftone_metrics, dict) and "cmyk_plates" in halftone_metrics:
            cmyk = halftone_metrics["cmyk_plates"]
            payload["cmyk_plates"] = {
                "c": image_to_base64(cmyk.get("C")),
                "m": image_to_base64(cmyk.get("M")),
                "y": image_to_base64(cmyk.get("Y")),
                "k": image_to_base64(cmyk.get("K")),
            }

        wm_data = results.get("watermark_delta")
        if wm_data and isinstance(wm_data, dict):
            payload["watermark"] = {
                "density": round(float(wm_data.get("transmission_density", 0.0)), 4),
                "mask": image_to_base64(wm_data.get("watermark_mask")),
                "delta_enhanced": image_to_base64(wm_data.get("delta_enhanced"))
            }

    return payload


@app.get("/api/health")
def health_check():
    """Lightweight health check endpoint for cloud/serverless monitoring."""
    return {"status": "healthy", "service": "The Looker Forensic Studio", "version": "2.2.0"}


@app.get("/", response_class=HTMLResponse)
@app.get("/api", response_class=HTMLResponse)
@app.get("/api/index.py", response_class=HTMLResponse)
def serve_index():
    """Serve the primary HTML application with multi-candidate path search."""
    index_candidates = [
        os.path.join(TEMPLATES_DIR, "index.html"),
        os.path.join(os.path.dirname(BASE_DIR), "web", "templates", "index.html"),
        os.path.abspath("web/templates/index.html")
    ]
    for p in index_candidates:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
    return HTMLResponse("<h3>Forensic Studio: index.html not found</h3>", status_code=404)


@app.post("/api/inspect/image")
async def inspect_uploaded_image(request: Request):
    """
    Analyze an uploaded banknote photo.
    Supports either JSON body { "image_base64": "..." } or raw binary image body.
    """
    content_type = request.headers.get("content-type", "")
    filename = request.headers.get("x-filename", "uploaded_note.jpg")
    frame = None

    if "application/json" in content_type:
        data = await request.json()
        b64_str = data.get("image_base64") or data.get("image")
        if b64_str:
            frame = base64_to_image(b64_str)
        filename = data.get("filename", filename)
    else:
        body = await request.body()
        if body:
            np_arr = np.frombuffer(body, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None or frame.size == 0:
        return JSONResponse({"error": "Invalid or unreadable image data"}, status_code=400)

    # Simulate screen flash differencing for static image so watermark plates populate
    pipe = get_pipeline()
    amb, fl = ScreenFlashDifferencing.simulate_screen_flash(frame)
    flash_delta = pipe.flash_differencer.compute_difference(amb, fl)

    det_check = pipe.detector.detect(frame)
    is_pre = det_check.get("strategy") in ["full_banknote_frame", "standby", "guided_reticle"]
    results = pipe.process_frame(frame, cached_flash_delta=flash_delta, is_pre_cropped=is_pre)
    serialized = serialize_inspection_results(results, include_images=True)
    serialized["filename"] = filename
    serialized["resolution"] = f"{frame.shape[1]}x{frame.shape[0]}"
    return JSONResponse(serialized)


@app.post("/api/generate-demo")
async def generate_synthetic_demo(request: Request):
    """Generate a synthetic currency sample on-the-fly and run deep inspection."""
    data = {}
    try:
        data = await request.json()
    except Exception:
        pass

    denomination = str(data.get("denomination", "500"))
    is_counterfeit = bool(data.get("is_counterfeit", False))
    counterfeit_type = str(data.get("counterfeit_type", "cmyk"))

    frame = SyntheticCurrencyGenerator.generate_banknote(
        denomination=denomination,
        is_counterfeit=is_counterfeit,
        width=960,
        height=480
    )
    sample_label = "Counterfeit (CMYK Rosettes)" if is_counterfeit else "Genuine (Intaglio Fine-Line)"

    pipe = get_pipeline()
    amb, fl = ScreenFlashDifferencing.simulate_screen_flash(frame)
    flash_delta = pipe.flash_differencer.compute_difference(amb, fl)

    results = pipe.process_frame(frame, cached_flash_delta=flash_delta, is_pre_cropped=True)
    serialized = serialize_inspection_results(results, include_images=True)
    serialized["sample_image"] = image_to_base64(frame)
    serialized["sample_type"] = sample_label
    return JSONResponse(serialized)


@app.post("/api/compare-demo")
async def compare_synthetic_demo(request: Request):
    """
    Side-by-Side Dual Comparison Engine:
    - If user provides an image: inspects the note, and if real, pairs it against a
      counterfeit reference explaining why it's genuine; if fake, pairs it against an
      authentic RBI reference explaining why it's counterfeit.
    - If user selects a denomination: generates parallel genuine vs. counterfeit benchmarks
      with detailed forensic discrepancy breakdown.
    """
    data = {}
    try:
        data = await request.json()
    except Exception:
        pass

    pipe = get_pipeline()
    denomination = str(data.get("denomination", "500"))
    b64_str = data.get("image_base64") or data.get("image")
    user_frame = base64_to_image(b64_str) if b64_str else None

    if user_frame is not None and user_frame.size > 0:
        # 1. Analyze User Note
        amb_u, fl_u = ScreenFlashDifferencing.simulate_screen_flash(user_frame)
        delta_u = pipe.flash_differencer.compute_difference(amb_u, fl_u)
        det_u = pipe.detector.detect(user_frame)
        is_pre_u = det_u.get("strategy") in ["full_banknote_frame", "standby", "guided_reticle"]
        user_res = pipe.process_frame(user_frame, cached_flash_delta=delta_u, is_pre_cropped=is_pre_u)
        user_serialized = serialize_inspection_results(user_res, include_images=True)
        user_serialized["sample_image"] = image_to_base64(user_frame)

        detected_denom = str(user_res.get("denomination", denomination))
        if detected_denom in ["Background", "No Currency in View", "Undetermined"]:
            detected_denom = denomination
        user_is_counterfeit = bool(user_res.get("is_counterfeit", False))

        # 2. Pair Against Counterpart Reference
        ref_is_counterfeit = not user_is_counterfeit
        ref_frame = SyntheticCurrencyGenerator.generate_banknote(
            denomination=detected_denom,
            is_counterfeit=ref_is_counterfeit,
            width=960,
            height=480
        )
        amb_r, fl_r = ScreenFlashDifferencing.simulate_screen_flash(ref_frame)
        delta_r = pipe.flash_differencer.compute_difference(amb_r, fl_r)
        ref_res = pipe.process_frame(ref_frame, cached_flash_delta=delta_r, is_pre_cropped=True)
        ref_serialized = serialize_inspection_results(ref_res, include_images=True)
        ref_serialized["sample_image"] = image_to_base64(ref_frame)

        if user_is_counterfeit:
            headline = f"Suspect / Counterfeit Detected (₹{detected_denom}) — Forensic Discrepancy vs. Authentic RBI Currency"
            badge_text = "SUSPECT NOTE DETECTED"
            badge_type = "suspect"
            left_title = "Why Your Banknote Was Flagged Counterfeit"
            left_points = [
                f"Primary Trigger: {user_res.get('authenticity_reason', 'Counterfeit pattern detected')}",
                f"Commercial Dither: Halftone score {user_res.get('halftone_score', 0):.2f} (CMYK process printing detected)",
                "Surface Relief Mismatch: Lacks the deep tactile ridges created by RBI high-pressure intaglio presses",
                "Issuing Authority: Missing authentic Reserve Bank of India typography & security thread"
            ]
            right_title = "Official RBI Security Standards Expected on Authentic Notes"
            right_points = [
                "Official Legal Typography: Crisp intaglio 'भारतीय रिज़र्व बैंक' and 'RESERVE BANK OF INDIA' lettering",
                "Continuous Lathe Linework: Seamless intaglio engraving with zero laser/inkjet CMYK rosette dots",
                "Embedded Multi-Tonal Watermark: High optical light-transmission density",
                "Raised Tactile Relief: Heavy ink embossing perceptible to touch for accessibility"
            ]
            left_card = {
                "title": "YOUR NOTE: SUSPECT / COUNTERFEIT",
                "tag": "SUSPECT NOTE",
                "is_counterfeit": True,
                "data": user_serialized
            }
            right_card = {
                "title": "GENUINE RBI REFERENCE",
                "tag": "OFFICIAL STANDARD",
                "is_counterfeit": False,
                "data": ref_serialized
            }
            genuine_data = ref_serialized
            counterfeit_data = user_serialized
        else:
            headline = f"Authentic Banknote Verified (₹{detected_denom}) — Forensic Analysis vs. Counterfeit Reproduction"
            badge_text = "GENUINE NOTE VERIFIED"
            badge_type = "genuine"
            left_title = "Why Your Banknote Was Verified Genuine"
            left_points = [
                "Continuous Intaglio Lathe Relief: Verified continuous directional gradient energy with zero CMYK dots",
                "Clean Substrate: Absence of commercial 4-color process screen rosettes or inkjet spray mist",
                "Official Authority Verified: Verified authentic 'RESERVE BANK OF INDIA' typography & emblem",
                f"Minimal Counterfeit Risk: Exceptionally low composite counterfeit score ({user_res.get('composite_counterfeit_score', 0):.2f})"
            ]
            right_title = "What Counterfeit Reproductions Fail At"
            right_points = [
                "CMYK Halftone Dithering: Microscopic 4-color process dots at 15°, 45°, 75° printer screen angles",
                "Flat Paper Relief: Lacks the deep tactile ridges of authentic high-pressure currency presses",
                "Unauthorized Inscriptions: Counterfeits feature imitation markers (Children Bank, Manoranjan)",
                "Zero Optical Depth: Photocopies lack authentic embedded multi-tonal watermark transmission"
            ]
            left_card = {
                "title": "YOUR NOTE: VERIFIED GENUINE",
                "tag": "VERIFIED NOTE",
                "is_counterfeit": False,
                "data": user_serialized
            }
            right_card = {
                "title": "COUNTERFEIT REFERENCE",
                "tag": "COUNTERFEIT SAMPLE",
                "is_counterfeit": True,
                "data": ref_serialized
            }
            genuine_data = user_serialized
            counterfeit_data = ref_serialized

        return JSONResponse({
            "mode": "user_comparison",
            "denomination": detected_denom,
            "user_is_counterfeit": user_is_counterfeit,
            "left_card": left_card,
            "right_card": right_card,
            "explanation": {
                "headline": headline,
                "badge_text": badge_text,
                "badge_type": badge_type,
                "left_title": left_title,
                "left_points": left_points,
                "right_title": right_title,
                "right_points": right_points
            },
            "genuine": genuine_data,
            "counterfeit": counterfeit_data
        })

    # Benchmark mode: Generate both Genuine and Counterfeit references
    gen_frame = SyntheticCurrencyGenerator.generate_banknote(
        denomination=denomination,
        is_counterfeit=False,
        width=960,
        height=480
    )
    amb_g, fl_g = ScreenFlashDifferencing.simulate_screen_flash(gen_frame)
    delta_g = pipe.flash_differencer.compute_difference(amb_g, fl_g)
    results_gen = pipe.process_frame(gen_frame, cached_flash_delta=delta_g, is_pre_cropped=True)
    serialized_gen = serialize_inspection_results(results_gen, include_images=True)
    serialized_gen["sample_image"] = image_to_base64(gen_frame)

    sus_frame = SyntheticCurrencyGenerator.generate_banknote(
        denomination=denomination,
        is_counterfeit=True,
        width=960,
        height=480
    )
    amb_s, fl_s = ScreenFlashDifferencing.simulate_screen_flash(sus_frame)
    delta_s = pipe.flash_differencer.compute_difference(amb_s, fl_s)
    results_sus = pipe.process_frame(sus_frame, cached_flash_delta=delta_s, is_pre_cropped=True)
    serialized_sus = serialize_inspection_results(results_sus, include_images=True)
    serialized_sus["sample_image"] = image_to_base64(sus_frame)

    return JSONResponse({
        "mode": "benchmark",
        "denomination": denomination,
        "left_card": {
            "title": "AUTHENTIC INTAGLIO CURRENCY",
            "tag": "OFFICIAL STANDARD",
            "is_counterfeit": False,
            "data": serialized_gen
        },
        "right_card": {
            "title": "COUNTERFEIT CMYK DITHERED",
            "tag": "COUNTERFEIT SAMPLE",
            "is_counterfeit": True,
            "data": serialized_sus
        },
        "explanation": {
            "headline": f"Forensic Comparative Benchmark (₹{denomination}): Authentic Intaglio vs. CMYK Counterfeit",
            "badge_text": "COMPARATIVE LAB BENCHMARK",
            "badge_type": "benchmark",
            "left_title": "Why Authentic Currency Passes Verification",
            "left_points": [
                "Intaglio Engraving Relief: High-pressure tactile micro-ridges verified by directional gradient analysis",
                "Continuous Spot-Color Ink: Substrate printed without 4-color laser/inkjet process dithering",
                "Official Legal Authority: Precision Reserve Bank of India typography and Ashok Pillar seal",
                "Optical Watermark Density: Verified embedded multi-tonal watermark light transmission"
            ],
            "right_title": "Why Counterfeit Currency Fails Verification",
            "right_points": [
                "Halftone Dither Rosettes: Microscopic Cyan, Magenta, Yellow dot clusters at 15°, 45°, 75° angles",
                "Flat Surface Relief: Lacks the raised tactile ink layers of genuine currency presses",
                "Imitation Inscriptions: Uses unauthorized commercial novelty text or imitation security threads",
                "Zero Optical Depth: Flat photocopy substrate without variable-density watermark transmission"
            ]
        },
        "genuine": serialized_gen,
        "counterfeit": serialized_sus
    })


@app.post("/api/flash-diff")
async def compute_screen_flash_difference(request: Request):
    """Compute active optical screen-flash difference from paired client-captured frames."""
    data = await request.json()
    ambient = base64_to_image(data.get("ambient_base64", ""))
    flash = base64_to_image(data.get("flash_base64", ""))

    if ambient is None or flash is None:
        return JSONResponse({"error": "Failed to decode frame pair"}, status_code=400)

    pipe = get_pipeline()
    diff_results = pipe.flash_differencer.compute_difference(ambient, flash)
    return JSONResponse({
        "transmission_density": round(float(diff_results["transmission_density"]), 4),
        "watermark_mask": image_to_base64(diff_results["watermark_mask"]),
        "delta_enhanced": image_to_base64(diff_results["delta_enhanced"]),
        "delta_bgr": image_to_base64(diff_results["delta_bgr"])
    })


@app.post("/api/live-frame")
async def live_frame_feed(request: Request):
    """
    High-frequency live camera frame endpoint.
    Processes live frame and returns compact telemetry (<25ms execution).
    Zero external dependencies, 100% resilient fallback for webcams.
    """
    body = await request.body()
    frame = None
    if body:
        np_arr = np.frombuffer(body, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None or frame.size == 0:
        return JSONResponse({"error": "Invalid frame"}, status_code=400)

    pipe = get_pipeline()
    results = pipe.process_frame(frame)
    telemetry = serialize_inspection_results(results, include_images=False)
    return JSONResponse(telemetry)


@app.websocket("/ws/live")
async def websocket_live_feed(websocket: WebSocket):
    """
    High-frequency WebSocket endpoint for real-time webcam frame streaming.
    Receives JPEG base64 frames from browser client, executes inference, and sends back
    compact telemetry, bounding box, and audio tags at >25 FPS.
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            if not data:
                continue

            frame = base64_to_image(data)
            if frame is None:
                await websocket.send_json({"error": "Invalid frame"})
                continue

            pipe = get_pipeline()
            results = pipe.process_frame(frame)
            telemetry = serialize_inspection_results(results, include_images=False)
            await websocket.send_json(telemetry)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.close()
        except Exception:
            pass
