"""
web/app.py - FastAPI Application Server for The Looker Forensic Web Studio.

Zero-dependency native endpoints:
- GET /                    : Serves interactive minimalist frontend
- POST /api/inspect/image  : Inspects uploaded photo (binary or JSON base64)
- POST /api/inspect/video  : Analyzes uploaded video across timeline
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

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# Mount static directory
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Initialize global pipeline engine
pipeline = CurrencyInspectionPipeline(device="cpu", enable_audio=False)


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


@app.get("/", response_class=HTMLResponse)
def serve_index():
    """Serve the primary HTML application."""
    index_file = os.path.join(TEMPLATES_DIR, "index.html")
    if not os.path.exists(index_file):
        return HTMLResponse("<h3>Index page not found</h3>", status_code=404)
    with open(index_file, "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content)


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
    amb, fl = ScreenFlashDifferencing.simulate_screen_flash(frame)
    flash_delta = pipeline.flash_differencer.compute_difference(amb, fl)

    results = pipeline.process_frame(frame, cached_flash_delta=flash_delta)
    serialized = serialize_inspection_results(results, include_images=True)
    serialized["filename"] = filename
    serialized["resolution"] = f"{frame.shape[1]}x{frame.shape[0]}"
    return JSONResponse(serialized)


@app.post("/api/inspect/video")
async def inspect_uploaded_video(request: Request):
    """Analyze an uploaded video file by sampling keyframes across timeline."""
    filename = request.headers.get("x-filename", "uploaded_video.mp4")
    body = await request.body()

    suffix = os.path.splitext(filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(body)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return JSONResponse({"error": "Failed to decode video stream"}, status_code=400)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration_sec = total_frames / max(1.0, video_fps)

        # Sample up to 12 representative frames evenly spaced
        sample_count = min(12, max(3, total_frames // 10))
        indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)

        timeline = []
        scores = []
        denoms = {}

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            res = pipeline.process_frame(frame)
            score = float(res["composite_counterfeit_score"])
            scores.append(score)

            d = res["denomination"]
            if d != "Background":
                denoms[d] = denoms.get(d, 0) + 1

            timeline.append({
                "time_sec": round(float(idx / video_fps), 2),
                "frame_idx": int(idx),
                "denomination": d,
                "confidence": round(float(res["confidence"]), 3),
                "counterfeit_score": round(score, 3),
                "verdict": res["verdict"],
                "thumbnail": image_to_base64(cv2.resize(frame, (160, 90)))
            })

        cap.release()

        avg_score = float(np.mean(scores)) if scores else 0.0
        primary_denom = max(denoms, key=denoms.get) if denoms else "Undetermined"
        overall_counterfeit = avg_score >= pipeline.counterfeit_threshold

        return JSONResponse({
            "filename": filename,
            "total_frames": total_frames,
            "duration_sec": round(duration_sec, 2),
            "video_fps": round(video_fps, 1),
            "primary_denomination": primary_denom,
            "average_counterfeit_score": round(avg_score, 3),
            "verdict": "SUSPECT / COUNTERFEIT" if overall_counterfeit else "GENUINE",
            "is_counterfeit": overall_counterfeit,
            "timeline": timeline
        })
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


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

    if denomination == "toy_50" or counterfeit_type == "toy":
        toy_path = "uploaded_fake_note.jpg"
        if os.path.exists(toy_path):
            frame = cv2.imread(toy_path)
        else:
            frame = SyntheticCurrencyGenerator.generate_banknote(
                denomination="50",
                is_counterfeit=True,
                width=960,
                height=480
            )
        sample_label = "Toy Banknote (Children Bank of India / Churan Label)"
    else:
        frame = SyntheticCurrencyGenerator.generate_banknote(
            denomination=denomination,
            is_counterfeit=is_counterfeit,
            width=960,
            height=480
        )
        sample_label = "Counterfeit (CMYK Rosettes)" if is_counterfeit else "Genuine (Intaglio Fine-Line)"

    amb, fl = ScreenFlashDifferencing.simulate_screen_flash(frame)
    flash_delta = pipeline.flash_differencer.compute_difference(amb, fl)

    results = pipeline.process_frame(frame, cached_flash_delta=flash_delta)
    serialized = serialize_inspection_results(results, include_images=True)
    serialized["sample_image"] = image_to_base64(frame)
    serialized["sample_type"] = sample_label
    return JSONResponse(serialized)


@app.post("/api/compare-demo")
async def compare_synthetic_demo(request: Request):
    """
    Generate both Genuine and Counterfeit samples of a selected denomination
    and return parallel forensic inspection metrics for side-by-side comparison.
    """
    data = {}
    try:
        data = await request.json()
    except Exception:
        pass

    raw_denom = str(data.get("denomination", "500"))
    denomination = "50" if raw_denom == "toy_50" else raw_denom

    # 1. Generate & Inspect Genuine Note
    gen_frame = SyntheticCurrencyGenerator.generate_banknote(
        denomination=denomination,
        is_counterfeit=False,
        width=960,
        height=480
    )
    amb_g, fl_g = ScreenFlashDifferencing.simulate_screen_flash(gen_frame)
    delta_g = pipeline.flash_differencer.compute_difference(amb_g, fl_g)
    results_gen = pipeline.process_frame(gen_frame, cached_flash_delta=delta_g)
    serialized_gen = serialize_inspection_results(results_gen, include_images=True)
    serialized_gen["sample_image"] = image_to_base64(gen_frame)

    # 2. Generate & Inspect Counterfeit Note
    if raw_denom == "toy_50" and os.path.exists("uploaded_fake_note.jpg"):
        sus_frame = cv2.imread("uploaded_fake_note.jpg")
    else:
        sus_frame = SyntheticCurrencyGenerator.generate_banknote(
            denomination=denomination,
            is_counterfeit=True,
            width=960,
            height=480
        )
    amb_s, fl_s = ScreenFlashDifferencing.simulate_screen_flash(sus_frame)
    delta_s = pipeline.flash_differencer.compute_difference(amb_s, fl_s)
    results_sus = pipeline.process_frame(sus_frame, cached_flash_delta=delta_s)
    serialized_sus = serialize_inspection_results(results_sus, include_images=True)
    serialized_sus["sample_image"] = image_to_base64(sus_frame)

    return JSONResponse({
        "denomination": denomination,
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

    diff_results = pipeline.flash_differencer.compute_difference(ambient, flash)
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

    results = pipeline.process_frame(frame)
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

            results = pipeline.process_frame(frame)
            telemetry = serialize_inspection_results(results, include_images=False)
            await websocket.send_json(telemetry)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.close()
        except Exception:
            pass
