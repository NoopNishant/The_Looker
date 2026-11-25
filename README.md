# The Looker: Currency Forensics & Counterfeit Guard

A high-performance edge computer vision and forensic analysis platform engineered for standard CPU execution and live laptop webcams. Features an editorial, minimalist web studio in warm linen and pastel spectrums, integrating deep learning semantic classification, active optical screen-flash differencing, neural super-resolution micro-texture enhancement, 2D FFT halftone rosette detection, robust face/skin-suppressed quadrilateral localization, and non-visual discreet acoustic feedback.

---

## Web Studio Highlights (`http://localhost:8000`)

- **Aesthetic Minimalist Design**: Editorial warm linen (`#F7F5F0`), oat cream, and ecru backdrop with frosted glassmorphism surfaces (`backdrop-filter: blur(18px)`), refined typography (`Newsreader` serif + `Plus Jakarta Sans` + `JetBrains Mono`), and a soft pastel spectrum:
  - **Genuine Verdict**: Soft sage / pistachio (`#4E7259`, background `#EBF2ED`)
  - **Counterfeit / Suspect**: Muted terracotta / dusty coral (`#B84A4A`, background `#FDF0EE`)
  - **Telemetry & Focus**: Warm amber sand (`#A58266`, background `#F7F1EB`)
- **3 Dedicated Forensic Operational Modes**:
  1. **Live Camera**:
     - High-speed WebSocket (`/ws/live`) and binary HTTP (`/api/live-frame`) streaming delivering real-time bounding quads, central reticles, and risk metrics at **>25–35 FPS**.
     - Face & skin suppression to actively eliminate user hands and faces from triggering false banknote detections.
     - Temporal stabilization that votes denomination over recent frames and adds hysteresis to counterfeit verdicts, eliminating reticle flicker.
     - Dual camera support with front/environment camera switching.
  2. **Photo Forensics**:
     - Instant drag-and-drop or file upload for high-resolution images (PNG, JPG, WEBP).
     - Generates all 6 multi-spectral micro-texture inspection plates (CLAHE micro-relief, 2x ESPCN neural super-res, 2D FFT rosette spectrum, Cyan & Magenta ink separations, and watermark mask).
  3. **Side-by-Side Dual Comparison Lab**:
     - **Bidirectional Comparative Analysis**:
       - **When a Genuine Note is Detected**: Places the user's note on the left (`YOUR NOTE: VERIFIED GENUINE`) and contrasts it against a synthetic counterfeit reference on the right, providing an educational discrepancy breakdown (*Continuous intaglio lathe relief, absence of 4-color process dithering, authentic RBI typography* vs. *flat relief and halftone rosettes*).
       - **When a Counterfeit / Suspect Note is Detected**: Places the user's note on the left (`YOUR NOTE: SUSPECT / COUNTERFEIT`) and pairs it against an authentic RBI reference on the right, explaining the detected fraud markers (*e.g., Children Bank / Manoranjan Bank, commercial dithering*) vs. authentic Reserve Bank of India security standards.
     - **Direct Note Upload Options**: Upload a banknote photo to compare directly via the top viewport header button, section header button, control bar, or by clicking anywhere directly on the split-screen canvas.
     - **Denomination Benchmarks**: Instant 1-click comparison across all Indian Rupee denominations (₹10 – ₹2000).
- **In-Browser Screen-Flash Watermark Extraction**:
  - Click **"Pulse Watermark Flash"** to trigger a full-screen white illumination pulse; snaps paired ambient and illuminated frames to compute the optical transmission delta $\Delta I = |I_{flash} - I_{ambient}|$.
- **Discreet Web Audio API Synthesizer**:
  - Pure in-browser acoustic rhythm synthesis mapped to each denomination without disruptive Text-to-Speech.

---

## Core Forensic & Computer Vision Architecture

### 1. Robust Banknote Localization & Face-Suppressed Tracking (`core/detector.py` / `detector.py`)
- **Active Face & Skin Suppression**: Built-in Haar Cascade face detection coupled with YCrCb skin chrominance masking (`Cb ∈ [77, 127]`, `Cr ∈ [133, 173]`) aggressively rejects human faces, fingers, and hands from false banknote detections.
- **Dual Candidate Generators**:
  - Auto-thresholded adaptive Canny edge detection for high-contrast scenes.
  - Lab-space background-difference color distance masking ($\Delta E_{Lab}$ against outer border pixels) to effortlessly isolate low-contrast notes (e.g., stone-grey ₹500 resting on a grey desk).
- **Perspective-Aware Quadrilateral Geometry**: Evaluates candidate aspect ratios on the 3D quadrilateral polygon (tolerating tilt between 1.55 and 2.95).
- **Perspective-Rectified Upright Cropping**: Quad fitting (`approxPolyDP`) produces an upright, perspective-corrected, background-free note crop matching the denomination's dimensional ratio.
- **Temporal Reticle Smoothing**: Corner-level exponential smoothing with large-jump reset and track-hold ensures smooth, zero-flicker live reticle tracking.

### 2. Calibrated Multi-Cue Denomination Deduction (`core/denomination_engine.py` / `denomination_engine.py`)
- **Illumination-Invariant Chromaticity**: Evaluates normalized RGB coordinates ($r = R/\Sigma, g = G/\Sigma, b = B/\Sigma$) strictly on well-exposed pixels (excluding specular glare and near-black ink).
- **Saturation-Weighted Circular Mean Hue**: Computes circular mean angles for Hue ($\theta = \text{atan2}(\Sigma w \sin \alpha, \Sigma w \cos \alpha)$), completely eliminating errors caused by OpenCV's 0°/180° red hue boundary wrap-around.
- **Multi-Reference Profiles**: Equips denominations with alternate reference anchors (`chroma_alt`, `hue_alt`) covering daylight vs. incandescent lighting shifts.
- **Renormalized Probability Blending**: Renormalizes CNN logits strictly across valid denominations, preventing background mass leakage.
- **Calibrated Margin Confidence**: Maps top-1 vs. top-2 winning margins directly to calibrated confidence percentages (0.50 – 0.99).

### 3. Temporal Feed Stabilization (`core/pipeline.py` / `pipeline.py`)
- **Sliding-Window Voting**: When operating in live stream mode (`live=True`), the `TemporalStabilizer` votes denomination candidates across recent frames.
- **Verdict Hysteresis**: Requires multiple consistent suspect frames before triggering alarms and debounces acoustic feedback so single-frame misreads never emit false signals.

### 4. Realistic Synthetic Scene & Banknote Generation (`synthetic_data.py`)
- **Authentic Aspect Ratios**: Implements official Mahatma Gandhi (New) Series dimensional aspect ratios:
  - ₹10: **1.95** | ₹20: **2.03** | ₹50: **2.03** | ₹100: **2.15** | ₹200: **2.21** | ₹500: **2.27** | ₹2000: **2.52**
- **Camera Scene Simulation (`generate_scene()`)**: Renders banknotes inside realistic cluttered environments featuring:
  - 3D in-plane rotation and keystone perspective warping.
  - Incandescent / cool-white color-temperature casts and exposure variations.
  - Optical lens vignetting, soft contact drop shadows, and Gaussian sensor noise.
  - Desk background textures with random clutter occlusions.
  - Outputs ground-truth $(4, 2)$ corner coordinates for detector validation.
- **Synthetic Banknote Generation (`generate_banknote()`)**: Generates both authentic intaglio currency (guilloche waves, microprint, embedded watermark) and counterfeit currency (CMYK halftone rosette screens, flat relief, and toy spoof markers).

### 5. Multi-Task Neural Network Backbone (MobileNetV3-Small)
- **OpenCV DNN & PyTorch Dual Runtime**: Supports pure CPU execution via OpenCV DNN using ONNX weights (`<65MB` bundle with zero PyTorch overhead) alongside full PyTorch fallback.
- **Head 1 (Denomination)**: Multi-class classification over Indian Rupee denominations.
- **Head 2 (Counterfeit Risk)**: Binary counterfeit risk probability prediction.

### 6. Super-Resolution CMYK Halftone Detector (`core/super_res_halftone.py`)
- **2x ESPCN Super-Resolution**: Direct CPU neural magnification of micro-texture patches.
- **CMYK Ink Decomposition**: Separates RGB into Cyan, Magenta, Yellow, and Key (Black) plates with dark-pixel stabilization.
- **2D FFT Rosette Spectrum Analysis**: Detects periodic harmonic lattice peaks characteristic of 4-color process commercial printer screens (15°, 45°, 75°) vs. continuous-line intaglio engraving.

### 7. Banknote Security Feature & Counterfeit Validator (`core/security_validator.py`)
- **Multi-Scale Template Matching**: Targets known commercial novelty/toy banknote spoof markers:
  - Header spoofing: `"CHILDREN BANK OF INDIA"` and `"भारतीय मनोरंजन बैंक"`.
  - Watermark spoofing: `"FULL OF FUN"`.
  - Security thread imitation: `"CHURAN LABEL"` printed bands.
- **Resolution-Gated Search**: Strictly prevents template downscaling below legible text resolution (`sh >= 12`), eliminating false positives on authentic Reserve Bank of India typography.

### 8. Discreet Acoustic Audio Feedback (`core/audio_feedback.py`)
- Replaces noisy Text-to-Speech with subtle tactile acoustic rhythms:
  - **₹10**: Single high pulse (880 Hz, 80ms)
  - **₹20**: Double micro-pulse rhythm
  - **₹50**: Melodic rising third
  - **₹100**: Rising fifth chime
  - **₹200**: Triple staccato pulse
  - **₹500**: Resonant major triad chord
  - **₹2000**: Quadruple ascending arpeggio
  - **COUNTERFEIT / SUSPECT**: Urgent discordant dual-sawtooth alarm (180 Hz + 194 Hz)

---

## Project Structure

```
The Looker/
├── core/
│   ├── __init__.py
│   ├── model.py              # MultiTaskMobileNetV3 neural architecture & joint loss
│   ├── detector.py           # Banknote localization, Haar face suppression & quad fitting
│   ├── denomination_engine.py# Circular-hue chromaticity & calibrated denomination deduction
│   ├── security_validator.py # Multi-scale toy counterfeit marker verification
│   ├── preprocessing.py      # LAB CLAHE & Screen-Flash Differencing
│   ├── super_res_halftone.py # ESPCN Super-Resolution & CMYK FFT Dot Detector
│   ├── audio_feedback.py     # NumPy acoustic waveform synthesizer & async engine
│   └── pipeline.py           # Unified real-time inspection engine with TemporalStabilizer
├── web/
│   ├── app.py                # FastAPI ASGI backend (<25ms frame streaming & comparison API)
│   ├── templates/index.html  # Minimalist editorial UI with 4 forensic modes
│   └── static/               # Styling (CSS), client streaming, and Web Audio API (JS)
├── data/
│   ├── counterfeit_templates/# Known toy fake note templates
│   └── real_crops_50/        # Reference authentic banknote crops
├── models/
│   ├── currency_mobilenet_v3.onnx # Pure OpenCV DNN CPU model weights (<65MB)
│   └── currency_mobilenet_v3.pth  # PyTorch model weights
├── utils/
│   ├── __init__.py
│   └── visualizer.py         # Real-time HUD, diagnostic panels, and gauges
├── tests/
│   ├── __init__.py
│   ├── test_model.py         # Tensor shape and gradient backprop tests
│   ├── test_preprocessing.py # CLAHE & screen-flash tests
│   ├── test_super_res.py     # ESPCN upscaler & halftone FFT tests
│   ├── test_audio.py         # Acoustic pulse & WAV byte header tests
│   ├── test_live_deduction.py# Live banknote deduction & authenticity tests
│   └── benchmark_fps.py      # End-to-end 1080p CPU FPS benchmark (>15 FPS)
├── denomination_engine.py    # Standalone modular denomination deduction engine
├── detector.py               # Standalone modular face-suppressed detector
├── pipeline.py               # Standalone modular inspection pipeline
├── synthetic_data.py         # Realistic banknote & cluttered camera scene synthesizer
├── train_currency_model.py   # Multi-scenario curriculum training script
├── run_web.py                # All-in-one autonomous web studio launcher
├── main.py                   # Real-time CLI & GUI application runner
├── render.yaml               # Render Blueprint deployment configuration
├── Procfile                  # Render / WSGI-ASGI process entrypoint
└── README.md
```

---

## Deployment on Render

This repository includes `render.yaml` and `Procfile` for 1-click deployment on [Render](https://render.com):
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn web.app:app --host 0.0.0.0 --port $PORT`
- **Python Version**: `3.11.9` (pinned in `render.yaml` for prebuilt `numpy` and `opencv-python-headless` wheels)

---

## Execution Commands

### 1. Launch Forensic Web Studio (Recommended)
```bash
python run_web.py
```
This single autonomous command automatically:
- Checks and reclaims port 8000 from any orphaned processes.
- Pre-warms the neural networks (MobileNetV3 ONNX, 2x ESPCN Super-Resolution, CLAHE, CMYK FFT).
- Launches the high-speed ASGI server (<25ms frame latency).
- Automatically opens your default web browser directly to **`http://127.0.0.1:8000`**.

### 2. Run Automated Test Suite
```bash
python -m unittest discover tests
```

### 3. Run 1080p CPU FPS Throughput Benchmark
```bash
python tests/benchmark_fps.py
```

### 4. Run Live Interactive Simulation Mode
Cycles between genuine and counterfeit banknotes with interactive HUD and acoustic feedback:
```bash
python main.py --demo
```

### 5. Run with Physical Webcam
```bash
python main.py --camera 0
```

### 6. Inspect a Static Note Image
```bash
python main.py --image output/genuine_note_500.png
python main.py --image output/counterfeit_note_500.png
```

---

## Interactive Controls (During Live GUI)

| Key | Action |
|-----|--------|
| `[SPACE]` | Trigger active Screen-Flash Differencing to extract watermark |
| `[M]` | Mute / unmute discreet audio rhythms |
| `[S]` | Save diagnostic inspection snapshot to disk |
| `[Q]` / `[ESC]` | Quit application |
