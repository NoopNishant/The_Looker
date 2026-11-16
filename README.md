# The Looker: Currency Forensics & Counterfeit Guard

A high-performance edge computer vision and forensic analysis platform engineered for CPU execution and standard laptop webcams. Features an editorial, minimalist web studio in warm beige and pastel spectrums, integrating deep learning semantic classification, active optical screen-flash differencing, neural super-resolution micro-texture enhancement, 2D FFT halftone rosette detection, and non-visual discreet acoustic feedback.

---

## Web Studio Highlights (`http://localhost:8000`)

- **Aesthetic Minimalist Design**: Editorial warm linen (`#F7F5F0`), oat cream, and ecru backdrop with frosted glassmorphism surfaces (`backdrop-filter: blur(18px)`), refined typography (`Newsreader` serif + `Plus Jakarta Sans` + `JetBrains Mono`), and a soft pastel spectrum:
  - **Genuine Verdict**: Soft sage / pistachio (`#4E7259`, background `#EBF2ED`)
  - **Counterfeit / Suspect**: Muted terracotta / dusty coral (`#B84A4A`, background `#FDF0EE`)
  - **Telemetry & Focus**: Warm amber sand (`#A58266`, background `#F7F1EB`)
- **Live Feed Detection**:
  - High-speed WebSocket streaming (`/ws/live`) delivering live inference bounding boxes, central targeting reticles, and risk metrics at **>25–30 FPS**.
  - Camera switching support (user-facing and environment cameras).
- **In-Browser Screen-Flash Watermark Extraction**:
  - Click **"Pulse Watermark Flash"** to trigger a full-screen white illumination pulse; snaps paired ambient and illuminated frames to compute the optical transmission delta $\Delta I = |I_{flash} - I_{ambient}|$.
- **Uploaded Media Detection**:
  - **Photos (PNG, JPG, WEBP)**: Instant drag-and-drop forensic breakdown with all 6 multi-spectral micro-texture inspection plates.
  - **Videos (MP4, MOV, WEBM)**: Timeline keyframe decomposition showing per-frame verdicts, thumbnail strips, and average counterfeit score.
- **Synthetic Demonstration Bench**:
  - 1-click test buttons to instantly inspect authentic intaglio notes vs. CMYK halftone counterfeits without requiring physical cash.
- **Discreet Web Audio API Synthesizer**:
  - Pure in-browser acoustic rhythm synthesis mapped to each denomination without disruptive Text-to-Speech.

---

## Core Forensic Capabilities

1. **Multi-Task CNN Backbone (MobileNetV3-Small)**:
   - Optimized for CPU inference (<15ms forward pass latency).
   - **Head 1 (Denomination)**: Multi-class classification (Cross-Entropy loss) over standard banknote denominations + background.
   - **Head 2 (Counterfeit Probability)**: Single-logit binary prediction (BCE loss with Sigmoid) evaluating counterfeit likelihood.

2. **Optical Preprocessing Pipeline**:
   - **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Contrast enhancement applied in LAB color space across the Lightness (L) channel, preserving color integrity while highlighting latent embossing and paper grain.
   - **Screen-Flash Differencing Module**: Active optical inspection using laptop monitor illumination. Rapidly pulses a pure white screen canvas and captures difference image $\Delta I = |I_{flash} - I_{ambient}|$ to reveal embedded semi-translucent watermarks, security threads, and see-through registration marks.

3. **Super-Resolution Halftone Detector**:
   - **ESPCN (Efficient Sub-Pixel Convolutional Network)**: Neural super-resolution upscaling (2x magnification) on micro-texture patches directly on CPU.
   - **CMYK Plate Decomposition**: Separates RGB into Cyan, Magenta, Yellow, and Key (Black) plates with dark-pixel stabilization.
   - **2D FFT Rosette & Dot Detector**: Computes 2D Fast Fourier Transform power spectrum to detect periodic harmonic lattice peaks characteristic of commercial inkjet and color laser printer halftones vs. genuine continuous-line intaglio engraving.

4. **Discreet Audio Feedback**:
   - Synthesizes clean 16-bit 44.1 kHz PCM sine and harmonic pulse patterns directly in NumPy.
   - Replaces disruptive Text-to-Speech (TTS) with subtle tactile acoustic rhythms:
     - **10**: Single crisp high pulse (880 Hz, 80ms)
     - **20**: Double micro-pulse rhythm
     - **50**: Melodic rising third
     - **100**: Rising fifth chime
     - **200**: Triple staccato pulse
     - **500**: Resonant major triad chord
     - **2000**: Quadruple ascending arpeggio
     - **COUNTERFEIT / SUSPECT**: Urgent discordant dual-sawtooth buzz (180 Hz + 194 Hz)
   - Zero-latency asynchronous playback via Windows `winsound` in-memory buffers with automatic debouncing.

5. **Hardware Performance**:
   - Benchmarked on 1080p frames at **~27–33 FPS** on standard laptop CPU (>15 FPS requirement exceeded).

---

## Project Structure

```
The Looker/
├── core/
│   ├── __init__.py
│   ├── model.py              # MultiTaskMobileNetV3 & joint loss
│   ├── preprocessing.py      # LAB CLAHE & Screen-Flash Differencing
│   ├── super_res_halftone.py # ESPCN Super-Resolution & CMYK FFT Dot Detector
│   ├── audio_feedback.py     # NumPy acoustic waveform synthesizer & async engine
│   └── pipeline.py           # Unified real-time inspection engine
├── utils/
│   ├── __init__.py
│   └── visualizer.py         # Real-time HUD, diagnostic panels, and gauges
├── tests/
│   ├── __init__.py
│   ├── test_model.py         # Tensor shape and gradient backprop tests
│   ├── test_preprocessing.py # CLAHE & screen-flash tests
│   ├── test_super_res.py     # ESPCN upscaler & halftone FFT tests
│   ├── test_audio.py         # Acoustic pulse & WAV byte header tests
│   └── benchmark_fps.py      # End-to-end 1080p CPU FPS benchmark (>15 FPS)
├── synthetic_data.py         # Algorithmic banknote generator (genuine & counterfeit)
├── main.py                   # Real-time CLI & GUI application runner
└── README.md
```

---

## Execution Commands

### 1. Launch Aesthetic Forensic Web Studio (Recommended)
```bash
python run_web.py
```
This single autonomous command automatically:
- Checks and reclaims port 8000 from any orphaned processes.
- Pre-warms the neural networks (MobileNetV3, 2x ESPCN Super-Resolution, CLAHE, CMYK FFT).
- Launches the high-speed ASGI server (<25ms frame latency).
- Automatically pops open your default web browser directly to **`http://127.0.0.1:8000`**.

### 2. Run Complete Automated Test Suite (15 Tests)
```bash
python -m unittest discover tests
```

### 2. Run 1080p CPU FPS Throughput Benchmark
```bash
python tests/benchmark_fps.py
```

### 3. Run Live Interactive Simulation Mode
Cycles between genuine and counterfeit banknotes with interactive HUD and acoustic feedback:
```bash
python main.py --demo
```

### 4. Run with Physical Webcam
```bash
python main.py --camera 0
```

### 5. Inspect a Static Note Image
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

