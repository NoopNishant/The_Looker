"""
main.py - Real-Time Currency Recognition and Counterfeit Detection Application.

Usage:
  python main.py --demo                      # Run live simulation mode (cycles genuine/counterfeit)
  python main.py --camera 0                  # Run with physical webcam
  python main.py --image path/to/note.jpg    # Analyze static image
  python main.py --headless --demo           # Run benchmark/headless demo loop

Controls:
  [SPACE] : Trigger active screen-flash watermark acquisition
  [M]     : Toggle audio feedback mute/unmute
  [S]     : Save diagnostic snapshot to disk
  [Q]/ESC : Exit application
"""

import argparse
import os
import sys
import time
import cv2
import numpy as np

from core.pipeline import CurrencyInspectionPipeline
from core.preprocessing import ScreenFlashDifferencing
from utils.visualizer import HUDVisualizer
from synthetic_data import SyntheticCurrencyGenerator


def run_live_stream(
    pipeline: CurrencyInspectionPipeline,
    visualizer: HUDVisualizer,
    camera_id: int = 0,
    demo_mode: bool = False,
    headless: bool = False,
    max_frames: int = 0,
    output_dir: str = "output"
) -> None:
    """
    Main real-time execution loop.
    Supports physical webcam or synthetic currency stream.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = None
    if not demo_mode:
        print(f"[System] Opening webcam device ID {camera_id}...")
        cap = cv2.VideoCapture(camera_id)
        # Attempt 1080p configuration
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        if not cap.isOpened():
            print(f"[Warning] Webcam {camera_id} unavailable. Falling back to synthetic DEMO stream.")
            demo_mode = True

    cached_flash_delta = None
    frame_idx = 0
    t0 = time.time()

    demo_denominations = ["10", "20", "50", "100", "200", "500", "2000"]
    demo_idx = 0
    is_counterfeit_cycle = False
    switch_interval = 75  # Switch simulated note every 75 frames

    window_name = "The Looker - Currency Recognition & Counterfeit Detection"
    if not headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

    print("\n" + "=" * 65)
    print(" THE LOOKER: Real-Time Currency Recognition & Counterfeit Guard")
    print(" Architecture: MobileNetV3 + CLAHE + Screen-Flash + ESPCN + CMYK FFT")
    print(" Audio: NumPy Discreet Acoustic Byte-Rhythms (No TTS)")
    print("=" * 65)
    print(" Controls:")
    print("   [SPACE] -> Trigger active Screen-Flash Watermark capture")
    print("   [M]     -> Toggle Audio Mute")
    print("   [S]     -> Save Diagnostic Snapshot")
    print("   [Q]     -> Quit")
    print("=" * 65 + "\n")

    try:
        while True:
            # 1. Acquire Frame
            if demo_mode:
                # Cycle simulated banknotes
                if frame_idx % switch_interval == 0 and frame_idx > 0:
                    demo_idx = (demo_idx + 1) % len(demo_denominations)
                    if demo_idx == 0:
                        is_counterfeit_cycle = not is_counterfeit_cycle

                denom_name = demo_denominations[demo_idx]
                raw_frame = SyntheticCurrencyGenerator.generate_banknote(
                    denomination=denom_name,
                    is_counterfeit=is_counterfeit_cycle,
                    width=640,
                    height=360
                )
                time.sleep(0.015)  # Simulate frame timing (~60 FPS)
            else:
                ret, raw_frame = cap.read()
                if not ret or raw_frame is None:
                    print("[Warning] Empty frame received from webcam. Exiting.")
                    break

            # 2. Pipeline Inspection Pass
            results = pipeline.process_frame(
                bgr_frame=raw_frame,
                cached_flash_delta=cached_flash_delta
            )

            # 3. Render HUD & Diagnostic Overlay
            composite = visualizer.draw_hud(raw_frame, results)

            # 4. Display Frame
            if not headless:
                cv2.imshow(window_name, composite)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord('q'), ord('Q'), 27):  # ESC or Q
                    print("[System] User quit requested.")
                    break

                elif key == ord(' '):  # SPACE -> Trigger screen-flash differencing
                    print("[Action] Triggering Screen-Flash Differencing...")
                    if demo_mode:
                        amb, fl = ScreenFlashDifferencing.simulate_screen_flash(raw_frame)
                        cached_flash_delta = pipeline.flash_differencer.compute_difference(amb, fl)
                        print(f"[Action] Flash Delta computed! Transmission density: {cached_flash_delta['transmission_density']:.3f}")
                    elif cap is not None:
                        try:
                            cached_flash_delta = pipeline.flash_differencer.capture_with_flash(cap)
                            print(f"[Action] Live Flash Delta captured! Transmission density: {cached_flash_delta['transmission_density']:.3f}")
                        except Exception as e:
                            print(f"[Error] Screen flash capture failed: {e}")

                elif key in (ord('m'), ord('M')):  # Mute audio
                    pipeline.audio_engine.enabled = not pipeline.audio_engine.enabled
                    state = "ENABLED" if pipeline.audio_engine.enabled else "MUTED"
                    print(f"[Action] Audio feedback {state}")

                elif key in (ord('s'), ord('S')):  # Save snapshot
                    snap_path = os.path.join(output_dir, f"snapshot_{int(time.time())}.png")
                    cv2.imwrite(snap_path, composite)
                    print(f"[Action] Saved inspection snapshot to {snap_path}")

            frame_idx += 1
            if max_frames > 0 and frame_idx >= max_frames:
                print(f"[System] Completed {max_frames} frames benchmark target.")
                break

    finally:
        total_time = time.time() - t0
        avg_fps = frame_idx / max(1e-4, total_time)
        print(f"\n[Telemetry] Total frames: {frame_idx} | Elapsed: {total_time:.2f}s | Average FPS: {avg_fps:.2f}")

        if cap is not None:
            cap.release()
        if not headless:
            cv2.destroyAllWindows()


def inspect_image_file(
    pipeline: CurrencyInspectionPipeline,
    visualizer: HUDVisualizer,
    image_path: str,
    output_path: str = "output/inspection_result.png"
) -> None:
    """Analyze a single banknote image and save the complete multi-sensor diagnostic card."""
    if not os.path.exists(image_path):
        print(f"[Error] Image not found: {image_path}")
        sys.exit(1)

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[Error] Failed to read image: {image_path}")
        sys.exit(1)

    print(f"[System] Inspecting image: {image_path} ({frame.shape[1]}x{frame.shape[0]})...")

    # Simulate flash differencing on static image
    amb, fl = ScreenFlashDifferencing.simulate_screen_flash(frame)
    flash_delta = pipeline.flash_differencer.compute_difference(amb, fl)

    results = pipeline.process_frame(frame, cached_flash_delta=flash_delta)
    composite = visualizer.draw_hud(frame, results)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, composite)

    print("-" * 50)
    print(" INSPECTION SUMMARY REPORT")
    print("-" * 50)
    print(f" Denomination Detected : {results['denomination']} (Confidence: {results['confidence'] * 100:.2f}%)")
    print(f" Counterfeit Score     : {results['composite_counterfeit_score']:.3f} (Threshold: {pipeline.counterfeit_threshold:.2f})")
    print(f" CMYK Halftone Score   : {results['halftone_score']:.3f}")
    print(f" Verdict               : {results['verdict']}")
    print(f" Processing Latency    : {results['latency_ms']:.2f} ms")
    print(f" Saved Diagnostic HUD  : {output_path}")
    print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description="The Looker - Currency Recognition and Counterfeit Detection")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--demo", action="store_true", help="Run with synthetic simulated currency video stream")
    parser.add_argument("--image", type=str, default=None, help="Inspect a static currency image file")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI windows")
    parser.add_argument("--frames", type=int, default=0, help="Stop after N frames (0 for infinite)")
    parser.add_argument("--mute", action="store_true", help="Disable audio feedback")
    parser.add_argument("--weights", type=str, default=None, help="Path to custom model weights")
    parser.add_argument("--output", type=str, default="output", help="Output directory for snapshots")

    args = parser.parse_args()

    pipeline = CurrencyInspectionPipeline(
        model_path=args.weights,
        device="cpu",
        enable_audio=not args.mute
    )
    visualizer = HUDVisualizer(panel_width=340)

    if args.image:
        out_file = os.path.join(args.output, f"inspect_{os.path.basename(args.image)}")
        inspect_image_file(pipeline, visualizer, args.image, out_file)
    else:
        run_live_stream(
            pipeline=pipeline,
            visualizer=visualizer,
            camera_id=args.camera,
            demo_mode=args.demo,
            headless=args.headless,
            max_frames=args.frames,
            output_dir=args.output
        )


if __name__ == "__main__":
    main()

