"""
tests/benchmark_fps.py - End-to-end performance and FPS verification on CPU.

Requirement:
Optimize inference to >15 FPS under standard laptop CPU constraints with 1080p frames.
"""

import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import torch

from core.pipeline import CurrencyInspectionPipeline
from utils.visualizer import HUDVisualizer
from synthetic_data import SyntheticCurrencyGenerator


def run_cpu_fps_benchmark(iterations: int = 50, frame_resolution=(1920, 1080)) -> float:
    print(f"\n{'=' * 65}")
    print(f" BENCHMARK: CPU Inference & Throughput Verification")
    print(f" Resolution: {frame_resolution[0]}x{frame_resolution[1]} | Iterations: {iterations}")
    print(f" PyTorch Device: CPU | Threads: {torch.get_num_threads()}")
    print(f"{'=' * 65}")

    pipeline = CurrencyInspectionPipeline(device="cpu", enable_audio=False)
    visualizer = HUDVisualizer(panel_width=340)

    # Generate synthetic 1080p test frame
    w, h = frame_resolution
    test_note = SyntheticCurrencyGenerator.generate_banknote(
        denomination="500",
        is_counterfeit=True,
        width=w,
        height=h
    )

    # Warmup runs (JIT compilation, cache population, BLAS init)
    print("[Benchmark] Running 5 warmup passes...")
    for _ in range(5):
        pipeline.process_frame(test_note)

    # Profiling timed iterations
    print(f"[Benchmark] Measuring {iterations} consecutive full-pipeline passes...")
    latencies = []
    t_start_total = time.perf_counter()

    for i in range(iterations):
        t0 = time.perf_counter()
        results = pipeline.process_frame(test_note)
        # Render visualizer HUD as in live production
        _ = visualizer.draw_hud(test_note, results)
        t_elapsed = time.perf_counter() - t0
        latencies.append(t_elapsed * 1000.0)

    total_time = time.perf_counter() - t_start_total
    avg_latency = np.mean(latencies)
    median_latency = np.median(latencies)
    p95_latency = np.percentile(latencies, 95)
    fps = iterations / total_time

    print("\n" + "-" * 50)
    print(" PERFORMANCE RESULTS:")
    print("-" * 50)
    print(f" Mean Latency    : {avg_latency:.2f} ms")
    print(f" Median Latency  : {median_latency:.2f} ms")
    print(f" 95th Percentile : {p95_latency:.2f} ms")
    print(f" Achieved FPS    : {fps:.2f} FPS")
    print(f" Target FPS      : > 15.0 FPS")
    print("-" * 50)

    if fps >= 15.0:
        print(f" [PASS] Meets & exceeds real-time CPU constraint (>15 FPS): {fps:.1f} FPS")
    else:
        print(f" [FAIL] Below target FPS: {fps:.1f} FPS")

    return fps


if __name__ == "__main__":
    fps = run_cpu_fps_benchmark(iterations=50)
    if fps < 15.0:
        exit(1)
