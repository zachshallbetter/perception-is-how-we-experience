#!/usr/bin/env python3
"""M1 timing (titan-node#699): the Sensory Perceiver's CPU forward pass on windows of the household's shape, in ms per
window (p50, p95), against M3's bar of ≤ 50 ms. CPU only: CUDA_VISIBLE_DEVICES is cleared before torch loads, so no
GPU is touched. The model is untrained here: M1 times it; M3 scores it.

  python household/m1_timing.py [--features 8] [--steps 15] [--dim 256] [--depth 2] [--runs 200]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # before torch: never a GPU
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=int, default=8)   # per minute: heating, setpoint, inside, tv_on, home×n, time
    ap.add_argument("--steps", type=int, default=15)     # a 15-minute window at one step a minute
    ap.add_argument("--dim", type=int, default=256)      # the CL runs' width
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    import torch
    assert not torch.cuda.is_available(), "a GPU is visible: refusing (CPU only)"
    torch.set_num_threads(a.threads)
    from src.models.core import SensoryPerceiver
    m = SensoryPerceiver({"ha": {"input_dim": a.features, "use_conv": False}}, dim=a.dim, num_heads=4,
                         mlp_dim=a.dim * 2, depth=a.depth, replay_buffer_size=16).eval()
    x = {"ha": torch.randn(1, a.steps, a.features)}
    with torch.no_grad():
        for _ in range(10):
            m(x, update_buffer=False)
        times = []
        for _ in range(a.runs):
            t0 = time.perf_counter()
            m(x, update_buffer=False)
            times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    print(json.dumps({"params": sum(p.numel() for p in m.parameters()), "dim": a.dim, "depth": a.depth,
                      "steps": a.steps, "features": a.features, "threads": a.threads,
                      "ms_p50": round(statistics.median(times), 2), "ms_p95": round(times[int(0.95 * len(times))], 2),
                      "bar_ms": 50}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
