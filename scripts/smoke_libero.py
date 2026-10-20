#!/usr/bin/env python
"""Milestone 1: actual official task, official initial state, RGB and references."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vlm_tamp.libero import collect

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/smoke")
    parser.add_argument("--gl", choices=["egl", "glfw", "osmesa"], default="egl")
    parser.add_argument("--suite", default="libero_goal")
    parser.add_argument("--task", type=int, default=0)
    args = parser.parse_args()
    rows = collect(args.output, suites=[args.suite], tasks=[args.task], gl=args.gl)
    print(f"PASS: {len(rows)} real LIBERO observation(s) saved in {Path(args.output).resolve()}")
