#!/usr/bin/env python3
"""Parse a trtexec log into a one-line CSV summary (throughput + latency).

Generic regex parser for the optional TensorRT path (vla/export_trt.py).

Usage:
    /usr/src/tensorrt/bin/trtexec --onnx=model.onnx --fp16 > results/trt.log
    python3 bench/parse_trtexec.py results/trt.log
"""
from __future__ import annotations

import argparse
import re
import sys

PATTERNS = {
    "throughput_qps": r"Throughput:\s*([\d.]+)\s*qps",
    "latency_mean_ms": r"Latency:.*?mean\s*=\s*([\d.]+)\s*ms",
    "latency_median_ms": r"Latency:.*?median\s*=\s*([\d.]+)\s*ms",
    "latency_p99_ms": r"Latency:.*?percentile\(99%\)\s*=\s*([\d.]+)\s*ms",
    "gpu_compute_mean_ms": r"GPU Compute Time:.*?mean\s*=\s*([\d.]+)\s*ms",
    "enqueue_mean_ms": r"Enqueue Time:.*?mean\s*=\s*([\d.]+)\s*ms",
}


def parse(text: str) -> dict:
    out = {}
    for key, pat in PATTERNS.items():
        m = re.search(pat, text, re.DOTALL)
        out[key] = float(m.group(1)) if m else None
    # heuristic: enqueue-bound if CPU enqueue >= GPU compute
    if out["enqueue_mean_ms"] and out["gpu_compute_mean_ms"]:
        out["enqueue_bound"] = out["enqueue_mean_ms"] >= out["gpu_compute_mean_ms"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    args = ap.parse_args()
    with open(args.log) as f:
        res = parse(f.read())
    print(",".join(res.keys()))
    print(",".join("" if v is None else str(v) for v in res.values()))
    if all(v is None for v in res.values()):
        sys.exit("warning: no trtexec metrics matched")


if __name__ == "__main__":
    main()
