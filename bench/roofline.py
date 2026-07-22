#!/usr/bin/env python3
"""Classify Nsight Compute kernels as compute / bandwidth / latency-bound.

Parses an `ncu --csv` export (long format: one row per kernel-metric pair,
"Metric Name" / "Metric Value" columns) and applies tunable thresholds on the
standard Speed-of-Light throughput metrics.

Usage:
    sudo ncu --set full -k "regex:softmax" --csv python3 ... > results/ncu.csv
    python3 bench/roofline.py results/ncu.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import sys

# Metric Name -> short key (values are % unless noted)
METRICS = {
    "Compute (SM) Throughput": "sm_pct",
    "Memory Throughput": "mem_pct",          # SOL max over memory pipelines
    "DRAM Throughput": "dram_pct",           # may be absent on Tegra iGPU
    "L1/TEX Hit Rate": "l1_hit",
    "L2 Hit Rate": "l2_hit",
    "Achieved Occupancy": "occupancy",
    "Duration": "duration_ns",               # ns
}


def classify(sm, mem, hi, lo):
    if sm is None or mem is None:
        return "unknown"
    if sm >= hi and sm >= mem:
        return "compute-bound"
    if mem >= hi and mem > sm:
        return "bandwidth-bound"
    if sm < lo and mem < lo:
        return "latency/occupancy-bound"
    return "balanced"


def parse(path):
    """Return {(launch_id, kernel_name): {key: float}} preserving launch order."""
    with open(path, newline="") as f:
        # strip ncu preamble (==PROF== / ==WARNING== lines) before the header
        lines = [ln for ln in f if not ln.startswith(("==", "\x00"))]
    kernels: dict = {}
    for row in csv.DictReader(io.StringIO("".join(lines))):
        name = (row.get("Kernel Name") or "?").strip()
        kid = row.get("ID", "?")
        metric = (row.get("Metric Name") or "").strip()
        if metric not in METRICS:
            continue
        raw = (row.get("Metric Value") or "").replace(",", "").strip()
        try:
            val = float(raw)
        except ValueError:
            continue
        kernels.setdefault((kid, name), {})[METRICS[metric]] = val
    return kernels


def short(name: str, width: int) -> str:
    # "void <unnamed>::softmax_v3<float>(const T1 *, T1 *, int, int)" -> "softmax_v3<float>"
    n = name.split("::")[-1].split("(")[0].strip()
    return (n or name)[:width]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--hi", type=float, default=60.0, help="high %% of peak threshold")
    ap.add_argument("--lo", type=float, default=20.0, help="low %% of peak threshold")
    args = ap.parse_args()

    kernels = parse(args.csv)
    if not kernels:
        sys.exit("no kernel metrics found — is this an `ncu --csv` long-format export?")

    hdr = f"{'kernel':30} {'dur_us':>8} {'SM%':>6} {'MEM%':>6} {'DRAM%':>6} {'L2hit%':>7} {'occ%':>6}  class"
    print(hdr)
    print("-" * len(hdr))
    for (kid, name), m in kernels.items():
        dur = m.get("duration_ns")
        row = [
            f"{short(name, 30):30}",
            f"{dur/1000:8.1f}" if dur is not None else f"{'-':>8}",
        ]
        for key in ("sm_pct", "mem_pct", "dram_pct", "l2_hit"):
            v = m.get(key)
            row.append(f"{v:6.1f}" if v is not None else f"{'-':>6}")
        occ = m.get("occupancy")
        row.append(f"{occ:6.1f}" if occ is not None else f"{'-':>6}")
        row.append("  " + classify(m.get("sm_pct"), m.get("mem_pct"), args.hi, args.lo))
        print(" ".join(row))


if __name__ == "__main__":
    main()
