#!/usr/bin/env python3
"""Classify Nsight Compute kernels as compute / bandwidth / latency-bound.

Parses an `ncu --csv` export (the wide "raw" layout) and applies simple,
tunable thresholds on the standard throughput metrics. Generic — no project-
specific scoring.

Usage:
    sudo ncu --set full --csv -o results/softmax_ncu python3 bench/bench_kernel.py --op softmax --once
    python3 bench/roofline.py results/softmax_ncu.csv
"""
from __future__ import annotations

import argparse
import csv
import sys

# metric name -> the columns ncu may emit it under (varies by version)
METRIC_ALIASES = {
    "sm_pct": ["Compute (SM) Throughput", "sm__throughput.avg.pct_of_peak_sustained_elapsed"],
    "dram_pct": ["Memory Throughput", "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"],
    "l1_hit": ["L1/TEX Hit Rate", "l1tex__t_sector_hit_rate.pct"],
    "l2_hit": ["L2 Hit Rate", "lts__t_sector_hit_rate.pct"],
    "duration": ["Duration", "gpu__time_duration.sum"],
}


def _find(headers, names):
    for n in names:
        if n in headers:
            return n
    return None


def classify(sm, dram, hi, lo):
    if sm is None or dram is None:
        return "unknown"
    if sm >= hi and sm >= dram:
        return "compute-bound"
    if dram >= hi and dram > sm:
        return "bandwidth-bound"
    if sm < lo and dram < lo:
        return "latency/occupancy-bound"
    return "balanced"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--hi", type=float, default=60.0, help="high %% of peak threshold")
    ap.add_argument("--lo", type=float, default=20.0, help="low %% of peak threshold")
    args = ap.parse_args()

    with open(args.csv, newline="") as f:
        # ncu csv often has a preamble; find the header row containing a known column
        rows = list(csv.reader(f))
    hdr_idx = next((i for i, r in enumerate(rows)
                    if any(c in r for c in METRIC_ALIASES["sm_pct"])), None)
    if hdr_idx is None:
        sys.exit("could not locate metric header row in ncu csv")
    headers = rows[hdr_idx]
    cols = {k: _find(headers, v) for k, v in METRIC_ALIASES.items()}
    kname = _find(headers, ["Kernel Name", "Demangled Name"])

    def get(row, key):
        c = cols[key]
        if c is None:
            return None
        try:
            return float(row[headers.index(c)].replace(",", ""))
        except (ValueError, IndexError):
            return None

    print(f"{'kernel':50} {'SM%':>6} {'DRAM%':>6} {'L2%':>6}  class")
    print("-" * 84)
    for row in rows[hdr_idx + 1:]:
        if not row or len(row) < len(headers):
            continue
        name = row[headers.index(kname)][:48] if kname else "?"
        sm, dram, l2 = get(row, "sm_pct"), get(row, "dram_pct"), get(row, "l2_hit")
        cls = classify(sm, dram, args.hi, args.lo)
        print(f"{name:50} {sm if sm is not None else '-':>6} "
              f"{dram if dram is not None else '-':>6} {l2 if l2 is not None else '-':>6}  {cls}")


if __name__ == "__main__":
    main()
