#!/usr/bin/env bash
# 3-tier profiling driver: trtexec baseline -> nsys timeline -> ncu hot kernels.
# Parameterized by an ONNX path (for the TensorRT/VLA path). Generic flow.
#
# Usage: bench/three_tier.sh model.onnx [outdir]
set -euo pipefail

ONNX="${1:?usage: three_tier.sh model.onnx [outdir]}"
OUT="${2:-results}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
mkdir -p "$OUT"

echo "=== Tier 1: trtexec baseline (fp16) ==="
"$TRTEXEC" --onnx="$ONNX" --fp16 --iterations=200 --avgRuns=200 \
  | tee "$OUT/trt.log"
python3 "$(dirname "$0")/parse_trtexec.py" "$OUT/trt.log" | tee "$OUT/trt_summary.csv"

echo "=== Tier 2: Nsight Systems timeline ==="
nsys profile --force-overwrite=true -o "$OUT/timeline" \
  "$TRTEXEC" --onnx="$ONNX" --fp16 --iterations=50 || echo "nsys skipped"

echo "=== Tier 3: Nsight Compute (hottest kernels; needs sudo on Tegra) ==="
echo "  run manually, e.g.:"
echo "  sudo ncu --set full --csv -o $OUT/ncu --launch-count 20 \\"
echo "      $TRTEXEC --onnx=$ONNX --fp16 --iterations=1"
