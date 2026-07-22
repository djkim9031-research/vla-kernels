#!/usr/bin/env bash
# Deterministic clocks for reproducible benchmarking on Jetson Thor.
# Sets max power mode + pins clocks, then snapshots tegrastats. Needs sudo.
# (Generic Jetson recipe — no proprietary rig config.)
set -euo pipefail

echo "[rig] setting MAXN power mode + locking clocks (sudo)"
sudo nvpmodel -m 0          # MAXN: all cores, max GPU clock ceiling
sudo jetson_clocks          # pin CPU/GPU/EMC to max, disable DVFS

echo "[rig] current state:"
sudo nvpmodel -q || true
echo "[rig] tegrastats snapshot (2s):"
timeout 2 tegrastats || true

echo "[rig] done. Re-run after reboot; 'sudo jetson_clocks --restore' to undo."
