#!/bin/bash
set -e
echo "=== Multi-Target Binder Pipeline (GPU Mode) ==="
cd "$(dirname "$0")/.."
export RFDIFFUSION_PATH="${RFDIFFUSION_PATH:-/home/ubuntu/RFdiffusion/scripts/run_inference.py}"
python3 -m src.pipeline --config "${1:-configs/targets.yaml}" --defaults "${2:-configs/defaults.yaml}"
echo "=== Complete ==="
