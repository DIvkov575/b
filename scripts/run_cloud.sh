#!/bin/bash
set -e
echo "=== Multi-Target Binder Pipeline (GPU Mode) ==="
cd "$(dirname "$0")/.."
python3 -m src.pipeline --config configs/targets.yaml --defaults configs/defaults.yaml
echo "=== Complete ==="
