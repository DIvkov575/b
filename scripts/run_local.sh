#!/bin/bash
set -e
echo "=== Multi-Target Binder Pipeline (Mock Mode) ==="
cd "$(dirname "$0")/.."
python3 -m src.pipeline --mock
echo "=== Complete ==="
