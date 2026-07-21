#!/bin/bash
# ============================================================
# Downloads Boltz's pre-processed RCSB MSA .npz shards
# (the raw data consumed by msa_data.py) into a local directory.
#
# Ported from PFold (github.com/DIvkov575/PFold, commit f44eecc)
# scripts/instance_setup.sh — extracted as a standalone download step.
#
# Usage:
#   ./download_boltz_msa.sh [DATA_DIR]
#   DATA_DIR defaults to ./data/rcsb_processed_msa
# ============================================================
set -euo pipefail

DATA_DIR="${1:-./data/rcsb_processed_msa}"

# Boltz public S3 (us-east-2) — no auth needed, ~250GB.
BOLTZ_TAR_URL="https://boltz1.s3.us-east-2.amazonaws.com/rcsb_processed_msa.tar"

mkdir -p "$DATA_DIR"
LOCAL_FILES=$(find "$DATA_DIR" -name "*.npz" | wc -l)
echo "Local .npz files already present: ${LOCAL_FILES}"

if [ "$LOCAL_FILES" -lt 1000 ]; then
    echo "Downloading rcsb_processed_msa.tar from Boltz S3 (~250GB) into ${DATA_DIR}..."
    # Stream directly into tar to avoid needing double the disk space.
    curl -L --progress-bar "$BOLTZ_TAR_URL" \
        | tar -x --strip-components=1 -C "$DATA_DIR"
    echo "Extract complete. $(find "$DATA_DIR" -name "*.npz" | wc -l) files in ${DATA_DIR}."
else
    echo "Data already present (${LOCAL_FILES} files), skipping download."
fi
