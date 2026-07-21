"""Fetches one canonical single-chain sequence per structure from RCSB, for the
structures present in a directory of Boltz-processed .npz shards. This is the
"no MSA augmentation" baseline data source for the L40 ablation: same structure
population as the Boltz side, but without the extra homolog sequences an MSA
shard provides.

Output: a JSONL file, one line per structure:
    {"structure_id": "8u3n_a", "pdb_id": "8U3N", "chain_id": "A", "sequence": "MGSS..."}

Resumable: structures already present in --out are skipped on re-run, same
manifest-based approach as upload_to_s3.py.

Usage:
    .venv-l38/bin/python -m src.l40.fetch_rcsb_structures \
        --npz-dir /path/to/rcsb_processed_msa --out src/l40/data_cache/rcsb_baseline.jsonl \
        --max-structures 2500
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Set, Tuple

import requests

FASTA_URL = "https://www.rcsb.org/fasta/chain/{pdb_id}.{chain_id}"


def parse_structure_id(filename: str) -> Tuple[str, str]:
    stem = filename[:-4] if filename.endswith(".npz") else filename
    if "_" not in stem:
        raise ValueError(f"expected '{{pdbid}}_{{chain}}.npz', got: {filename}")
    pdb_id, chain_id = stem.rsplit("_", 1)
    return pdb_id.upper(), chain_id.upper()


def parse_fasta_response(text: str) -> str:
    lines = [line.strip() for line in text.strip().splitlines()]
    seq_lines = [line for line in lines if line and not line.startswith(">")]
    if not seq_lines:
        raise ValueError(f"no sequence lines in FASTA response: {text!r}")
    return "".join(seq_lines)


def fetch_one(session, pdb_id: str, chain_id: str) -> str:
    url = FASTA_URL.format(pdb_id=pdb_id, chain_id=chain_id)
    response = session.get(url, timeout=10)
    if response.status_code != 200:
        raise RuntimeError(f"{url} -> HTTP {response.status_code}")
    return parse_fasta_response(response.text)


def load_existing_ids(path: Path) -> Set[str]:
    if not Path(path).exists():
        return set()
    ids = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["structure_id"])
    return ids


def main():
    parser = argparse.ArgumentParser(description="Fetch RCSB baseline sequences matching a Boltz .npz directory")
    parser.add_argument("--npz-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-structures", type=int, default=2500)
    parser.add_argument("--sleep-seconds", type=float, default=0.1, help="delay between requests")
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    files = sorted(f for f in os.listdir(args.npz_dir) if f.endswith(".npz"))[:args.max_structures]
    print(f"Targeting {len(files)} structures from {args.npz_dir}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    already_done = load_existing_ids(out_path)
    print(f"Already fetched: {len(already_done)}")

    session = requests.Session()
    fetched, failed = 0, []

    with open(out_path, "a") as out_f:
        for filename in files:
            structure_id = filename[:-4]
            if structure_id in already_done:
                continue

            pdb_id, chain_id = parse_structure_id(filename)
            last_err = None
            for attempt in range(args.retries):
                try:
                    sequence = fetch_one(session, pdb_id, chain_id)
                    out_f.write(json.dumps({
                        "structure_id": structure_id, "pdb_id": pdb_id,
                        "chain_id": chain_id, "sequence": sequence,
                    }) + "\n")
                    out_f.flush()
                    fetched += 1
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    time.sleep(args.sleep_seconds * (attempt + 1))

            if last_err is not None:
                failed.append((structure_id, str(last_err)))
                print(f"FAILED: {structure_id}: {last_err}")

            time.sleep(args.sleep_seconds)

            if fetched % 100 == 0 and fetched > 0:
                print(f"  fetched {fetched}/{len(files) - len(already_done)}")

    print(f"Done. Fetched: {fetched}  Failed: {len(failed)}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
