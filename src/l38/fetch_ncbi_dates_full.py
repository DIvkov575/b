"""One-shot script: fetch NCBI createdates for every accession in the real
PhaVIP PVP/non-PVP dataset, cache to disk (this is a ~5-10min network
operation across ~82K accessions -- run once, cache, reuse).

Saves incrementally after every batch (not just at the end) after a real
run crashed on an unretried network timeout partway through and lost all
progress -- see ncbi_dates.py's fetch_createdates docstring.
"""
import json
import time
from pathlib import Path

from src.l38.ncbi_dates import fetch_createdates, parse_fasta_with_accessions

DATA_DIR = Path(__file__).resolve().parent / "data_cache" / "phavip_real"
OUT_PATH = DATA_DIR / "ncbi_createdates.json"


def save_progress(dates: dict) -> None:
    # Write to a temp file then rename -- avoids a truncated/corrupt JSON
    # file if the process is killed mid-write.
    tmp_path = OUT_PATH.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(dates, f)
    tmp_path.replace(OUT_PATH)


def main():
    pvp_pairs = parse_fasta_with_accessions(DATA_DIR / "pvp.fa")
    non_pvp_pairs = parse_fasta_with_accessions(DATA_DIR / "non_pvp.fa")
    all_accessions = sorted({a for a, _ in pvp_pairs} | {a for a, _ in non_pvp_pairs})
    print(f"pvp: {len(pvp_pairs)}, non_pvp: {len(non_pvp_pairs)}, unique accessions: {len(all_accessions)}", flush=True)

    # Resume support: skip accessions we already resolved in a prior
    # (possibly crashed) run, rather than re-querying NCBI for everything.
    already_have = {}
    if OUT_PATH.exists():
        with open(OUT_PATH) as f:
            already_have = json.load(f)
        print(f"resuming: {len(already_have)} dates already cached from a prior run", flush=True)
    remaining = [a for a in all_accessions if a not in already_have]
    print(f"fetching {len(remaining)} remaining accessions", flush=True)

    t0 = time.time()

    def on_progress(done, total, dates_so_far):
        merged = {**already_have, **dates_so_far}
        save_progress(merged)
        if done % 20 == 0 or done == total:
            elapsed = time.time() - t0
            print(f"  batch {done}/{total} ({elapsed:.0f}s elapsed, {len(merged)} dates cached)", flush=True)

    dates = fetch_createdates(remaining, on_progress=on_progress)
    final = {**already_have, **dates}
    save_progress(final)

    all_accessions_set = set(all_accessions)
    missing = all_accessions_set - set(final)
    print(f"\nDONE: fetched {len(final)}/{len(all_accessions)} dates in {time.time()-t0:.0f}s", flush=True)
    print(f"missing: {len(missing)} ({100*len(missing)/len(all_accessions):.1f}%)", flush=True)
    print(f"saved to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
