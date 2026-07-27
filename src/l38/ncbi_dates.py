"""L39 v3 refinement: reconstruct ESM-PVP's exact pre/post-Dec-2020 RefSeq
time-split, since their own committed data was never pushed to git (see
docs/L39_PHAGE_ESM_FINETUNE.md's Refinement section -- their retrain/
scripts reference a local data/binary/train.txt that isn't in their repo).

RefSeq accession numbers ARE public, and NCBI's E-utilities esummary
endpoint exposes each record's `createdate` -- this reconstructs the same
kind of pre/post-cutoff split their paper describes, from public metadata,
rather than downloading a file that doesn't exist anywhere.
"""
import time
from pathlib import Path
from typing import Dict, List, Tuple

import requests

ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
BATCH_SIZE = 200  # NCBI allows up to ~500 IDs/request without an API key; stay comfortably under
REQUEST_DELAY_SECONDS = 0.35  # NCBI's documented limit is 3 req/sec without an API key


def parse_fasta_with_accessions(path: Path) -> List[Tuple[str, str]]:
    """Returns [(accession, sequence), ...], preserving the accession (first
    whitespace-delimited token after '>') that plain phage_data.parse_fasta
    discards."""
    records = []
    accession = None
    seq_lines: List[str] = []

    def flush():
        if accession is not None:
            records.append((accession, "".join(seq_lines)))

    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                flush()
                accession = line[1:].split()[0] if line[1:].split() else line[1:]
                seq_lines = []
            else:
                seq_lines.append(line)
    flush()
    return records


def fetch_createdates(
    accessions: List[str], batch_size: int = BATCH_SIZE, delay: float = REQUEST_DELAY_SECONDS,
    max_retries: int = 5, on_progress=None,
) -> Dict[str, str]:
    """Returns {accession: createdate_string ('YYYY/MM/DD')} for every
    accession NCBI recognizes; silently omits any accession NCBI doesn't
    return a record for (stale/withdrawn/merged accessions do happen at
    this scale -- the caller decides how to handle missing entries).

    Retries each batch on network failure (timeout, connection error) with
    exponential backoff, since a single flaky request out of ~400 batches
    should not lose the entire run -- confirmed necessary after a real
    30s-read-timeout crashed an unprotected version of this function with
    zero results saved partway through the real ~82K-accession dataset.

    on_progress(batches_done, total_batches, dates_so_far), if given, is
    called after every batch -- lets the caller checkpoint incrementally
    instead of only writing output after the full call returns.
    """
    dates: Dict[str, str] = {}
    total_batches = -(-len(accessions) // batch_size)

    for batch_idx, i in enumerate(range(0, len(accessions), batch_size)):
        batch = accessions[i : i + batch_size]

        for attempt in range(max_retries):
            try:
                resp = requests.get(
                    ESUMMARY_URL,
                    params={"db": "protein", "id": ",".join(batch), "retmode": "json"},
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                break
            except (requests.exceptions.RequestException, ValueError) as e:
                if attempt == max_retries - 1:
                    raise
                backoff = delay * (2 ** attempt)
                print(f"  batch {batch_idx} attempt {attempt+1} failed ({e}); retrying in {backoff:.1f}s", flush=True)
                time.sleep(backoff)
        else:
            continue  # unreachable given the raise above, but keeps intent explicit

        for uid in data["result"].get("uids", []):
            record = data["result"][uid]
            accession_version = record.get("accessionversion")
            if accession_version:
                dates[accession_version] = record["createdate"]

        if on_progress is not None:
            on_progress(batch_idx + 1, total_batches, dates)

        time.sleep(delay)
    return dates


def split_by_date_cutoff(
    accession_seq_pairs: List[Tuple[str, str]], dates: Dict[str, str], cutoff: str = "2020/12/01"
) -> Tuple[List[str], List[str], List[str]]:
    """Returns (train_sequences, test_sequences, missing_accessions).
    train = createdate < cutoff, test = createdate >= cutoff, matching
    ESM-PVP's paper's described pre/post-Dec-2020 split. Sequences whose
    accession has no resolvable date go to `missing_accessions` (reported,
    not silently dropped or guessed into either split)."""
    train, test, missing = [], [], []
    for accession, seq in accession_seq_pairs:
        date = dates.get(accession)
        if date is None:
            missing.append(accession)
        elif date < cutoff:
            train.append(seq)
        else:
            test.append(seq)
    return train, test, missing
