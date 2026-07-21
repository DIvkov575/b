"""
Upload local .npz protein data files to S3.

Usage:
    python scripts/upload_to_s3.py \
        --src /home/dima/data/boltz/rcsb_processed_msa \
        --bucket my-pfold-bucket \
        --prefix data/rcsb_processed_msa \
        [--workers 32] \
        [--dry-run]

The script is resumable: it skips files that already exist in S3 (checked via
a local manifest so you don't pay for 151k HEAD requests on re-runs).
"""

import argparse
import hashlib
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm


def s3_key(prefix: str, filename: str) -> str:
    return f"{prefix.rstrip('/')}/{filename}"


def upload_file(s3_client, local_path: Path, bucket: str, key: str) -> tuple[str, bool, str]:
    """Returns (key, success, error_msg)."""
    try:
        s3_client.upload_file(str(local_path), bucket, key)
        return key, True, ""
    except Exception as e:
        return key, False, str(e)


def load_manifest(manifest_path: Path) -> set[str]:
    if manifest_path.exists():
        with open(manifest_path) as f:
            return set(json.load(f))
    return set()


def save_manifest(manifest_path: Path, uploaded: set[str]):
    with open(manifest_path, "w") as f:
        json.dump(sorted(uploaded), f)


def main():
    parser = argparse.ArgumentParser(description="Upload .npz files to S3")
    parser.add_argument("--src", required=True, help="Local directory containing .npz files")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--prefix", default="data/rcsb_processed_msa", help="S3 key prefix")
    parser.add_argument("--workers", type=int, default=32, help="Parallel upload threads")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be uploaded, don't upload")
    parser.add_argument("--manifest", default=".upload_manifest.json", help="Path to resume manifest")
    parser.add_argument("--profile", default=None, help="AWS profile name")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    args = parser.parse_args()

    src = Path(args.src)
    if not src.is_dir():
        print(f"ERROR: {src} is not a directory")
        sys.exit(1)

    files = sorted(src.glob("*.npz"))
    if not files:
        print(f"No .npz files found in {src}")
        sys.exit(1)

    print(f"Found {len(files):,} .npz files in {src}")

    manifest_path = Path(args.manifest)
    already_uploaded = load_manifest(manifest_path)
    to_upload = [f for f in files if f.name not in already_uploaded]
    print(f"Already uploaded: {len(already_uploaded):,}  |  Remaining: {len(to_upload):,}")

    if args.dry_run:
        for f in to_upload[:10]:
            print(f"  would upload: {f.name} -> s3://{args.bucket}/{s3_key(args.prefix, f.name)}")
        if len(to_upload) > 10:
            print(f"  ... and {len(to_upload) - 10} more")
        return

    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    s3 = session.client("s3")

    # Ensure bucket exists
    try:
        s3.head_bucket(Bucket=args.bucket)
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "404":
            print(f"Bucket {args.bucket} does not exist. Creating...")
            if args.region == "us-east-1":
                s3.create_bucket(Bucket=args.bucket)
            else:
                s3.create_bucket(
                    Bucket=args.bucket,
                    CreateBucketConfiguration={"LocationConstraint": args.region}
                )
        else:
            raise

    lock = threading.Lock()
    uploaded = set(already_uploaded)
    failed = []

    # Thread-local S3 clients for better throughput
    thread_local = threading.local()

    def get_client():
        if not hasattr(thread_local, "s3"):
            thread_local.s3 = session.client("s3")
        return thread_local.s3

    def upload_task(local_path: Path):
        key = s3_key(args.prefix, local_path.name)
        client = get_client()
        return upload_file(client, local_path, args.bucket, key)

    with tqdm(total=len(to_upload), unit="file", desc="Uploading") as pbar:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(upload_task, f): f for f in to_upload}
            for future in as_completed(futures):
                key, success, err = future.result()
                filename = Path(key).name
                if success:
                    with lock:
                        uploaded.add(filename)
                else:
                    failed.append((filename, err))
                    print(f"\nFAILED: {filename}: {err}")
                pbar.update(1)

                # Save manifest every 500 files so progress isn't lost on crash
                if len(uploaded) % 500 == 0:
                    with lock:
                        save_manifest(manifest_path, uploaded)

    save_manifest(manifest_path, uploaded)

    print(f"\nDone. Uploaded: {len(uploaded) - len(already_uploaded):,}  |  Failed: {len(failed):,}")
    if failed:
        print("Failed files:")
        for name, err in failed[:20]:
            print(f"  {name}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
