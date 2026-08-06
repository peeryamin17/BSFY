"""Download the BPCC Kashmiri parallel corpus (en <-> kas_Arab) from HuggingFace.

The BPCC dataset is gated, so you must be logged in to HuggingFace first:

    huggingface-cli login

Usage:
    python data/download_data.py [--out data/raw] [--force]
"""
import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "ai4bharat/BPCC"
KAS_FILE = "bpcc-seed-latest/kas_Arab.tsv"


def main():
    parser = argparse.ArgumentParser(description="Download BPCC Kashmiri corpus")
    parser.add_argument(
        "--out", default="data/raw", help="Output directory for the downloaded file"
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if the file exists"
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "kas_Arab.tsv"

    if dest.exists() and not args.force:
        print(f"File already exists: {dest}. Use --force to re-download.")
        return

    print(f"Downloading {REPO_ID} :: {KAS_FILE} ...")
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=KAS_FILE,
        repo_type="dataset",
        local_dir=out_dir,
    )
    print(f"Downloaded to {local_path}")


if __name__ == "__main__":
    main()
