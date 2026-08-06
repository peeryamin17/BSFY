import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "ai4bharat/BPCC"
KAS_FILE = "bpcc-seed-latest/kas_Arab.tsv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw")
    parser.add_argument("--force", action="store_true")
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
