"""Split the BPCC Kashmiri parallel corpus into train / val / test sets.

Reads the raw TSV downloaded by data/download_data.py and writes three CSVs:

    data/train.csv   (80%)
    data/val.csv     (10%)
    data/test.csv    (10%)

Each CSV has the columns:
    english_text     - English source sentence
    kashmiri_text    - Kashmiri translation

Usage:
    python data/train_val_split.py [--input data/raw/kas_Arab.tsv] [--seed 42]
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent

# Language code header line in the raw TSV, e.g. "en\tkas_Arab"
HEADER_FIELDS = {"en", "kas_arab"}


def load_parallel_tsv(path: Path) -> pd.DataFrame:
    """Parse the BPCC TSV into a DataFrame of (english, kashmiri) pairs.

    Skips the header line (if present) and drops empty / unpaired rows.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            en, ks = parts[0].strip(), parts[1].strip()
            # Skip a header line such as "en\tkas_Arab"
            if line_no == 0 and (en.lower() in HEADER_FIELDS or ks.lower() in HEADER_FIELDS):
                continue
            if en and ks:
                rows.append((en, ks))

    df = pd.DataFrame(rows, columns=["english_text", "kashmiri_text"])
    df.drop_duplicates(subset=["english_text"], keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Split BPCC corpus into train/val/test")
    parser.add_argument("--input", default="data/raw/kas_Arab.tsv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.10, help="Validation fraction")
    parser.add_argument("--test-size", type=float, default=0.10, help="Test fraction")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.is_absolute():
        in_path = ROOT / in_path
    if not in_path.exists():
        raise FileNotFoundError(
            f"Corpus not found at {in_path}. Run 'python data/download_data.py' first."
        )

    df = load_parallel_tsv(in_path)
    print(f"Loaded {len(df):,} parallel sentence pairs from {in_path}")

    # First hold out the test set, then split the remainder into train/val
    train, test = train_test_split(
        df, test_size=args.test_size, random_state=args.seed
    )
    val_size_rel = args.val_size / (1 - args.test_size)
    train, val = train_test_split(
        train, test_size=val_size_rel, random_state=args.seed
    )

    out_dir = ROOT / "data"
    out_dir.mkdir(exist_ok=True)
    splits = {"train": train, "val": val, "test": test}
    for name, split_df in splits.items():
        path = out_dir / f"{name}.csv"
        split_df.to_csv(path, index=False)
        print(f"{name}: {len(split_df):,} rows -> {path}")


if __name__ == "__main__":
    main()
