import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent


def load_parallel_tsv(path):
    df = pd.read_csv(path, sep="\t", encoding="utf-8")
    columns = df.columns.tolist()

    if "src" in columns and "tgt" in columns:
        df = df[["src", "tgt"]].rename(
            columns={"src": "english_text", "tgt": "kashmiri_text"}
        )
    else:
        df = df.iloc[:, [1, 0]].copy()
        df.columns = ["english_text", "kashmiri_text"]

    df["english_text"] = df["english_text"].astype(str).str.strip()
    df["kashmiri_text"] = df["kashmiri_text"].astype(str).str.strip()
    df = df[df["english_text"].ne("") & df["kashmiri_text"].ne("")]
    df.drop_duplicates(subset=["english_text"], keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/kas_Arab.tsv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.10)
    parser.add_argument("--test-size", type=float, default=0.10)
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

    train, test = train_test_split(df, test_size=args.test_size, random_state=args.seed)
    val_size_rel = args.val_size / (1 - args.test_size)
    train, val = train_test_split(train, test_size=val_size_rel, random_state=args.seed)

    out_dir = ROOT / "data"
    out_dir.mkdir(exist_ok=True)
    for name, split_df in {"train": train, "val": val, "test": test}.items():
        path = out_dir / f"{name}.csv"
        split_df.to_csv(path, index=False)
        print(f"{name}: {len(split_df):,} rows -> {path}")


if __name__ == "__main__":
    main()
