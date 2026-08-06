import argparse

import pandas as pd

from inference import (
    ROOT,
    collect_checkpoints,
    load_config,
    predict,
)
from metrics import compute_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=str(ROOT / "model" / "config.yaml"))
    parser.add_argument("--data", default="data/val.csv")
    parser.add_argument("--out", default="outputs/predictions/val_predictions.csv")
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--mbr", dest="mbr", action="store_true")
    parser.add_argument("--no-mbr", dest="mbr", action="store_false")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.mbr is not None:
        config["mbr"] = args.mbr
    source_column = config.get("source_column", "english_text")
    target_column = config.get("target_column", "kashmiri_text")

    df = pd.read_csv(ROOT / args.data)
    if args.max_samples and len(df) > args.max_samples:
        df = df.head(args.max_samples)
    texts = df[source_column].astype(str).tolist()
    references = df[target_column].astype(str).tolist()
    print(f"Evaluating on {len(df):,} sentences ...")

    checkpoints = collect_checkpoints(args.checkpoint)
    if checkpoints:
        print(f"Using {len(checkpoints)} checkpoint(s): {[c.rsplit('/', 1)[-1] for c in checkpoints]}")

    predictions = predict(texts, config, checkpoints)

    metrics = compute_all(predictions, references)
    print("\n========== EVALUATION ==========")
    for key, value in metrics.items():
        print(f"{key:<12}: {value}")
    print("================================")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = df.copy()
    out_df["predicted_kashmiri_text"] = predictions
    out_df.to_csv(out_path, index=False)
    print(f"Predictions saved to {out_path}")


if __name__ == "__main__":
    main()
