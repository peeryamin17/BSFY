import argparse

import pandas as pd
from sacrebleu import CHRF, BLEU

from inference import (
    ROOT,
    collect_checkpoints,
    load_config,
    predict,
)


def compute_metrics(predictions, references):
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length")

    refs = [[ref] for ref in references]
    chrf = CHRF(word_order=2)
    bleu = BLEU()

    chrf_score = chrf.corpus_score(predictions, refs).score
    bleu_score = bleu.corpus_score(predictions, refs).score
    geo_mean = (chrf_score * bleu_score) ** 0.5

    return {
        "chrF++": round(chrf_score, 2),
        "BLEU": round(bleu_score, 2),
        "geo_mean": round(geo_mean, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=str(ROOT / "model" / "config.yaml"))
    parser.add_argument("--data", default="data/val.csv")
    parser.add_argument("--out", default="outputs/predictions/val_predictions.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    source_column = config.get("source_column", "english_text")
    target_column = config.get("target_column", "kashmiri_text")

    df = pd.read_csv(ROOT / args.data)
    texts = df[source_column].astype(str).tolist()
    references = df[target_column].astype(str).tolist()
    print(f"Evaluating on {len(df):,} sentences ...")

    checkpoints = collect_checkpoints(args.checkpoint)
    if checkpoints:
        print(f"Using {len(checkpoints)} checkpoint(s): {[c.rsplit('/', 1)[-1] for c in checkpoints]}")

    predictions = predict(texts, config, checkpoints)

    metrics = compute_metrics(predictions, references)
    print(f"\nchrF++   : {metrics['chrF++']}")
    print(f"BLEU     : {metrics['BLEU']}")
    print(f"geo_mean : {metrics['geo_mean']}")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = df.copy()
    out_df["predicted_kashmiri_text"] = predictions
    out_df.to_csv(out_path, index=False)
    print(f"Predictions saved to {out_path}")


if __name__ == "__main__":
    main()
