"""Evaluate a fine-tuned model with chrF++ and BLEU (geometric mean).

Loads a checkpoint, translates the held-out split, and reports:

    chrF++     - character-level fluency
    BLEU       - n-gram precision
    geo_mean   - sqrt(chrF++ * BLEU)  <- the competition metric

The predictions are also saved to disk so they can be inspected.

Usage:
    python model/evaluate.py \
        --checkpoint outputs/models/nllb-baseline \
        --config model/config.yaml \
        --data data/val.csv
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml
from sacrebleu import CHRF, BLEU

from inference import (
    ROOT,
    generate_predictions,
    load_config,
    load_model_and_tokenizer,
)


def compute_metrics(predictions: list[str], references: list[str]) -> dict:
    """chrF++ and BLEU + geometric mean for corpus-level scoring."""
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length")

    refs = [[ref] for ref in references]
    chrf = CHRF(word_order=2)  # chrF++
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
    parser = argparse.ArgumentParser(description="Evaluate EN->KS model")
    parser.add_argument("--checkpoint", required=True, help="Fine-tuned model dir")
    parser.add_argument(
        "--config", default=str(ROOT / "model" / "config.yaml"), help="YAML config"
    )
    parser.add_argument(
        "--data", default="data/val.csv", help="CSV with english_text + kashmiri_text"
    )
    parser.add_argument(
        "--out", default="outputs/predictions/val_predictions.csv", help="Save predictions"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    source_column = config.get("source_column", "english_text")
    target_column = config.get("target_column", "kashmiri_text")

    df = pd.read_csv(ROOT / args.data)
    texts = df[source_column].astype(str).tolist()
    references = df[target_column].astype(str).tolist()
    print(f"Evaluating on {len(df):,} sentences ...")

    model, tokenizer = load_model_and_tokenizer(config, args.checkpoint)
    predictions = generate_predictions(model, tokenizer, texts, config)

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
