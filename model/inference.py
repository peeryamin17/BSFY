"""Generate English -> Kashmiri translations with a fine-tuned model.

Shared helpers used by both model/inference.py and model/evaluate.py.

Usage (make a Kaggle submission from test.csv):
    python model/inference.py \
        --checkpoint outputs/models/nllb-baseline \
        --config model/config.yaml \
        --test data/test.csv \
        --out outputs/submission.csv
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model_and_tokenizer(config: dict, checkpoint: str | None = None):
    """Load a fine-tuned checkpoint, falling back to the pretrained model."""
    if checkpoint and (Path(checkpoint) / "config.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    else:
        model_name = config["model_name"]
        model_type = config.get("model_type", "nllb")
        tgt_lang = config.get("tgt_lang", "")
        if model_type in ("nllb", "mbart"):
            tokenizer = AutoTokenizer.from_pretrained(model_name, tgt_lang=tgt_lang)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        elif model_type == "indic2":
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    return model, tokenizer


def _setup_language(config: dict, tokenizer, model) -> int | None:
    """Configure source/target languages for generation.

    Returns the forced_bos_token_id (target language id) for NLLB/mBART, or
    None for IndicTrans2 (which uses the source prefix tag instead).
    """
    model_type = config.get("model_type", "nllb")
    src_lang = config.get("src_lang", "")
    tgt_lang = config.get("tgt_lang", "")

    if model_type in ("nllb", "mbart"):
        if src_lang:
            tokenizer.src_lang = src_lang
        if tgt_lang:
            tokenizer.tgt_lang = tgt_lang
            return tokenizer.convert_tokens_to_ids(tgt_lang)
    return None


def _prefix_source(texts: list[str], config: dict) -> list[str]:
    prefix = config.get("source_prefix", "") or ""
    return [prefix + t for t in texts] if prefix else texts


@torch.no_grad()
def generate_predictions(
    model, tokenizer, texts: list[str], config: dict, batch_size: int = 32
) -> list[str]:
    """Translate a list of English sentences to Kashmiri."""
    device = model.device
    forced_bos = _setup_language(config, tokenizer, model)
    sources = _prefix_source(texts, config)

    model.eval()
    all_preds = []
    gen_kwargs = {
        "max_new_tokens": config.get("max_generate_tokens", 128),
        "num_beams": config.get("num_beams", 4),
        "length_penalty": config.get("length_penalty", 0.6),
        "early_stopping": True,
    }
    if forced_bos is not None:
        gen_kwargs["forced_bos_token_id"] = forced_bos

    for i in range(0, len(sources), batch_size):
        batch = sources[i : i + batch_size]
        inputs = tokenizer(
            batch,
            max_length=config.get("max_source_length", 128),
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)
        outputs = model.generate(**inputs, **gen_kwargs)
        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_preds.extend(preds)

    return all_preds


def translate(
    texts: list[str],
    checkpoint: str | None = None,
    config_path: str | None = None,
) -> list[str]:
    """Convenience wrapper for notebooks / scripts."""
    config = load_config(config_path or str(ROOT / "model" / "config.yaml"))
    model, tokenizer = load_model_and_tokenizer(config, checkpoint)
    return generate_predictions(model, tokenizer, texts, config)


def main():
    parser = argparse.ArgumentParser(description="Generate EN->KS predictions")
    parser.add_argument("--checkpoint", default=None, help="Fine-tuned model dir")
    parser.add_argument(
        "--config", default=str(ROOT / "model" / "config.yaml"), help="YAML config"
    )
    parser.add_argument("--test", default="data/test.csv", help="Test CSV with English text")
    parser.add_argument("--out", default="outputs/submission.csv", help="Output CSV")
    args = parser.parse_args()

    config = load_config(args.config)
    test_df = pd.read_csv(ROOT / args.test)
    source_column = config.get("source_column", "english_text")

    id_col = test_df.columns[0]
    texts = test_df[source_column].astype(str).tolist()
    print(f"Translating {len(texts):,} sentences ...")

    model, tokenizer = load_model_and_tokenizer(config, args.checkpoint)
    predictions = generate_predictions(model, tokenizer, texts, config)

    out_df = pd.DataFrame({id_col: test_df[id_col], "kashmiri_text": predictions})
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df):,} predictions to {out_path}")


if __name__ == "__main__":
    main()
