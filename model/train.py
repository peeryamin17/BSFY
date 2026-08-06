"""Fine-tune a pretrained seq2seq model for English -> Kashmiri translation.

The best checkpoint (lowest eval loss) is saved to config['output_dir']
together with the tokenizer, so it can be used directly by evaluate.py and
inference.py.

Usage:
    python model/train.py --config model/config.yaml
"""
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset
from sacrebleu import CHRF, BLEU
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tokenizer_and_model(config: dict):
    """Load the tokenizer and model for the configured model_type."""
    model_type = config.get("model_type", "nllb")
    model_name = config["model_name"]
    tgt_lang = config.get("tgt_lang", "")

    if model_type in ("nllb", "mbart"):
        kwargs = {"src_lang": config.get("src_lang")} if model_type == "mbart" else {}
        tokenizer = AutoTokenizer.from_pretrained(model_name, tgt_lang=tgt_lang, **kwargs)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if tgt_lang:
            # Force decoding to start with the target language id.
            model.config.forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    elif model_type == "indic2":
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return tokenizer, model


def load_datasets(config: dict):
    train_df = pd.read_csv(ROOT / config["train_file"])
    val_df = pd.read_csv(ROOT / config["val_file"])
    print(f"Train pairs: {len(train_df):,}  Val pairs: {len(val_df):,}")
    return Dataset.from_pandas(train_df), Dataset.from_pandas(val_df)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune EN -> KS translation model")
    parser.add_argument(
        "--config",
        default=str(ROOT / "model" / "config.yaml"),
        help="Path to the YAML config file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    tokenizer, model = load_tokenizer_and_model(config)
    train_dataset, val_dataset = load_datasets(config)

    source_column = config.get("source_column", "english_text")
    target_column = config.get("target_column", "kashmiri_text")
    source_prefix = config.get("source_prefix", "") or ""

    max_src = config.get("max_source_length", 128)
    max_tgt = config.get("max_target_length", 128)
    truncation = config.get("truncation", True)
    padding = config.get("padding", "max_length")

    def preprocess(examples):
        sources = [source_prefix + s for s in examples[source_column]]
        inputs = tokenizer(
            sources,
            max_length=max_src,
            truncation=truncation,
            padding=padding,
        )
        targets = tokenizer(
            text_target=examples[target_column],
            max_length=max_tgt,
            truncation=truncation,
            padding=padding,
        )
        inputs["labels"] = targets["input_ids"]
        return inputs

    cols_to_remove = [source_column, target_column]
    train_dataset = train_dataset.map(
        preprocess, batched=True, remove_columns=cols_to_remove
    )
    val_dataset = val_dataset.map(
        preprocess, batched=True, remove_columns=cols_to_remove
    )

    def compute_metrics(eval_preds):
        """chrF++ and BLEU + geometric mean on the validation set."""
        pred_ids, label_ids = eval_preds
        label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
        predictions = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        references = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        references = [[ref] for ref in references]

        chrf = CHRF(word_order=2)  # chrF++ (character n-grams up to 6)
        bleu = BLEU()

        chrf_score = chrf.corpus_score(predictions, references).score
        bleu_score = bleu.corpus_score(predictions, references).score
        geo_mean = (chrf_score * bleu_score) ** 0.5

        return {
            "chrF++": round(chrf_score, 2),
            "BLEU": round(bleu_score, 2),
            "geo_mean": round(geo_mean, 2),
        }

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

    output_dir = ROOT / config.get("output_dir", "outputs/models/kathe-model")
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        evaluation_strategy="steps",
        eval_steps=config.get("eval_steps", 500),
        save_steps=config.get("save_steps", 500),
        save_total_limit=config.get("save_total_limit", 3),
        learning_rate=config.get("learning_rate", 3e-5),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 8),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 8),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        num_train_epochs=config.get("num_train_epochs", 3),
        weight_decay=config.get("weight_decay", 0.01),
        warmup_steps=config.get("warmup_steps", 500),
        logging_steps=config.get("logging_steps", 100),
        predict_with_generate=config.get("predict_with_generate", True),
        fp16=config.get("fp16", True) and torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="eval_geo_mean",
        greater_is_better=True,
        save_strategy="steps",
        report_to=["tensorboard"],
        seed=config.get("seed", 42),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print(f"Saving best model to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Dump the config alongside the weights for reproducibility.
    with open(output_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)


if __name__ == "__main__":
    main()
