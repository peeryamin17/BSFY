import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset, load_from_disk
from sacrebleu import CHRF, BLEU
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

ROOT = Path(__file__).resolve().parent.parent


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tokenizer_and_model(config):
    model_type = config.get("model_type", "nllb")
    model_name = config["model_name"]
    tgt_lang = config.get("tgt_lang", "")

    if model_type in ("nllb", "mbart"):
        kwargs = {"src_lang": config.get("src_lang")} if model_type == "mbart" else {}
        tokenizer = AutoTokenizer.from_pretrained(model_name, tgt_lang=tgt_lang, **kwargs)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if tgt_lang:
            model.config.forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    elif model_type == "indic2":
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.config.use_cache = not config.get("gradient_checkpointing", False)

    if config.get("use_lora", False):
        from peft import LoraConfig, TaskType, get_peft_model

        targets = lora_targets(model, config)
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=config.get("lora_r", 16),
            lora_alpha=config.get("lora_alpha", 32),
            lora_dropout=config.get("lora_dropout", 0.05),
            target_modules=targets,
        )
        model = get_peft_model(model, lora_config)

    return tokenizer, model


def lora_targets(model, config):
    keywords = config.get("lora_target_modules") or []
    all_names = [n for n, _ in model.named_modules()]
    if keywords:
        found = [n for n in all_names if any(k in n for k in keywords)]
        if found:
            return found
    return [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]


def load_datasets(config, tokenizer):
    source_column = config.get("source_column", "english_text")
    target_column = config.get("target_column", "kashmiri_text")
    source_prefix = config.get("source_prefix", "") or ""
    max_src = config.get("max_source_length", 128)
    max_tgt = config.get("max_target_length", 128)
    model_type = config.get("model_type", "nllb")
    cache_dir = ROOT / config.get("cache_dir", "outputs/cache")
    num_proc = config.get("num_proc", 4)

    processor = None
    if model_type == "indic2" and config.get("use_indic_processor", False):
        try:
            from IndicTransToolkit.processor import IndicProcessor

            processor = IndicProcessor(inference=False)
        except ImportError:
            processor = None

    src_lang = config.get("src_lang", "")
    tgt_lang = config.get("tgt_lang", "")

    if model_type == "indic2":
        num_proc = 1

    def tokenize(examples):
        if processor is not None:
            sources = processor.preprocess_batch(
                examples[source_column], src_lang=src_lang, tgt_lang=tgt_lang
            )
            raw_targets = processor.preprocess_batch(
                examples[target_column],
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                is_target=True,
            )
        else:
            sources = [source_prefix + s for s in examples[source_column]]
            raw_targets = examples[target_column]

        inputs = tokenizer(
            sources, max_length=max_src, truncation=True, padding=False
        )
        if model_type == "indic2":
            tokenizer._switch_to_target_mode()
            targets = tokenizer(
                raw_targets, max_length=max_tgt, truncation=True, padding=False
            )
            tokenizer._switch_to_input_mode()
        else:
            targets = tokenizer(
                text_target=raw_targets,
                max_length=max_tgt,
                truncation=True,
                padding=False,
            )
        inputs["labels"] = targets["input_ids"]
        return inputs

    def build(csv_path, cache_path):
        if cache_path.exists():
            print(f"Loading tokenized dataset from {cache_path}")
            return load_from_disk(str(cache_path))
        df = pd.read_csv(ROOT / csv_path)
        print(f"Loaded {len(df):,} pairs from {csv_path}")
        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(
            tokenize,
            batched=True,
            num_proc=num_proc,
            remove_columns=[source_column, target_column],
        )
        dataset.save_to_disk(str(cache_path))
        return dataset

    train_dataset = build(config["train_file"], cache_dir / "train")
    val_dataset = build(config["val_file"], cache_dir / "val")
    return train_dataset, val_dataset


def compute_metrics(tokenizer):
    def _compute(eval_preds):
        pred_ids, label_ids = eval_preds
        label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
        predictions = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        references = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        references = [[ref] for ref in references]

        chrf = CHRF(word_order=2)
        bleu = BLEU()
        chrf_score = chrf.corpus_score(predictions, references).score
        bleu_score = bleu.corpus_score(predictions, references).score
        geo_mean = (chrf_score * bleu_score) ** 0.5

        return {
            "chrF++": round(chrf_score, 2),
            "BLEU": round(bleu_score, 2),
            "geo_mean": round(geo_mean, 2),
        }

    return _compute


def latest_checkpoint(output_dir):
    if not output_dir.exists():
        return None
    steps = [
        int(d.name.rsplit("-", 1)[-1])
        for d in output_dir.iterdir()
        if d.is_dir() and d.name.startswith("checkpoint-")
    ]
    if not steps:
        return None
    return str(output_dir / f"checkpoint-{max(steps)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "model" / "config.yaml"))
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    tokenizer, model = load_tokenizer_and_model(config)
    train_dataset, val_dataset = load_datasets(config, tokenizer)

    if config.get("model_type", "nllb") == "indic2":
        try:
            from IndicTransToolkit.collator import IndicDataCollator

            data_collator = IndicDataCollator(tokenizer=tokenizer, model=model, padding=True)
        except ImportError:
            data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
    else:
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
        logging_first_step=True,
        predict_with_generate=config.get("predict_with_generate", True),
        fp16=config.get("fp16", True) and torch.cuda.is_available(),
        gradient_checkpointing=config.get("gradient_checkpointing", False),
        optim=config.get("optim", "adamw_torch"),
        dataloader_num_workers=config.get("dataloader_num_workers", 2),
        torch_compile=config.get("torch_compile", False),
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
        compute_metrics=compute_metrics(tokenizer),
    )

    resume_from = args.resume or latest_checkpoint(output_dir)
    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    with open(output_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)


if __name__ == "__main__":
    main()
