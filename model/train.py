import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

_orig_torch_load = torch.load
def _torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load
from datasets import Dataset, load_from_disk
from metrics import compute_all
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
    model_kwargs = {"attn_implementation": config.get("attn_implementation")}

    if model_type in ("nllb", "mbart"):
        kwargs = {"src_lang": config.get("src_lang")} if model_type == "mbart" else {}
        tokenizer = AutoTokenizer.from_pretrained(model_name, tgt_lang=tgt_lang, **kwargs)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_kwargs)
        if tgt_lang:
            model.config.forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    elif model_type == "indic2":
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, trust_remote_code=True, **model_kwargs
        )
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
        model.print_trainable_parameters()
        if config.get("gradient_checkpointing", False):
            model.get_base_model().enable_input_require_grads()

    gen_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    gc = gen_model.generation_config
    gc.early_stopping = True
    gc.num_beams = config.get("num_beams", 8)
    gc.length_penalty = config.get("length_penalty", 0.6)
    gc.no_repeat_ngram_size = config.get("no_repeat_ngram_size", 3)
    gc.max_new_tokens = config.get("max_generate_tokens", 128)

    return tokenizer, model


def lora_targets(model, config):
    keywords = config.get("lora_target_modules") or []
    all_names = [n for n, _ in model.named_modules()]
    if keywords:
        found = [n for n in all_names if any(k in n for k in keywords)]
        if found:
            return found
    return [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]


def _target_token_ids(tokenizer, texts, max_len):
    spm = tokenizer.tgt_spm
    encoder = tokenizer.tgt_encoder
    unk = tokenizer.unk_token_id
    eos = encoder.get(tokenizer.eos_token, tokenizer.eos_token_id)
    ids = []
    for text in texts:
        pieces = spm.EncodeAsPieces(text)[: max_len - 1]
        ids.append([encoder.get(p, unk) for p in pieces] + [eos])
    return ids


def load_datasets(config, tokenizer):
    source_column = config.get("source_column", "english_text")
    target_column = config.get("target_column", "kashmiri_text")
    source_prefix = config.get("source_prefix", "") or ""
    max_src = config.get("max_source_length", 128)
    max_tgt = config.get("max_target_length", 128)
    model_type = config.get("model_type", "nllb")
    cache_dir = ROOT / config.get("cache_dir", "outputs/cache")

    processor = None
    if model_type == "indic2" and config.get("use_indic_processor", False):
        try:
            from IndicTransToolkit.processor import IndicProcessor

            processor = IndicProcessor(inference=False)
        except ImportError:
            processor = None

    src_lang = config.get("src_lang", "")
    tgt_lang = config.get("tgt_lang", "")

    def tokenize_df(df):
        sources_raw = df[source_column].astype(str).tolist()
        targets_raw = df[target_column].astype(str).tolist()

        if processor is not None:
            sources = processor.preprocess_batch(
                sources_raw, src_lang=src_lang, tgt_lang=tgt_lang
            )
            targets = processor.preprocess_batch(
                targets_raw, src_lang=src_lang, tgt_lang=tgt_lang, is_target=True
            )
        else:
            sources = [source_prefix + s for s in sources_raw]
            targets = targets_raw

        input_ids, attention, labels = [], [], []
        for i in range(0, len(sources), 1000):
            chunk_src = sources[i : i + 1000]
            chunk_tgt = targets[i : i + 1000]
            inputs = tokenizer(
                chunk_src, max_length=max_src, truncation=True, padding=False
            )
            if model_type == "indic2":
                chunk_labels = _target_token_ids(tokenizer, chunk_tgt, max_tgt)
            else:
                tgt = tokenizer(
                    text_target=chunk_tgt,
                    max_length=max_tgt,
                    truncation=True,
                    padding=False,
                )
                chunk_labels = tgt["input_ids"]
            input_ids.extend(inputs["input_ids"])
            attention.extend(inputs["attention_mask"])
            labels.extend(chunk_labels)
        return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}

    def build(csv_path, cache_path):
        df = pd.read_csv(ROOT / csv_path)
        sample = " ".join(df[target_column].astype(str).head(500))
        arabic = sum(1 for c in sample if "\u0600" <= c <= "\u06ff" or "\u0750" <= c <= "\u077f")
        total = sum(1 for c in sample if not c.isspace())
        ratio = arabic / total if total else 0.0
        if ratio < 0.2:
            raise SystemExit(
                f"Data check failed: kashmiri_text is only {ratio:.2f} Arabic script. "
                "Run data/train_val_split.py to fix the column order."
            )
        if cache_path.exists():
            print(f"Loading tokenized dataset from {cache_path} (kashmiri Arabic {ratio:.2f})")
            return load_from_disk(str(cache_path))
        print(f"Loaded {len(df):,} pairs from {csv_path} (kashmiri Arabic {ratio:.2f})")
        dataset = Dataset.from_dict(tokenize_df(df))
        dataset.save_to_disk(str(cache_path))
        return dataset

    train_dataset = build(config["train_file"], cache_dir / "train")
    val_dataset = build(config["val_file"], cache_dir / "val")
    val_max = config.get("eval_max_samples", 0)
    if val_max and len(val_dataset) > val_max:
        rng = random.Random(config.get("seed", 42))
        val_dataset = val_dataset.select(rng.sample(range(len(val_dataset)), val_max))
    return train_dataset, val_dataset


def compute_metrics(tokenizer):
    def _compute(eval_preds):
        pred_ids, label_ids = eval_preds
        label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
        predictions = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        references = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return compute_all(predictions, references)

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
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    tokenizer, model = load_tokenizer_and_model(config)
    train_dataset, val_dataset = load_datasets(config, tokenizer)

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
        max_steps=args.max_steps if args.max_steps is not None else -1,
        weight_decay=config.get("weight_decay", 0.01),
        warmup_steps=config.get("warmup_steps", 500),
        logging_steps=config.get("logging_steps", 100),
        logging_first_step=True,
        predict_with_generate=config.get("predict_with_generate", True),
        fp16=config.get("fp16", True) and torch.cuda.is_available(),
        gradient_checkpointing=config.get("gradient_checkpointing", False),
        gradient_checkpointing_kwargs=(
            {"use_reentrant": False} if config.get("gradient_checkpointing", False) else None
        ),
        ddp_find_unused_parameters=False,
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
        callbacks=_backup_callbacks(config, output_dir),
    )

    resume_from = args.resume or latest_checkpoint(output_dir)
    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    with open(output_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

    _push_to_hub(output_dir, config, final=True)


def _backup_callbacks(config, output_dir):
    repo_id = config.get("backup_hf_repo")
    if not repo_id:
        return []
    from transformers import TrainerCallback

    class _Backup(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            step = state.global_step
            try:
                _push_to_hub(output_dir / f"checkpoint-{step}", config)
            except Exception as e:
                print(f"[backup] step {step} upload failed: {e}")

    return [_Backup()]


def _push_to_hub(path, config, final=False):
    repo_id = config.get("backup_hf_repo")
    if not repo_id:
        return
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"[backup] create_repo skipped: {e}")
    kind = "final" if final else "step"
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(path),
        commit_message=f"backup {kind}",
        ignore_patterns=["*.pt", "rng_state.pth", "scheduler.pt", "optimizer.pt"],
    )
    print(f"[backup] uploaded {kind} to HF repo {repo_id}")


if __name__ == "__main__":
    main()
