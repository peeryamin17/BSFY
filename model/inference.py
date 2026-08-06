import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sacrebleu import CHRF, BLEU
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

try:
    from orthography import normalize
except ImportError:
    from model.orthography import normalize

ROOT = Path(__file__).resolve().parent.parent


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_base(config):
    model_type = config.get("model_type", "nllb")
    model_name = config["model_name"]
    tgt_lang = config.get("tgt_lang", "")
    model_kwargs = {"attn_implementation": config.get("attn_implementation")}

    if model_type in ("nllb", "mbart"):
        tokenizer = AutoTokenizer.from_pretrained(model_name, tgt_lang=tgt_lang)
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
    return model, tokenizer


def load_model_and_tokenizer(config, checkpoint=None):
    if checkpoint and (Path(checkpoint) / "adapter_config.json").exists():
        from peft import PeftModel

        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        base_model, _ = _load_base(config)
        model = PeftModel.from_pretrained(base_model, checkpoint)
        if config.get("merge_lora", True):
            model = model.merge_and_unload()
    elif checkpoint and (Path(checkpoint) / "config.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    else:
        model, tokenizer = _load_base(config)
    return model, tokenizer


def collect_checkpoints(checkpoint):
    path = Path(checkpoint)
    if not path.exists():
        return []
    subdirs = sorted(
        (p for p in path.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")),
        key=lambda p: int(p.name.rsplit("-", 1)[-1]),
    )
    if subdirs:
        return [str(p) for p in subdirs]
    if (path / "config.json").exists():
        return [str(path)]
    return []


def _setup_language(config, tokenizer, model):
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


def _prefix_source(texts, config):
    prefix = config.get("source_prefix", "") or ""
    return [prefix + t for t in texts] if prefix else texts


def _gen_kwargs(config, n_best=1):
    if n_best > 1:
        return {
            "max_new_tokens": config.get("max_generate_tokens", 128),
            "do_sample": True,
            "top_p": config.get("top_p", 0.9),
            "temperature": config.get("temperature", 0.7),
            "num_return_sequences": n_best,
            "num_beams": 1,
            "no_repeat_ngram_size": config.get("no_repeat_ngram_size", 3),
        }
    return {
        "max_new_tokens": config.get("max_generate_tokens", 128),
        "num_beams": config.get("num_beams", 8),
        "length_penalty": config.get("length_penalty", 0.6),
        "early_stopping": True,
        "no_repeat_ngram_size": config.get("no_repeat_ngram_size", 3),
    }


def _get_processor(config):
    if config.get("model_type", "nllb") == "indic2" and config.get("use_indic_processor", False):
        try:
            from IndicTransToolkit.processor import IndicProcessor

            return IndicProcessor(inference=True)
        except ImportError:
            return None
    return None


@torch.no_grad()
def generate_predictions(model, tokenizer, texts, config, batch_size=None):
    if batch_size is None:
        batch_size = config.get("batch_size", 32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forced_bos = _setup_language(config, tokenizer, model)
    processor = _get_processor(config)
    src_lang = config.get("src_lang", "")
    tgt_lang = config.get("tgt_lang", "")

    model.eval()
    all_preds = []
    gen_kwargs = _gen_kwargs(config)
    if forced_bos is not None:
        gen_kwargs["forced_bos_token_id"] = forced_bos

    for i in range(0, len(texts), batch_size):
        raw = texts[i : i + batch_size]
        if processor is not None:
            batch = processor.preprocess_batch(raw, src_lang=src_lang, tgt_lang=tgt_lang)
        else:
            batch = _prefix_source(raw, config)
        inputs = tokenizer(
            batch,
            max_length=config.get("max_source_length", 128),
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)
        outputs = model.generate(**inputs, **gen_kwargs)
        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        if processor is not None:
            preds = processor.postprocess_batch(preds, lang=tgt_lang)
        all_preds.extend(preds)

    return all_preds


@torch.no_grad()
def generate_candidates(model, tokenizer, texts, config, n_best=8, batch_size=None):
    if batch_size is None:
        batch_size = config.get("batch_size", 32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forced_bos = _setup_language(config, tokenizer, model)
    processor = _get_processor(config)
    src_lang = config.get("src_lang", "")
    tgt_lang = config.get("tgt_lang", "")

    model.eval()
    all_candidates = []
    gen_kwargs = _gen_kwargs(config, n_best=n_best)
    if forced_bos is not None:
        gen_kwargs["forced_bos_token_id"] = forced_bos

    for i in range(0, len(texts), batch_size):
        raw = texts[i : i + batch_size]
        if processor is not None:
            batch = processor.preprocess_batch(raw, src_lang=src_lang, tgt_lang=tgt_lang)
        else:
            batch = _prefix_source(raw, config)
        inputs = tokenizer(
            batch,
            max_length=config.get("max_source_length", 128),
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)
        outputs = model.generate(**inputs, **gen_kwargs)
        batch_len = outputs.shape[0] // n_best
        outputs = outputs.view(batch_len, n_best, -1)
        rows = [tokenizer.batch_decode(row, skip_special_tokens=True) for row in outputs]
        if processor is not None:
            flat = processor.postprocess_batch(
                [c for row in rows for c in row],
                lang=tgt_lang,
                num_return_sequences=n_best,
            )
            rows = [flat[j * n_best : (j + 1) * n_best] for j in range(batch_len)]
        all_candidates.extend(rows)

    return all_candidates


def mbr_rerank(candidates, metric="chrf"):
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) < 2:
        return candidates[0]

    if metric == "bleu":
        scorer = BLEU()
    else:
        scorer = CHRF(word_order=2)

    scores = []
    for hyp in candidates:
        refs = [r for r in candidates if r != hyp]
        utility = np.mean([scorer.sentence_score(hyp, [r]).score for r in refs])
        scores.append(utility)
    return candidates[int(np.argmax(scores))]


def predict(texts, config, checkpoints, batch_size=None):
    if not config.get("mbr", False):
        checkpoint = None
        if checkpoints:
            root = Path(checkpoints[0]).parent
            if (root / "adapter_config.json").exists():
                checkpoint = str(root)
            else:
                checkpoint = checkpoints[0]
        model, tokenizer = load_model_and_tokenizer(config, checkpoint)
        results = generate_predictions(model, tokenizer, texts, config, batch_size)
    else:
        checkpoints = checkpoints or [None]
        max_ckpts = config.get("mbr_max_checkpoints", 0)
        if max_ckpts and len(checkpoints) > max_ckpts:
            checkpoints = checkpoints[-max_ckpts:]

        n_best = config.get("mbr_n_best", 8)
        pooled = [[] for _ in texts]
        for ckpt in checkpoints:
            model, tokenizer = load_model_and_tokenizer(config, ckpt)
            candidates = generate_candidates(model, tokenizer, texts, config, n_best, batch_size)
            for i, cands in enumerate(candidates):
                pooled[i].extend(cands)
            del model, tokenizer
            torch.cuda.empty_cache()

        pooled = [[normalize(c) for c in cands] for cands in pooled]
        metric = config.get("mbr_metric", "chrf")
        results = [mbr_rerank(cands, metric) for cands in pooled]
    return [normalize(p) for p in results]


def translate(texts, checkpoint=None, config_path=None):
    config = load_config(config_path or str(ROOT / "model" / "config.yaml"))
    checkpoints = collect_checkpoints(checkpoint) if checkpoint else []
    return predict(texts, config, checkpoints)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=str(ROOT / "model" / "config.yaml"))
    parser.add_argument("--test", default="data/test.csv")
    parser.add_argument("--out", default="outputs/submission.csv")
    parser.add_argument("--mbr", dest="mbr", action="store_true")
    parser.add_argument("--no-mbr", dest="mbr", action="store_false")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.mbr is not None:
        config["mbr"] = args.mbr
    test_df = pd.read_csv(ROOT / args.test)
    source_column = config.get("source_column", "english_text")

    id_col = test_df.columns[0]
    texts = test_df[source_column].astype(str).tolist()
    print(f"Translating {len(texts):,} sentences ...")

    checkpoints = collect_checkpoints(args.checkpoint) if args.checkpoint else []
    if checkpoints:
        print(f"Using {len(checkpoints)} checkpoint(s): {[Path(c).name for c in checkpoints]}")

    predictions = predict(texts, config, checkpoints)

    out_df = pd.DataFrame({id_col: test_df[id_col], "kashmiri_text": predictions})
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df):,} predictions to {out_path}")


if __name__ == "__main__":
    main()
