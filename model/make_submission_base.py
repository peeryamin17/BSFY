import argparse

import pandas as pd

from inference import load_config, load_model_and_tokenizer, predict

DEV_CSV = "/kaggle/input/competitions/kathe-2026/englishdev.csv"
OUT_CSV = "outputs/submission_base.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-beams", type=int, default=None)
    parser.add_argument("--length-penalty", type=float, default=None)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--out", default=OUT_CSV)
    parser.add_argument("--config", default="model/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.num_beams is not None:
        config["num_beams"] = args.num_beams
    if args.length_penalty is not None:
        config["length_penalty"] = args.length_penalty
    if args.no_repeat_ngram_size is not None:
        config["no_repeat_ngram_size"] = args.no_repeat_ngram_size
    if args.max_new_tokens is not None:
        config["max_generate_tokens"] = args.max_new_tokens
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size

    df = pd.read_csv(DEV_CSV)
    texts = df["sentence"].astype(str).tolist()
    model, tokenizer = load_model_and_tokenizer(config, None)
    predictions = predict(texts, config, [], batch_size=config.get("batch_size"))
    out = pd.DataFrame({"ID": df["ID"], "kashmiri_text": predictions})
    out.to_csv(args.out, index=False)
    print(f"DONE: {len(out)} predictions saved to {args.out}")


if __name__ == "__main__":
    main()
