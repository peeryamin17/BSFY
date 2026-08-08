import pandas as pd

from inference import load_config, load_model_and_tokenizer, predict

DEV_CSV = "/kaggle/input/competitions/kathe-2026/englishdev.csv"
OUT_CSV = "outputs/submission_base.csv"


def main():
    config = load_config("model/config.yaml")
    df = pd.read_csv(DEV_CSV)
    texts = df["sentence"].astype(str).tolist()
    model, tokenizer = load_model_and_tokenizer(config, None)
    predictions = predict(texts, config, [])
    out = pd.DataFrame({"ID": df["ID"], "kashmiri_text": predictions})
    out.to_csv(OUT_CSV, index=False)
    print(f"DONE: {len(out)} predictions saved to {OUT_CSV}")


if __name__ == "__main__":
    main()
