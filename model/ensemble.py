import argparse
from pathlib import Path

import pandas as pd

from inference import collect_checkpoints, load_config, predict

ROOT = Path(__file__).resolve().parent.parent
DEV_CSV = "/kaggle/input/competitions/kathe-2026/englishdev.csv"
OUT_CSV = "outputs/ensemble_submission.csv"

LITE_DIR = str(ROOT / "outputs/models/indic2-lite")
WIN_DIR = str(ROOT / "outputs/models/indic2-win")
HF_REPO = "peeryamin17/kathe-checkpoints"


def ensure_checkpoints(lite_dir, win_dir, config):
    """Collect checkpoints for both model families; warn if either is empty."""
    lite = collect_checkpoints(lite_dir)
    win = collect_checkpoints(win_dir)

    if not lite:
        print(f"[ensemble] WARNING: no lite checkpoints at {lite_dir}. "
              f"Downloading from HF {HF_REPO} ...")
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(
                HF_REPO,
                repo_type="model",
                local_dir=lite_dir,
                allow_patterns="checkpoint-*",
            )
            lite = collect_checkpoints(lite_dir)
        except Exception as e:
            print(f"[ensemble] lite download failed (non-fatal): {e}")

    if not win:
        print(f"[ensemble] WARNING: no win checkpoints at {win_dir}. "
              "The ensemble will be lite-only.")

    return lite, win


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "model" / "config_lite.yaml"))
    parser.add_argument("--num-beams", type=int, default=8)
    parser.add_argument("--length-penalty", type=float, default=None)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--out", default=str(ROOT / OUT_CSV))
    parser.add_argument("--lite-repeats", type=int, default=2, help="How many times to repeat lite (tie-break weight)")
    args = parser.parse_args()

    config = load_config(args.config)
    config["mbr"] = True
    config["mbr_n_best"] = args.num_beams
    config["mbr_max_checkpoints"] = 0  # keep our explicit list intact

    for key, val in (
        ("length_penalty", args.length_penalty),
        ("no_repeat_ngram_size", args.no_repeat_ngram_size),
        ("max_generate_tokens", args.max_new_tokens),
        ("batch_size", args.batch_size),
    ):
        if val is not None:
            config[key] = val

    df = pd.read_csv(DEV_CSV)
    texts = df["sentence"].astype(str).tolist()

    lite_ckpts, win_ckpts = ensure_checkpoints(LITE_DIR, WIN_DIR, config)
    lite_anchor = lite_ckpts[-1] if lite_ckpts else None
    win_anchor = win_ckpts[-1] if win_ckpts else None

    # Tie-break trick: repeat the stronger (lite) model so its style wins ties.
    checkpoints = []
    if lite_anchor:
        checkpoints.extend([lite_anchor] * max(1, args.lite_repeats))
    if win_anchor:
        checkpoints.append(win_anchor)

    if not checkpoints:
        raise SystemExit("No checkpoints available at all — cannot run ensemble.")

    print(f"[ensemble] {len(checkpoints)} checkpoint(s): "
          f"{[Path(c).name for c in checkpoints]}")
    print(f"[ensemble] Generating {config['mbr_n_best']} candidates per checkpoint, "
          f"MBR rerank metric={config.get('mbr_metric', 'geo')}")

    predictions = predict(texts, config, checkpoints, batch_size=config.get("batch_size"))

    out = pd.DataFrame({"ID": df["ID"], "kashmiri_text": predictions})
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"DONE: {len(out)} predictions saved to {out_path}")


if __name__ == "__main__":
    main()
