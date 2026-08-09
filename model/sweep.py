import argparse
import itertools
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "model" / "config_lite.yaml"

GRID = {
    "lora_r": [4, 8, 16],
    "learning_rate": [3e-5, 5e-5],
}


def run(cmd, timeout=None):
    print(">>>", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        print(proc.stdout[-4000:])
        print(proc.stderr[-4000:])
        return None
    return proc.stdout


def parse_geo(out):
    if not out:
        return None
    for line in out.splitlines():
        if "geo_mean" in line and "eval_geo_mean" in line:
            try:
                return float(line.split("eval_geo_mean")[1].split(":")[1].strip().rstrip(","))
            except Exception:
                continue
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--out", default=str(ROOT / "outputs" / "sweep_results.tsv"))
    parser.add_argument("--gpus", type=int, default=2)
    args = parser.parse_args()

    keys = ["lora_r", "learning_rate"]
    trials = [dict(zip(keys, combo)) for combo in itertools.product(GRID["lora_r"], GRID["learning_rate"])]

    base = yaml.safe_load(open(BASE, encoding="utf-8"))
    results = []
    sweep_dir = ROOT / "outputs" / "sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    for i, over in enumerate(trials):
        cfg = dict(base)
        cfg.update(over)
        cfg["output_dir"] = str(ROOT / f"outputs/models/tune-{i}")
        cfg["num_train_epochs"] = 1
        cfg["label_smoothing_factor"] = 0.1
        cfg["lr_scheduler_type"] = "cosine"
        cfg["eval_steps"] = 300
        cfg["save_steps"] = 300
        cfg["save_total_limit"] = 2
        cfg["warmup_steps"] = 60
        name = f"trial_{i}"
        cfg_path = sweep_dir / f"{name}.yaml"
        yaml.safe_dump(cfg, open(cfg_path, "w", encoding="utf-8"))

        cmd = [
            sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={args.gpus}",
            str(ROOT / "model" / "train.py"), "--config", str(cfg_path), "--max-steps", str(args.max_steps),
        ]
        out = run(cmd)
        geo = parse_geo(out)
        results.append((name, over["lora_r"], over["learning_rate"], geo))
        print(f"=== {name}: r={over['lora_r']} lr={over['learning_rate']:.0e} geo={geo} ===")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("trial\tlora_r\tlearning_rate\tgeo_mean\n")
        for name, r, lr, geo in results:
            f.write(f"{name}\t{r}\t{lr}\t{geo}\n")
    print(f"\nResults written to {out_path}")
    for name, r, lr, geo in sorted(results, key=lambda x: -(x[3] or -1)):
        print(f"  {name}: r={r} lr={lr:.0e} geo={geo}")


if __name__ == "__main__":
    main()
