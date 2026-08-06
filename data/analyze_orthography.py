import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

VARIANT_GROUPS = [
    ["ی", "ے", "ي", "ې", "ێ"],
    ["و", "ۄ", "ۆ", "ۏ"],
    ["ه", "ھ", "ۂ", "ة"],
    ["ن", "ں"],
    ["ا", "آ", "أ", "إ"],
]

DOMINANCE_THRESHOLD = 0.90


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/train.csv")
    parser.add_argument("--column", default="kashmiri_text")
    parser.add_argument("--out", default="model/orthography_config.json")
    args = parser.parse_args()

    df = pd.read_csv(ROOT / args.input)
    text = "".join(df[args.column].astype(str).tolist())
    counts = Counter(text)
    total = len(text)

    print("=== Character frequency (top 50) ===")
    for ch, n in counts.most_common(50):
        print(f"  U+{ord(ch):04X} {ch!r}: {n} ({100 * n / total:.3f}%)")

    char_map = {}
    print("\n=== Variant groups ===")
    for group in VARIANT_GROUPS:
        present = {ch: counts[ch] for ch in group if counts[ch] > 0}
        if not present:
            continue
        main_char = max(present, key=present.get)
        group_total = sum(present.values())
        for ch, n in present.items():
            if ch == main_char:
                continue
            if present[main_char] / group_total >= DOMINANCE_THRESHOLD:
                char_map[ch] = main_char
                print(
                    f"  map {ch!r} (U+{ord(ch):04X}) -> {main_char!r} (U+{ord(main_char):04X})"
                    f"  [{n} vs {present[main_char]}]"
                )
            else:
                print(
                    f"  keep {ch!r} (U+{ord(ch):04X}) — co-existing with {main_char!r}"
                    f"  [{n} vs {present[main_char]}]"
                )

    out_path = ROOT / args.out
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = {"char_map": {}, "suffixes": []}
    existing["char_map"].update(char_map)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {out_path}")
    print("Add space/suffix rules under 'suffixes' if needed, e.g. {\"suffix\": \"ہند\", \"attach\": true}")


if __name__ == "__main__":
    main()
