import argparse
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

VARIANT_GROUPS = [
    ["ی", "ے", "ي", "ې", "ێ", "ؠ"],
    ["و", "ۄ", "ۆ", "ۏ"],
    ["ه", "ھ", "ۂ", "ة", "ہ"],
    ["ن", "ں"],
    ["ا", "آ", "أ", "إ", "ٲ", "ٲ"],
    ["ك", "ک", "ڪ"],
    ["ٗ", "ٚ", "ٰ"],
    ["ہ", "ھ"],
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--column", default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.column is None:
        text_col = "kashmiri_text" if "kashmiri_text" in df.columns else df.columns[-1]
    else:
        text_col = args.column

    text = unicodedata.normalize("NFC", "".join(df[text_col].astype(str).tolist()))
    counts = Counter(text)
    total = len(text)

    print(f"Analyzing {text_col} from {args.input}")
    print("=== Character frequency (top 40) ===")
    for ch, n in counts.most_common(40):
        if ch.isspace():
            name = "SPACE"
        else:
            name = unicodedata.name(ch, "UNKNOWN")
        print(f"  U+{ord(ch):04X} {ch!r} {name}: {n} ({100*n/total:.3f}%)")

    print("\n=== Variant usage (model's own inconsistency) ===")
    for group in VARIANT_GROUPS:
        present = {ch: counts[ch] for ch in group if counts[ch] > 0}
        if len(present) < 2:
            continue
        total_g = sum(present.values())
        for ch, n in sorted(present.items(), key=lambda kv: -kv[1]):
            print(f"  U+{ord(ch):04X} {ch!r}: {n} ({100*n/total_g:.1f}% of group)")


if __name__ == "__main__":
    main()
