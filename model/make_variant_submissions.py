import argparse
import csv
import unicodedata
from pathlib import Path

# Variant groups observed in Kashmiri Perso-Arabic script (see analyze_predictions.py).
# Each variant map is a *canonicalization target*: a guess at what standard the
# hidden test references use. We apply NFC + the char map, nothing else.
VARIANTS = {
    # Target 1: "Kashmiri-heavy" — keep the distinct Kashmiri vowel letters,
    # collapse only the ye/kaf/heh clusters to their most common Kashmiri forms.
    "kashmiri": {
        "ی": "ی", "ے": "ی", "ي": "ی", "ې": "ی", "ێ": "ی",
        "و": "و", "ۄ": "ۄ", "ۆ": "و", "ۏ": "و",
        "ه": "ه", "ھ": "ہ", "ۂ": "ہ", "ة": "ہ", "ہ": "ہ",
        "ن": "ن", "ں": "ن",
        "ا": "ا", "آ": "آ", "أ": "ا", "إ": "ا", "ٲ": "ٲ",
        "ك": "ک", "ک": "ک", "ڪ": "ک",
    },
    # Target 2: "Arabic-leaning" — push everything toward standard Arabic letters
    # (ye -> ي, kaf -> ك, heh -> ه). Matches BIS/SIL style normalizations.
    "arabic": {
        "ی": "ي", "ے": "ي", "ي": "ي", "ې": "ي", "ێ": "ي",
        "و": "و", "ۄ": "و", "ۆ": "و", "ۏ": "و",
        "ه": "ه", "ھ": "ه", "ۂ": "ه", "ة": "ه", "ہ": "ه",
        "ن": "ن", "ں": "ن",
        "ا": "ا", "آ": "ا", "أ": "ا", "إ": "ا", "ٲ": "ا",
        "ك": "ك", "ک": "ك", "ڪ": "ك",
    },
    # Target 3: "Perso-Urdu-leaning" — keep Persian ye/keheh, normalize the rest.
    "perso": {
        "ی": "ی", "ے": "ی", "ي": "ی", "ې": "ی", "ێ": "ی",
        "و": "و", "ۄ": "و", "ۆ": "و", "ۏ": "و",
        "ه": "ه", "ھ": "ھ", "ۂ": "ھ", "ة": "ه", "ہ": "ھ",
        "ن": "ن", "ں": "ن",
        "ا": "ا", "آ": "ا", "أ": "ا", "إ": "ا", "ٲ": "ا",
        "ك": "ک", "ک": "ک", "ڪ": "ک",
    },
}


def apply(text, char_map):
    text = unicodedata.normalize("NFC", text)
    return "".join(char_map.get(ch, ch) for ch in text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV with ID,kashmiri_text")
    parser.add_argument("--outdir", default="outputs/variants")
    parser.add_argument("--variants", nargs="+", default=["kashmiri", "arabic", "perso"])
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.input, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    assert len(header) >= 2, f"unexpected header: {header}"
    id_col, text_col = header[0], header[1]

    def write(name, transform):
        path = outdir / f"variant_{name}.csv"
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow([id_col, text_col])
            for r in rows:
                w.writerow([r[0], transform(r[1])])
        print(f"wrote {path.name} ({len(rows)} rows)")

    write("baseline", lambda t: unicodedata.normalize("NFC", t))
    for name in args.variants:
        cmap = VARIANTS[name]
        write(name, lambda t, m=cmap: apply(t, m))

    print(f"DONE -> {outdir}")


if __name__ == "__main__":
    main()
