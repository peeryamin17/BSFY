import json
import re
from functools import lru_cache
from pathlib import Path

_CONFIG_PATH = Path(__file__).resolve().parent / "orthography_config.json"


@lru_cache(maxsize=1)
def _load():
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"char_map": {}, "suffixes": []}


def normalize(text, config=None):
    if config is None:
        config = _load()
    char_map = config.get("char_map") or {}
    if char_map:
        text = "".join(char_map.get(ch, ch) for ch in text)
    for rule in config.get("suffixes") or []:
        suffix = rule.get("suffix", "")
        if not suffix:
            continue
        if rule.get("attach"):
            text = re.sub(r"\s+" + re.escape(suffix), suffix, text)
        else:
            text = re.sub(r"(?<!\s)" + re.escape(suffix), " " + suffix, text)
    return text
