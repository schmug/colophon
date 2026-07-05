#!/usr/bin/env python3
"""build_kana.py -- generate Colophon's hiragana kana *teaching corpus*.

Why this exists
---------------
Both other corpora (the OSAI index and the periodic table) are Latin-only, so
the project's canonical out-of-distribution prompt -- 日本語で書いてください --
can only ever show characters *no* model in the suite has seen. This corpus
puts a model in the suite for which that same prompt is partly home turf: the
four kanji stay off-map while the six hiragana light up as known, character by
character, in Marginalia's heatmap. The lesson it adds is that "off-map" is a
fact about the model-data pairing, not about the text.

Design choices (deliberate)
---------------------------
* **71 kana: the 46 modern gojūon (incl. ん, を) plus the 25 voiced
  dakuten/handakuten kana (が..ぽ).** The voiced set is load-bearing: で and だ
  in the canonical prompt are voiced, and without them the demo would muddy
  into three categories instead of a clean kanji-vs-kana split. Archaic ゐ/ゑ,
  small kana (ゃゅょっ), katakana, and kanji are excluded -- undisputed modern
  chart facts only.
* **Two fields, no digits.** `kana:` + `romaji:` keeps digits and uppercase
  out of the vocabulary entirely, so the characters shared with the Latin
  corpora shrink to lowercase letters, space, colon, and newline -- and no
  fake overlap with the elements corpus's `number:` prompts.
* **No provenance comments inside the data files.** The corpus IS the training
  text; comments would pollute the tiny vocabulary. Provenance lives here and
  in the README, so each YAML file stays two clean lines.
* **Checked in AND regenerable.** The 71 files are committed so `--src` works
  offline, and this script reproduces them byte-for-byte.

Source / attribution
--------------------
The gojūon chart and kana readings are public-domain facts. Romanization
follows traditional Hepburn as printed on learners' charts: し shi, ち chi,
つ tsu, ふ fu, を wo; ぢ and づ collapse onto ji and zu (same as じ/ず) -- that
collision is the Hepburn fact, not an error. Filenames stay unique via their
numeric chart-order prefix.

Usage
-----
  python teaching_data/build_kana.py          # (re)write teaching_data/kana/*.yaml
  python colophon.py --src teaching_data/kana --out kana.npz --steps 3000 train
"""
from __future__ import annotations
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "kana")

# (kana, Hepburn romaji) in chart order: gojūon rows a..wa, を, ん, then the
# voiced rows が ざ だ ば and handakuten ぱ.
_TABLE = [
    ("あ", "a"), ("い", "i"), ("う", "u"), ("え", "e"), ("お", "o"),
    ("か", "ka"), ("き", "ki"), ("く", "ku"), ("け", "ke"), ("こ", "ko"),
    ("さ", "sa"), ("し", "shi"), ("す", "su"), ("せ", "se"), ("そ", "so"),
    ("た", "ta"), ("ち", "chi"), ("つ", "tsu"), ("て", "te"), ("と", "to"),
    ("な", "na"), ("に", "ni"), ("ぬ", "nu"), ("ね", "ne"), ("の", "no"),
    ("は", "ha"), ("ひ", "hi"), ("ふ", "fu"), ("へ", "he"), ("ほ", "ho"),
    ("ま", "ma"), ("み", "mi"), ("む", "mu"), ("め", "me"), ("も", "mo"),
    ("や", "ya"), ("ゆ", "yu"), ("よ", "yo"),
    ("ら", "ra"), ("り", "ri"), ("る", "ru"), ("れ", "re"), ("ろ", "ro"),
    ("わ", "wa"), ("を", "wo"), ("ん", "n"),
    ("が", "ga"), ("ぎ", "gi"), ("ぐ", "gu"), ("げ", "ge"), ("ご", "go"),
    ("ざ", "za"), ("じ", "ji"), ("ず", "zu"), ("ぜ", "ze"), ("ぞ", "zo"),
    ("だ", "da"), ("ぢ", "ji"), ("づ", "zu"), ("で", "de"), ("ど", "do"),
    ("ば", "ba"), ("び", "bi"), ("ぶ", "bu"), ("べ", "be"), ("ぼ", "bo"),
    ("ぱ", "pa"), ("ぴ", "pi"), ("ぷ", "pu"), ("ぺ", "pe"), ("ぽ", "po"),
]

KANA = [
    {"number": i, "kana": kana, "romaji": romaji}
    for i, (kana, romaji) in enumerate(_TABLE, start=1)
]


def render(k: dict) -> str:
    """One kana as its two-line YAML training document (trailing newline)."""
    return f"kana: {k['kana']}\nromaji: {k['romaji']}\n"


def filename(k: dict) -> str:
    """Zero-padded so a lexical sort matches chart order; the numeric prefix
    keeps Hepburn collisions (ji/zu twice) unique. Numbers appear only here,
    never in the training text."""
    return f"{k['number']:03d}-{k['romaji']}.yaml"


def write_all(out_dir: str = OUT_DIR) -> int:
    os.makedirs(out_dir, exist_ok=True)
    for k in KANA:
        with open(os.path.join(out_dir, filename(k)), "w", encoding="utf-8") as f:
            f.write(render(k))
    return len(KANA)


if __name__ == "__main__":
    n = write_all()
    print(f"wrote {n} kana files to {OUT_DIR}")
