#!/usr/bin/env python3
"""build_dialogue.py -- generate Colophon's *dialogue* teaching corpus.

Why this exists
---------------
Incipit's three-act teaching sequence ends by DEMONSTRATING, not asserting,
why chat format works: Act 2 shows chat format failing on the YAML-trained
elements model; Act 3 shows the identical architecture answering in chat
format -- because THIS corpus renders the same 118 IUPAC element facts as
user:/model: dialogues. The only variable between the acts is the format
of the training data.

Design choices (mirroring build_elements.py, deliberately)
----------------------------------------------------------
* Undisputed facts only: number / symbol / name / period. Group is omitted
  (the f-block group-3 assignment is genuinely unsettled).
* No provenance comments inside the data files -- they would pollute the
  tiny character vocabulary. Provenance lives here and in the README.
* Checked in AND regenerable byte-for-byte from this script.
* Deterministic: no randomness; the corpus is a pure function of the table.
* One two-exchange dialogue per element ends with a follow-up that says
  "it" instead of the element's name -- answering it requires the earlier
  turn to still be inside the context window. That is the multiturn-context
  training signal Incipit's saliency panel makes visible.

Format
------
One .txt file per element: five single-exchange dialogues plus the
two-exchange follow-up, blank-line separated. Each exchange is
    user: <question>\nmodel: <answer>\n
Role markers are lowercase; element names keep their canonical
capitalization (they are facts).

Source / attribution: IUPAC Periodic Table of the Elements
(https://iupac.org/what-we-do/periodic-table-of-elements/), same as
build_elements.py.

Usage
-----
  python teaching_data/build_dialogue.py    # (re)write teaching_data/dialogue/*.txt
  python colophon.py --src teaching_data/dialogue --out dialogue_k64.npz \
      --steps 30000 --K 64 --E 64 --H 512 train
"""
from __future__ import annotations
import os

from build_elements import ELEMENTS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "dialogue")


def render_dialogues(el: dict) -> str:
    """All dialogues for one element: five single exchanges plus one
    two-exchange follow-up, blank-line separated."""
    n, sym, name, per = el["number"], el["symbol"], el["name"], el["period"]
    singles = [
        (f"what is element {n}?",
         f"element {n} is {name} ({sym}), in period {per}."),
        (f"what is the symbol for {name}?",
         f"the symbol for {name} is {sym}."),
        (f"which element has the symbol {sym}?",
         f"{sym} is {name}, element {n}."),
        (f"which period is {name} in?",
         f"{name} is in period {per}."),
        (f"tell me about {name}.",
         f"{name} has symbol {sym}, atomic number {n}, and sits in period {per}."),
    ]
    parts = [f"user: {q}\nmodel: {a}\n" for q, a in singles]
    parts.append(
        f"user: what is element {n}?\n"
        f"model: element {n} is {name} ({sym}), in period {per}.\n"
        f"user: which period is it in?\n"
        f"model: {name} is in period {per}.\n"
    )
    return "\n".join(parts)


def filename(el: dict) -> str:
    """Zero-padded so a lexical sort matches atomic-number order."""
    return f"{el['number']:03d}-{el['name'].lower()}.txt"


def write_all(out_dir: str = OUT_DIR) -> int:
    os.makedirs(out_dir, exist_ok=True)
    for el in ELEMENTS:
        with open(os.path.join(out_dir, filename(el)), "w", encoding="utf-8") as f:
            f.write(render_dialogues(el))
    return len(ELEMENTS)


if __name__ == "__main__":
    n = write_all()
    print(f"wrote {n} dialogue files to {OUT_DIR}")
