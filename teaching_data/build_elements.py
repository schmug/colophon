#!/usr/bin/env python3
"""build_elements.py -- generate Colophon's periodic-table *teaching corpus*.

Why this exists
---------------
The flagship corpus (the OSAI openness index) is jargon a layperson can't grade
by eye, so they have to take the model's confidence signals on faith -- the very
thing Colophon exists to abolish. This corpus is the on-ramp: 118 elements, one
tiny YAML file each, whose ground truth already lives in the reader's head. A
person who knows Fe = Iron can watch the model be confidently right in
distribution, then confidently *wrong* on a made-up `number: 250`, and finally
trip the categorical off-map flag on a never-seen character -- validating the
machinery against facts they already hold before trusting it on the index they
can't check.

Design choices (deliberate)
---------------------------
* **Undisputed facts only.** Fields are `number / symbol / name / period`. Group
  is omitted on purpose: the group-3 assignment of the f-block is genuinely
  unsettled, and shipping a disputed fact would undercut the whole point ("facts
  you already know"). Period is defined and undisputed for every element.
* **No provenance comments inside the data files.** The corpus IS the training
  text; a repeated header would pollute the vocabulary with URL/punctuation
  characters and skew the distribution. Provenance lives here and in the README
  instead, so each YAML file stays four clean lines.
* **Checked in AND regenerable.** The 118 files are committed so `--src` works
  offline (like `sample_data/`), and this script reproduces them byte-for-byte
  so the data can never silently drift from its source.

Source / attribution
--------------------
Element facts are public-domain (not copyrightable). Symbols, names, and atomic
numbers follow the IUPAC Periodic Table of the Elements
(https://iupac.org/what-we-do/periodic-table-of-elements/); names use IUPAC
spellings (Aluminium, Sulfur, Caesium). Periods are the table rows.

Usage
-----
  python teaching_data/build_elements.py          # (re)write teaching_data/elements/*.yaml
  python colophon.py --src teaching_data/elements --out elements.npz train
"""
from __future__ import annotations
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "elements")

# (atomic number, symbol, name) for Z = 1..118, in order. IUPAC spellings.
_TABLE = [
    (1, "H", "Hydrogen"), (2, "He", "Helium"), (3, "Li", "Lithium"),
    (4, "Be", "Beryllium"), (5, "B", "Boron"), (6, "C", "Carbon"),
    (7, "N", "Nitrogen"), (8, "O", "Oxygen"), (9, "F", "Fluorine"),
    (10, "Ne", "Neon"), (11, "Na", "Sodium"), (12, "Mg", "Magnesium"),
    (13, "Al", "Aluminium"), (14, "Si", "Silicon"), (15, "P", "Phosphorus"),
    (16, "S", "Sulfur"), (17, "Cl", "Chlorine"), (18, "Ar", "Argon"),
    (19, "K", "Potassium"), (20, "Ca", "Calcium"), (21, "Sc", "Scandium"),
    (22, "Ti", "Titanium"), (23, "V", "Vanadium"), (24, "Cr", "Chromium"),
    (25, "Mn", "Manganese"), (26, "Fe", "Iron"), (27, "Co", "Cobalt"),
    (28, "Ni", "Nickel"), (29, "Cu", "Copper"), (30, "Zn", "Zinc"),
    (31, "Ga", "Gallium"), (32, "Ge", "Germanium"), (33, "As", "Arsenic"),
    (34, "Se", "Selenium"), (35, "Br", "Bromine"), (36, "Kr", "Krypton"),
    (37, "Rb", "Rubidium"), (38, "Sr", "Strontium"), (39, "Y", "Yttrium"),
    (40, "Zr", "Zirconium"), (41, "Nb", "Niobium"), (42, "Mo", "Molybdenum"),
    (43, "Tc", "Technetium"), (44, "Ru", "Ruthenium"), (45, "Rh", "Rhodium"),
    (46, "Pd", "Palladium"), (47, "Ag", "Silver"), (48, "Cd", "Cadmium"),
    (49, "In", "Indium"), (50, "Sn", "Tin"), (51, "Sb", "Antimony"),
    (52, "Te", "Tellurium"), (53, "I", "Iodine"), (54, "Xe", "Xenon"),
    (55, "Cs", "Caesium"), (56, "Ba", "Barium"), (57, "La", "Lanthanum"),
    (58, "Ce", "Cerium"), (59, "Pr", "Praseodymium"), (60, "Nd", "Neodymium"),
    (61, "Pm", "Promethium"), (62, "Sm", "Samarium"), (63, "Eu", "Europium"),
    (64, "Gd", "Gadolinium"), (65, "Tb", "Terbium"), (66, "Dy", "Dysprosium"),
    (67, "Ho", "Holmium"), (68, "Er", "Erbium"), (69, "Tm", "Thulium"),
    (70, "Yb", "Ytterbium"), (71, "Lu", "Lutetium"), (72, "Hf", "Hafnium"),
    (73, "Ta", "Tantalum"), (74, "W", "Tungsten"), (75, "Re", "Rhenium"),
    (76, "Os", "Osmium"), (77, "Ir", "Iridium"), (78, "Pt", "Platinum"),
    (79, "Au", "Gold"), (80, "Hg", "Mercury"), (81, "Tl", "Thallium"),
    (82, "Pb", "Lead"), (83, "Bi", "Bismuth"), (84, "Po", "Polonium"),
    (85, "At", "Astatine"), (86, "Rn", "Radon"), (87, "Fr", "Francium"),
    (88, "Ra", "Radium"), (89, "Ac", "Actinium"), (90, "Th", "Thorium"),
    (91, "Pa", "Protactinium"), (92, "U", "Uranium"), (93, "Np", "Neptunium"),
    (94, "Pu", "Plutonium"), (95, "Am", "Americium"), (96, "Cm", "Curium"),
    (97, "Bk", "Berkelium"), (98, "Cf", "Californium"), (99, "Es", "Einsteinium"),
    (100, "Fm", "Fermium"), (101, "Md", "Mendelevium"), (102, "No", "Nobelium"),
    (103, "Lr", "Lawrencium"), (104, "Rf", "Rutherfordium"), (105, "Db", "Dubnium"),
    (106, "Sg", "Seaborgium"), (107, "Bh", "Bohrium"), (108, "Hs", "Hassium"),
    (109, "Mt", "Meitnerium"), (110, "Ds", "Darmstadtium"), (111, "Rg", "Roentgenium"),
    (112, "Cn", "Copernicium"), (113, "Nh", "Nihonium"), (114, "Fl", "Flerovium"),
    (115, "Mc", "Moscovium"), (116, "Lv", "Livermorium"), (117, "Ts", "Tennessine"),
    (118, "Og", "Oganesson"),
]

# Period boundaries: the highest atomic number in each row of the table.
_PERIOD_MAX = [(2, 1), (10, 2), (18, 3), (36, 4), (54, 5), (86, 6), (118, 7)]


def _period(number: int) -> int:
    for hi, per in _PERIOD_MAX:
        if number <= hi:
            return per
    raise ValueError(f"no period for atomic number {number}")


ELEMENTS = [
    {"number": n, "symbol": sym, "name": name, "period": _period(n)}
    for n, sym, name in _TABLE
]


def render(el: dict) -> str:
    """One element as its four-line YAML training document (trailing newline)."""
    return (
        f"number: {el['number']}\n"
        f"symbol: {el['symbol']}\n"
        f"name: {el['name']}\n"
        f"period: {el['period']}\n"
    )


def filename(el: dict) -> str:
    """Zero-padded so a lexical sort matches atomic-number order."""
    return f"{el['number']:03d}-{el['name'].lower()}.yaml"


def write_all(out_dir: str = OUT_DIR) -> int:
    os.makedirs(out_dir, exist_ok=True)
    for el in ELEMENTS:
        with open(os.path.join(out_dir, filename(el)), "w", encoding="utf-8") as f:
            f.write(render(el))
    return len(ELEMENTS)


if __name__ == "__main__":
    n = write_all()
    print(f"wrote {n} element files to {OUT_DIR}")
