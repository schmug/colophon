# ❧ Colophon

**A model that ships its own colophon.**

A colophon is the note at the end of a book stating how it was made — press,
typeface, paper, print run. It's the original transparency statement, centuries
before "model cards." `Colophon` is a ~45K-parameter character language model
trained from zero in pure NumPy on the **European Open Source AI Index** (the
CC-BY database that scores AI systems across 14 dimensions of openness). It learns
to describe model openness, then gets scored against those same 14 dimensions —
and emits a `colophon.json` documenting its own data, training, and grades.

This is a demonstrator, not a chatbot. Its value is entirely structural.

## The argument, in three moves

**Concept.** A model only knows its training distribution. Usually that
distribution is unknowable to you. Here you can *read the entire corpus*. Nothing
else exists to the model.

**Problem.** Outside its distribution a model is confidently wrong, and its own
confidence signals don't reliably warn you. The demo shows it: on a Japanese
prompt the model reports *lower* next-char entropy than on its own native text —
it "feels" certain — while being categorically off-map (every character unseen).
Entropy is fooled; only the separate off-map signal catches it. With mainstream
closed models you get neither the corpus to check against nor, often, the raw
signals to compute — which is why the OSAI Index exists: most systems score red on
data transparency.

**Solution.** Full openness across all 14 dimensions turns every trust question
from an estimate into a fact you can audit. Colophon is 10/12 open (honestly red
on "preprint" and "paper" — no free greens); a typical closed API model is 0/12.
The contrast *is* the pitch.

## Ground-truth end of the same spectrum

Colophon is the companion to `competence_gate.py` (included). The gate is the
*estimate* end — domain-cluster distance plus a cutoff/temporal parser as a
pre-generation gate for models you can't see inside. Colophon is the
*ground-truth* end:

| Signal from the gate work | In Colophon it is… |
|---|---|
| Training-data composition | Known exactly — you can read the corpus |
| Knowledge cutoff | Absolute — the corpus *is* the model's universe |
| Domain competence map | The literal edge of the corpus |
| Token-level confidence | Fully white-box (we own every logit) |
| Off-map detection | Unknown-character flag (categorical, un-foolable) |

## Run it

```bash
pip install -r requirements.txt          # just numpy
python colophon.py demo                   # train + generate + confidence + scorecard
python colophon.py prepare                # write colophon.json (data section) + print scope
python colophon.py --steps 8000 train     # train and save colophon.npz + colophon.json
python colophon.py generate --prompt "availability_weights_"
```

Global flags (`--src`, `--steps`, `--seed`) go **before** the subcommand:
`python colophon.py --src ./osai --steps 8000 demo`.

Alternatively, install it as a package and use the `colophon` console command:

```bash
pip install -e .
colophon demo
colophon generate --prompt "availability_weights_"
```

`python colophon.py ...` keeps working unchanged either way.

Bundled `sample_data/` (three OSAI-style entries) lets it run with zero external
data. For the real thing, clone the index and point at it:

```bash
git clone https://codeberg.org/AI-Technology-Assessment/main-database osai
python colophon.py --src ./osai demo
```

Outputs (gitignored, regenerable): `colophon.npz` (weights) and `colophon.json`
— the model's own colophon: data section (datasheet), training section (model
card), and its openness scorecard, in one self-describing file.

## Honest limits — read before showing anyone

- **It is not intelligent.** A char-level MLP models local character statistics.
  It produces corpus-*shaped* text, not reasoning. Don't read coherence into it.
- **It overfits the tiny sample**, by design and on-theme — with a fully known
  corpus you can watch it memorize. The full index (200+ files) is less degenerate.
- **The architecture is deliberately swappable.** A transformer block is a
  drop-in upgrade for a real GPU/MLX run; it changes capability, not the argument.
  The MLP keeps the manual backprop auditable by eye.
- **The entropy-fooled-by-Japanese result is the most important output**, not a
  bug: confidence signals miss out-of-distribution inputs; you need a categorical
  off-map gate alongside them.

## The counter-position (so this isn't propaganda)

Openness here means **transparency about what a model is** — separate from the
contested question of whether to **release all weights**. Critics of full
open-weight release raise real concerns: irreversibility, misuse and
safety-stripping, cyber/bio uplift. A rigorous advocate keeps those distinct: you
can argue for radical transparency (data, code, docs, evals) without resolving the
weights-release debate, and the OSAI Index scores those as *separate* dimensions
for exactly that reason. Present the scorecard as "here is what is and isn't
disclosed," not "open beats closed on every axis."

## Attribution

The bundled `sample_data/` entries are original, fictional illustrations of the
schema — not drawn from the index. The index and schema are the work of the OSAI
Index / AI Technology Assessment project:

- Project: <https://osai-index.eu>
- Main database (Codeberg): <https://codeberg.org/AI-Technology-Assessment/main-database>
- License: CC-BY 4.0 · Cite: doi:[10.5281/zenodo.15386042](https://doi.org/10.5281/zenodo.15386042)

If you train Colophon on the real index (`--src ./osai`), your corpus is the
index's CC-BY content — cite it per its license. The Colophon code is MIT (see
`LICENSE`).
