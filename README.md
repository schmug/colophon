# openlm — a fully-open, from-scratch model that argues for openness by being it

A ~45K-parameter character language model, trained from zero in pure NumPy on the
**European Open Source AI Index** — the CC-BY database that scores AI systems
across 14 dimensions of openness. The model learns to describe model openness,
and is then scored against the very same 14 dimensions. The recursion is the
point: it advocates for transparency not by *asserting* it, but by being an
artifact whose every property you can *verify*.

This is a demonstrator, not a chatbot. Its value is entirely structural.

## The argument, in three moves

**Concept.** A model only knows its training distribution. Usually that
distribution is unknowable to you — you estimate it from benchmarks and tokenizer
tricks. Here you can *read the entire corpus*. Nothing else exists to the model.

**Problem.** Outside its distribution, a model is confidently wrong, and its own
confidence signals don't reliably warn you. The demo shows this directly: on a
Japanese prompt the model reported *lower* next-char entropy than on its own
native text — it "felt" certain — while being categorically off-map (every
character unseen in training). Entropy was fooled; only the separate off-map
signal caught it. With mainstream closed models you get neither the corpus to
check against nor, often, the raw signals to compute — which is exactly why the
OSAI Index exists: most systems score red on data transparency.

**Solution.** Full openness across all 14 dimensions turns every trust question
from an estimate into a fact you can audit. This model is 10/12 open (honestly
red on "preprint" and "paper" — we don't hand ourselves free greens); a typical
closed API model is 0/12. The contrast *is* the pitch.

## What the pieces map to (tie-in with the competence gate)

This is the ground-truth end of the same spectrum as `competence_gate.py`:

| Signal from the gate work | Here it is… |
|---|---|
| Training-data composition | Known exactly — you wrote/curated the corpus |
| Knowledge cutoff | Absolute — the corpus *is* the model's universe |
| Domain competence map | The literal edge of the corpus |
| Token-level confidence | Fully white-box (we own every logit) |
| Off-map detection | Unknown-character flag (categorical, un-foolable) |

## Run it

```bash
python openlm.py demo                 # train + generate + confidence + scorecard
python openlm.py prepare              # print the datasheet-style manifest only
python openlm.py --steps 8000 train   # train and save model.npz
python openlm.py generate --prompt "availability_weights_"
```

Bundled `sample_data/` (three OSAI-style entries) lets it run with zero external
data. For the real thing, clone the index and point at it:

```bash
git clone https://codeberg.org/AI-Technology-Assessment/main-database osai
python openlm.py --src ./osai demo
```

Outputs written next to the script: `model.npz` (weights),
`data_manifest.json` (the datasheet: file list, char count, SHA-256, scope),
`train_manifest.json` (the model card: architecture, params, loss, seed, wall-clock).

## Honest limits — read before showing anyone

- **It is not intelligent.** A char-level MLP models local character statistics.
  It produces corpus-*shaped* text, not reasoning. Do not let an audience read
  coherence into it.
- **It overfits the tiny sample**, by design and on-theme — with a fully known
  corpus you can literally watch it memorize. The full index (200+ files) is less
  degenerate.
- **The architecture is deliberately swappable.** A transformer block is a
  drop-in upgrade for a real GPU/MLX run; it changes capability, not the
  argument. The MLP keeps the manual backprop small enough to audit by eye.
- **The entropy-fooled-by-Japanese result is the most important output**, not a
  bug. It reproduces the thread's core caveat: confidence signals miss
  out-of-distribution inputs; you need a categorical off-map gate alongside them.

## The counter-position (so this isn't propaganda)

Openness here means **transparency about what a model is** — which is separately
valuable from the contested question of whether to **release everything**. Critics
of full open-weight release raise real concerns: irreversibility (weights can't be
recalled), misuse and safety-stripping by downstream actors, and cyber/bio uplift.
A rigorous advocate keeps those distinct: you can argue for radical transparency
(data, code, docs, evals) without resolving the weights-release debate, and the
OSAI Index itself scores those as *separate* dimensions for exactly that reason.
Present the scorecard as "here is what is and isn't disclosed," not "open beats
closed on every axis for every purpose."

## Attribution

The bundled `sample_data/` entries are original, fictional illustrations that
imitate the schema of the **European Open Source AI Index** — they are not drawn
from the index itself. The index and its schema are the work of the OSAI Index /
AI Technology Assessment project:

- Project: European Open Source AI Index — <https://osai-index.eu>
- Main database (Codeberg): <https://codeberg.org/AI-Technology-Assessment/main-database>
- License: CC-BY 4.0
- Cite: doi:[10.5281/zenodo.15386042](https://doi.org/10.5281/zenodo.15386042)

If you train `openlm` on the real index (`--src ./osai`), your training corpus is
the index's CC-BY content, so cite it per its license in anything you publish.

The `openlm` code in this repository is licensed separately under the MIT License
(see `LICENSE`).
