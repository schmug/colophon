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
python colophon.py generate --prompt "weights_basemodel:"
```

Global flags (`--src`, `--steps`, `--seed`) go **before** the subcommand:
`python colophon.py --src ./osai --steps 8000 demo`.

`demo`'s sample generation and its IN/OUT confidence prompts are hardcoded to
the OSAI-index schema (keys like `weights_basemodel:`, `datasheet:`,
`licenses:`). On a `--src` corpus that doesn't use that schema, `demo` prints a
warning instead of silently mislabeling in-corpus text as off-map -- use
`marginalia.py`'s live inspector for other corpora instead.

Alternatively, install it as a package and use the `colophon` console command:

```bash
pip install -e .
colophon demo
colophon generate --prompt "weights_basemodel:"
```

`python colophon.py ...` keeps working unchanged either way.

Bundled `sample_data/` (three OSAI-style entries) lets it run with zero external
data. For the real thing, clone the index and point at it:

```bash
git clone https://codeberg.org/AI-Technology-Assessment/main-database osai
python colophon.py --src ./osai --steps 25000 demo
```

### Measured on the real index

A run on the full index (snapshot: **196 files, 579,109 chars, vocab 98**; corpus
`sha256` recorded in `colophon.json`) trains **51,986 params** to a final-step
training loss of **~0.64** in ~11s on one CPU core. Unlike the 3-file sample it
does **not** collapse to a memorized ~0.13 — 579K characters can't be stored in
52K parameters, which is the point: *more data, not regularization*, is the fix.

The in- vs out-of-distribution spread comes out clean (normalized next-char
entropy, 0 = certain, 1 = uniform):

| Prompt | Entropy | |
|---|---|---|
| `weights_basemodel:\n    class: ` (in-dist) | **0.148** | |
| `datasheet:\n    class: ` (in-dist) | **0.209** | |
| `licenses:\n    class: ` (in-dist) | **0.179** | |
| "The mitochondria is the powerhouse of the cell." (out) | 0.311 | fluent, but not in its world |
| "In 2027 the election results showed" (out) | 0.338 | fluent, but not in its world |
| 日本語で書いてください (out) | 0.590 | **10 distinct chars never seen** — off-map flag fires |

In-distribution prompts (~0.15–0.21) sit well below the out-of-distribution ones,
and the Japanese case shows the lesson the whole project is built around: entropy
alone under-reacts to out-of-distribution input, so the categorical off-map signal
(unseen characters) has to sit next to it. These figures are a snapshot — the index
grows over time; re-run to refresh, and check the `sha256` in `colophon.json` for
the exact corpus you trained on.

### Optional: transformer architecture

The NumPy MLP above is the default and the auditable reference. For a
higher-capacity run on the full index, install PyTorch and pass
`--arch transformer` to `prepare`/`train`/`demo`:

```bash
pip install torch
python colophon.py --arch transformer --src ./osai --steps 8000 train
python colophon.py generate --prompt "weights_basemodel:"   # arch is read back from colophon.npz
```

Swapping architectures changes capability, not the argument: `generate` and the
entropy/off-map confidence signals work identically on either arch, and
`colophon.json`'s training section still records exactly what was trained
(`"arch": "transformer"` or `"mlp"`). Omitting `--arch` keeps today's exact
dependency-free behavior — torch is never imported unless you ask for it.

Outputs (gitignored, regenerable): `colophon.npz` (weights) and `colophon.json`
— the model's own colophon: data section (datasheet), training section (model
card), and its openness scorecard, in one self-describing file.

## Marginalia — live inspection UI

Once you've trained a model, `marginalia.py` serves a small local page that
makes the same signals `demo` prints interactive:

```bash
python colophon.py demo      # or `train`; writes colophon.npz
python marginalia.py         # serves http://127.0.0.1:8765
```

Type a prompt and its white-box signals update live (detailed below). The
off-map/unknown-character flag is always shown as its own indicator, never
folded into entropy — entropy can be fooled by out-of-distribution input (see
below), so the categorical signal has to stay separate. Marginalia adds no
dependencies: the server is Python's stdlib `http.server` and the page is plain
HTML/JS with no build step or CDN calls; it's local-only and reuses
`inspect_prompt()` / `context_saliency()` / `scorecard_section()` from
`colophon.py` rather than re-deriving those signals in JavaScript.

### What Marginalia shows

Marginalia is a live, zero-dependency inspector for a trained Colophon model.
Type a prompt and every white-box signal the model has is rendered from its own
weights: a **confidence heatmap** (each character tinted by next-char entropy),
the **literal K-character context window** the model saw (with the pad horizon it
cannot see past), **occlusion-based context saliency** (which remembered
characters actually drove the prediction), a **top-k next-char inspector** with
where the real next character ranked, an **embedding-space panel** (the entire
character embedding table, PCA-projected to 2D so it fits on screen exactly,
with click-to-see nearest neighbors by cosine similarity), and the OSAI
**openness scorecard**. It is framed as the honest counterpart to black-box
"observability" tools: a hosted API exposes none of this, and where a tool
like glassboxllm has to *simulate* per-token confidence, Colophon reads it
straight from the weights.

### Teaching mode — a corpus you can grade by eye

The openness index is real and rich, but it's jargon a layperson can't check, so
they have to take "94% sure" on faith — the exact thing Colophon exists to
abolish. So Marginalia ships a second corpus whose ground truth already lives in
your head: **the periodic table** (118 elements, `number / symbol / name /
period`). Toggle to it and the confidence signals become auditable against facts
you already know.

```bash
# train the teaching model to its own weights (never clobbers colophon.npz)
python colophon.py --src teaching_data/elements --out elements.npz --steps 4000 train
python marginalia.py        # the page now shows an "Openness index / Periodic table" toggle
```

The payoff is a lesson you can verify yourself: ask for `number: 26` and it
confidently answers `Fe / Iron` — correct. Ask for a **made-up** `number: 250`
and it stays *just as* confident (~94%, off-map flag clean) while inventing a
fake element. The friendly percentage and the character-level off-map check both
miss it; only reading the corpus — or knowing 250 isn't an element — catches it.
That's the "fluent, certain, and wrong" failure made legible on common
knowledge. The 118 files are checked in and regenerable
(`python teaching_data/build_elements.py`), so the data can't silently drift from
its cited source.

## Incipit — multiturn glass-box chat

Marginalia inspects one prompt. `Incipit` inspects a **conversation**: chat
bubbles on top, and underneath them the truth — one growing text tape, the
literal K-character window the model re-reads on every turn, and per-character
records (probability, entropy, top-k, occlusion saliency) computed straight
from the weights via `colophon.inspect_prompt()`. It's the honest realization
of a UI that was mocked with fake numbers in 2025 (the `glassboxllm-2025-mock`
prototype): every signal Incipit shows is real, because there's a real model
underneath.

An incipit is a book's opening words — the complement of a colophon.

### The three-act teaching sequence

Incipit ships a guided rail of three acts, each isolating one variable:

1. **"It's just a tape."** Send `number: 1\n` to the YAML-trained
   periodic-table model (`elements64`, K=64) and watch it complete cleanly —
   `symbol: H\nname: Hydrogen\nperiod: 1`. Then look left: your "conversation"
   is one growing string, and the highlighted band is the *only* part the
   model can see. Keep going until early turns slide out of the window — there
   is no memory beyond that band.
2. **"Chat is a format it never saw."** Same model, same K=64 window — but
   wrap the turn as `user: ... / model: ...` and ask `what is element 26?`.
   It was trained on YAML, not conversation, so it completes YAML-ish noise,
   and `?` isn't even in its vocabulary (watch the off-map flag fire). The
   failure is the training data's *format*, not the window.
3. **"Train on dialogue, get a chatbot."** Identical architecture (K=64,
   ~2M params), the same 118 facts — but the training corpus is
   `user:/model:` dialogues instead of YAML. Now chat format completes
   correctly: `what is element 26?` → `element 26 is Iron (Fe), in period 4.`
   Ask the follow-up `which period is it in?` and it answers `Iron is in
   period 4.` — correct only because the earlier turn is still inside the
   K=64 window (check the saliency panel). The whole trick: **a chatbot is a
   completion model whose training data contains conversations.**

Acts 2 and 3 train with identical hyperparameters (`--K 64 --E 64 --H 512`)
so the comparison isolates exactly one variable — the corpus's format, not
model size or training length.

### Stateless by design

The server holds no conversation state. Every `/api/turn` request carries the
full turn history; the server rebuilds the tape, runs the forward pass, and
returns fresh records. This mirrors how real hosted LLM APIs work — "memory"
is just the client re-sending transcript text that gets re-read from scratch —
and it keeps `/api/turn` deterministic given the same request + seed.

### Run it

```bash
cd incipit && npm install && npm run build && cd ..

python colophon.py --src teaching_data/elements --out elements_k64.npz \
    --steps 60000 --K 64 --E 64 --H 512 --lr 0.0005 train
python colophon.py --src teaching_data/dialogue --out dialogue_k64.npz \
    --steps 60000 --K 64 --E 64 --H 512 --lr 0.0005 train

python incipit.py        # serves http://127.0.0.1:8790
```

`incipit.py` is stdlib `http.server` + NumPy, same as Marginalia; it serves the
built `incipit/dist/` output, so once the front-end is built, production runs
on Python alone. If `incipit/dist/` is missing, the server degrades to a build
help page instead of crashing.

### Honest limits

- **The dialogue model needs `--lr 0.0005`, not the 3e-3 default.** At the
  default learning rate the fixed-LR optimizer (which by design has no
  schedule) oscillates and never cleanly settles on the harder ~63K-char
  dialogue corpus; `--lr 0.0005` finds a stable basin. Both K=64 models use
  the same lower rate so the Act 2 vs. Act 3 comparison isolates the training
  data's format, not a hyperparameter difference.
- **These teaching models substantially memorize their tiny corpora.**
  Disclosed and on-theme, same as Colophon's own 3-file demo — not hidden,
  not "fixed" with regularization.
- **Not a chatbot in the capability sense.** ~2M parameters trained on 118
  facts reproduces those facts; it doesn't reason or generalize past them.
- **Cold-start completion of an arbitrary mid-corpus element is imperfect.**
  The model memorized the corpus as one long sequence, so a prompt matching
  the corpus's actual start (element 1) completes cleanly, but an arbitrary
  element's YAML block, seeded from a synthetic PAD-only context, does not
  reproduce as reliably. That's exactly what the entropy/off-map signals are
  for — the imperfection is auditable instead of hidden.

## Honest limits — read before showing anyone

- **It is not intelligent.** A char-level MLP models local character statistics.
  It produces corpus-*shaped* text, not reasoning. Don't read coherence into it.
- **It overfits the tiny sample**, by design and on-theme — with a fully known
  corpus you can watch it memorize (final loss ~0.13 on 3 files). The full index
  (196 files, 579K chars) does not: loss settles near ~0.64 and the in/out-of-dist
  spread sharpens — see "Measured on the real index" above.
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

The periodic-table teaching corpus (`teaching_data/`) is built from element
facts — atomic numbers, symbols, and IUPAC names/spellings — which are
public-domain data (not copyrightable). Source of record: the
[IUPAC Periodic Table of the Elements](https://iupac.org/what-we-do/periodic-table-of-elements/).
Group is deliberately omitted: the f-block's group assignment is genuinely
unsettled, and this corpus ships only undisputed facts.
