# CLAUDE.md — working context for ❧ Colophon

You're picking up a small, finished-but-extendable project. Treat this as a
senior-collaborator handoff: the code works and the design choices are
deliberate, so read the intent below before "improving" anything — several of the
rough edges are load-bearing.

## What this is (one paragraph)

`Colophon` is a ~45K-parameter character language model trained from scratch in
pure NumPy on the European Open Source AI Index (a CC-BY database scoring AI
systems across 14 openness dimensions). A colophon is a book's note on how it was
made; this model is named for that and embodies it — it emits a `colophon.json`
describing its own data, training, and openness grades. It is a **demonstrator,
not a chatbot**: its value is that every property people normally *estimate* about
a model — training-data composition, knowledge cutoff, competence boundary,
confidence signals — is here **ground truth you can audit**, because the corpus is
readable and the code hides nothing behind a framework. Tagline: *"A model that
ships its own colophon."*

## Status: working, verified

- `python colophon.py demo` trains from zero (~10s, one CPU core) and prints
  generation, in- vs out-of-distribution confidence, and the openness scorecard,
  then writes `colophon.npz` + `colophon.json`.
- Last verified: 3 sample files → 6,898 chars, vocab 56, 45,560 params, final
  loss ~0.13. Scorecard: 10/12 open vs 0/12 for a typical closed model.

## Run it

```bash
pip install -r requirements.txt
python colophon.py demo
python colophon.py prepare
python colophon.py --steps 8000 train
python colophon.py generate --prompt "availability_weights_"
```

Global flags (`--src`, `--steps`, `--seed`) go **before** the subcommand
(argparse quirk): `python colophon.py --src ./osai --steps 8000 demo`.

## File map

- `colophon.py` — everything: data loading, the NumPy char-MLP (forward + manual
  backprop + Adam), training, generation, white-box entropy, scorecard, and the
  `colophon.json` writer.
- `colophon.json` — generated. The model's self-description: data section
  (datasheet), training section (model card), and openness scorecard in one file.
- `sample_data/*.yaml` — three **original, fictional** OSAI-schema entries so the
  demo runs with zero external data. Not lifted from the real index.
- `competence_gate.py` — companion from the same design session: the *estimate*
  end (domain-cluster distance + cutoff/temporal parser as a pre-generation gate).
  Colophon is the *ground-truth* end. Not imported by `colophon.py`.
- `test_colophon.py` — stdlib `unittest` + NumPy only. Finite-difference gradient
  check on the manual backprop, the off-map/unknown-char signal, and the unified
  `colophon.json` contract. Run: `python -m unittest test_colophon`.
- `marginalia.py` — the live inspection UI (item #1 of the former "Open work"
  list). Stdlib-only `http.server` + a single vanilla-JS page; loads a trained
  `colophon.npz` and serves `prompt_confidence()` / `scorecard_section()` live.
  No new dependencies. Not imported by `colophon.py`.
- `test_marginalia.py` — stdlib `unittest`; checks `analyze_prompt()` is a
  faithful wrapper around `prompt_confidence()` / `generate()`, including the
  off-map signal on an out-of-distribution prompt.
- `README.md` — full concept/problem/solution + honest limits + counter-position +
  OSAI attribution.

## Design intent — do NOT "fix" these

- **Pure NumPy, single hidden layer, by choice.** No PyTorch/MLX. The point is
  that the model, gradients, and optimizer are auditable by eye. Keep it
  dependency-free unless a task explicitly asks to scale up.
- **It overfits the tiny sample. That's on-theme.** Don't add regularization to
  "improve" the 3-file demo; the fix is more data (the real index), not hiding it.
- **The entropy-fooled-by-Japanese result is a feature.** On an OOD prompt the
  model reported *lower* entropy than on its own text while every char was unseen.
  That's the core lesson: confidence misses OOD; you need a categorical off-map
  signal alongside it. Keep it visible in any UI.
- **Architecture is deliberately swappable.** A transformer block is a drop-in
  upgrade; it changes capability, not the argument.

## Constraints / environment notes

- Real index: <https://codeberg.org/AI-Technology-Assessment/main-database>
  (CC-BY 4.0). Clone into the gitignored `osai/` and use `--src ./osai`. Do **not**
  vendor it — reference it, per its license.
- Some sandboxes block `codeberg.org` egress; a local machine won't. "Host not in
  allowlist" is the egress allowlist, not the URL.

## Open work (rough priority)

1. **Transformer option** — a `--arch transformer` path (torch/MLX) for the full
   index; keep the NumPy MLP as the auditable default.
2. **Run on the full OSAI index** and capture a cleaner in/out-of-dist spread.
3. Optional: `pip install -e .` with a `colophon` console entrypoint.

Done: **tests** — `test_colophon.py` covers the finite-difference gradient check
and the entropy/off-map signals (was item #4).
Done: **Marginalia** — the live inspection UI (`marginalia.py`, stdlib-only
`http.server` + a single-page frontend) shows live entropy, the off-map/
unknown-char flag, and the OSAI scorecard against a trained `colophon.npz`
(was item #1).

## The print-shop family (future naming)

If this grows into a suite, names come from the same vocabulary and each reinforces
the openness pitch:

- **Colophon** — the model / artifact (this repo).
- **Errata** — the eval + error-analysis report (what it got wrong, published).
- **Marginalia** — the live inspection UI (item 1 above).
- **Incipit** — a prompt / inference front-end (a book's opening words; the
  complement of a colophon).
- **Imprint** — the training harness, if it's ever split out.

## Keep it honest

Advocacy-adjacent. The README's counter-position section is intentional and must
stay: "transparency about what a model is" ≠ "release all weights," and open-weight
release has real critics (irreversibility, misuse, cyber/bio uplift). Present the
scorecard as "here's what is and isn't disclosed," never "open beats closed on
every axis." Don't let it drift into propaganda.

## Attribution reminder

Training on the real index → the corpus is CC-BY; cite doi:10.5281/zenodo.15386042.
Colophon code is MIT (`LICENSE`), © schmug.
