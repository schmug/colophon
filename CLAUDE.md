# CLAUDE.md — working context for `openlm`

You're picking up a small, finished-but-extendable project. Treat this as a
senior-collaborator handoff: the code works and the design choices are
deliberate, so read the intent below before "improving" anything — several of
the rough edges are load-bearing.

## What this is (one paragraph)

`openlm` is a ~45K-parameter character language model trained from scratch in
pure NumPy on the European Open Source AI Index (a CC-BY database that scores AI
systems across 14 openness dimensions). The model learns to describe model
openness, then gets scored against those same 14 dimensions. It is a
**demonstrator, not a chatbot**: its entire value is that every property people
normally have to *estimate* about a model — training-data composition, knowledge
cutoff, competence boundary, confidence signals — is here **ground truth you can
audit**, because the corpus is small enough to read and the code has no framework
hiding anything. The thesis: it argues for transparency by *being* auditable, not
by asserting anything.

## Status: working, verified

- `python openlm.py demo` trains from zero (~10s, one CPU core) and prints:
  generation, in- vs out-of-distribution confidence, and the openness scorecard.
- Last verified run: 3 sample files → 6,898 chars, vocab 56, 45,560 params,
  final loss ~0.13. Scorecard: 10/12 open (this artifact) vs 0/12 (typical closed).

## Run it

```bash
pip install -r requirements.txt          # just numpy
python openlm.py demo                     # full showcase on bundled sample_data
python openlm.py prepare                  # print the datasheet-style manifest only
python openlm.py --steps 8000 train       # train + save model.npz
python openlm.py generate --prompt "availability_weights_"
```

Global flags (`--src`, `--steps`, `--seed`) go **before** the subcommand
(argparse quirk): `python openlm.py --src ./osai --steps 8000 demo`.

## File map

- `openlm.py` — everything: data loading, the NumPy char-MLP (forward + manual
  backprop + Adam), training, generation, white-box entropy, and the scorecard.
- `sample_data/*.yaml` — three **original, fictional** OSAI-schema entries so the
  demo runs with zero external data. Not lifted from the real index.
- `competence_gate.py` — companion module from the same design session. It's the
  *estimate* end of the spectrum (domain-cluster distance + cutoff/temporal
  parser as a pre-generation gate); `openlm` is the *ground-truth* end. Not
  imported by `openlm.py`; kept here because the README cross-references it.
- `README.md` — the full concept/problem/solution writeup + honest limits +
  counter-position + OSAI attribution.

## Design intent — do NOT "fix" these

- **Pure NumPy, single hidden layer, by choice.** No PyTorch/MLX. The whole point
  is that the model, its gradients, and its optimizer are auditable by eye. Keep
  it dependency-free unless the task explicitly asks to scale up.
- **It overfits the tiny sample. That's on-theme.** With a fully known corpus you
  can literally watch it memorize. Don't add regularization to "improve" the demo
  on 3 files; the fix is more data (the real index), not hiding the behavior.
- **The entropy-fooled-by-Japanese result is a feature, not a bug.** On an
  out-of-distribution prompt the model reported *lower* entropy than on its own
  text while every char was unseen. That reproduces the core lesson: confidence
  signals miss OOD; you need a categorical off-map signal alongside them. Keep
  that result visible in any UI you build.
- **Architecture is deliberately swappable.** A transformer block is a drop-in
  upgrade for a real GPU/MLX run; it changes capability, not the argument.

## Constraints / environment notes

- The real index lives at
  <https://codeberg.org/AI-Technology-Assessment/main-database> (CC-BY 4.0). Clone
  it into the gitignored `osai/` and run `--src ./osai`. Do **not** vendor it into
  the repo — reference it, per its license.
- Some sandboxes block `codeberg.org` egress; a local machine won't. If a clone
  fails with "Host not in allowlist", that's the egress allowlist, not the URL.

## Open work (rough priority)

1. **Live UI layer.** A small local page: type a prompt, watch entropy, the
   off-map/unknown-char flag, and the scorecard update in real time. Turns this
   from a terminal demo into something presentable to leadership.
2. **Transformer option.** Add a `--arch transformer` path (torch or MLX) for a
   real run on the full index; keep the NumPy MLP as the auditable default.
3. **Run on the full OSAI index** and capture a cleaner in/out-of-dist spread.
4. **Tests** for the manual backprop (finite-difference gradient check) and the
   entropy/off-map signals.
5. Optional: package as a `pip install -e .` with a console entrypoint.

## Keep it honest

This is advocacy-adjacent. The README's counter-position section is intentional
and must stay: "transparency about what a model is" is a separate claim from
"release all weights," and open-weight release has real critics (irreversibility,
misuse, cyber/bio uplift). Present the scorecard as "here's what is and isn't
disclosed," never "open beats closed on every axis." Don't let the artifact drift
into propaganda.

## Attribution reminder

If you train on the real index, the corpus is CC-BY content — cite
doi:10.5281/zenodo.15386042. The `openlm` code is MIT (`LICENSE`); update the
copyright holder from the `schmug` placeholder.
