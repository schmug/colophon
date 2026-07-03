# Incipit — design spec

Date: 2026-07-03
Status: approved (brainstorm 2026-07-03)
Repo: colophon (this repo). The `glassboxllm` repo is referenced but never modified.

## 1. Problem and pitch

GlassBox AI (`~/glassboxllm`, an AI Studio React mock from 2025) designed the right
UX for a transparent chat interface — per-token probability, entropy, top-k
candidates, attention buckets, token banning, context include/exclude, a context
budget meter — but had to **simulate every metric** (its `analysisUtils.ts` says so
outright: metrics are a hash of the token text; Gemini writes the words, the
introspection is theater).

Colophon computes every one of those signals **for real**, but Marginalia is
single-prompt: it answers "what did the model see and believe for this one
prompt?", not "what happens across a conversation?".

**Incipit** is the third print-shop artifact (the name was already reserved in
CLAUDE.md for an inference front-end): a multiturn, maximally-inspectable chat
front-end for Colophon models, realizing the GlassBox mock with every number
honest. Its teaching centerpiece is a three-act sequence in which the only
experimental variable is training data.

Goals, in the user's words: maximal learning, inspection, openness; full token and
context introspection; demoing multiturn context.

## 2. Decisions made during brainstorming

- **Backend honesty:** Colophon only. Every displayed signal is computed from the
  weights. No cloud-model contrast pane, no simulated metrics anywhere.
- **Venue:** new `incipit/` front-end inside the colophon repo (Vite + React + TS
  — the stdlib-only constraint is explicitly loosened for Incipit's front-end;
  `marginalia.py` and `colophon.py` remain dependency-free, numpy-only). The
  Python API server also serves the built bundle, so runtime needs no Node.
- **Multiturn framing:** both honest-completion and chat-format, staged as a
  three-act teaching sequence (below), plus free-play mode.
- **Third act is demonstrated, not asserted:** a tiny dialogue-formatted corpus
  and model are built and trained so "chatbots are completion models whose
  training data contains conversations" is shown as ground truth.
- **v1 affordances:** all four tiers — sampling lab (temperature/top-k sliders +
  three-candidate picker), logit surgery (ban a char via a real logit mask),
  context surgery (include/exclude/edit turns), session aggregates.
- **Bigger models:** K=64 models are trained for the teaching sequence, and
  hyperparameters (K/E/H) become CLI flags on `colophon.py` so larger variants
  are always an option. The flagship `colophon.npz` is unchanged.

## 3. The three-act teaching sequence

- **Act 1 — "It's just a tape."** Chat with `elements_k64`. Completion works; the
  raw-tape view shows the "conversation" is one growing document; the K-char
  window is highlighted and slides; old turns visibly fall off and are forgotten.
- **Act 2 — "Chat is a format it never saw."** Same architecture, same window,
  same facts — but the corpus was YAML, so `user: what is element 26?` completes
  as YAML-ish noise. Because Act 2's model is also K=64, the failure is
  attributable to exactly one thing: the training data. (This is why Act 2 does
  NOT use the K=12 flagship — a skeptic could blame the window.)
- **Act 3 — "Train on dialogue, get a chatbot."** `dialogue_k64` (identical
  K/E/H, trained on Q&A over the same 118 elements) answers in chat format.

After the rail: free-play with any available model and all instruments.

## 4. Models

| Model file | Corpus | K / E / H | ~Params | Role |
|---|---|---|---|---|
| `colophon.npz` | OSAI index | 12 / 24 / 128 (unchanged) | 52K | Flagship; optional Incipit mode (tiny-window drama) |
| `elements.npz` | elements YAML | 12 / 24 / 128 (unchanged) | ~46K | Marginalia teaching mode; not used by Incipit |
| `elements_k64.npz` | elements YAML | 64 / 64 / 512 | ~2.1M | Acts 1–2 (controlled comparison, YAML arm) |
| `dialogue_k64.npz` | dialogue corpus (new) | 64 / 64 / 512 | ~2.1M | Act 3 (dialogue arm) |

Rationale for K=64: `user: what is element 26?\nmodel: ` is ~35 chars; K=12
cannot see the question when the answer starts. E/H scale to ~2.1M params —
minutes to train on the reference machine (M4 Max: measured ~0.8 TFLOP/s fp64
GEMM in numpy). A ~2M-param model on a ~50–70K-char corpus will substantially
memorize; for a facts-recall teaching model this is the disclosed, on-theme point
(the corpus IS the ground truth being recalled).

`.npz` weights files remain generated artifacts (not committed), consistent with
the existing convention; training commands are documented in the README and
encoded in tests/scripts.

## 5. Dialogue corpus

`teaching_data/build_dialogue.py` → `teaching_data/dialogue/*.txt`.

Conventions mirror the elements corpus exactly: deterministic generation, files
checked in and byte-for-byte regenerable from the committed script, no provenance
comments inside data files (vocab pollution), IUPAC-sourced undisputed fields
only (number / symbol / name / period; group intentionally omitted).

Content: ~4–6 Q&A templates per element ("user: what is element 26?\nmodel:
element 26 is Iron (Fe), in period 4.\n" and variants asking symbol / name /
period), lowercase `user:` / `model:` role markers, newline as turn separator,
blank line between dialogues. Element names keep their canonical capitalization
(they are facts). Target size ~50–70K chars, ~500–700 dialogues.

## 6. Backend — `incipit.py`

Stdlib-only `http.server`, same skeleton and degradation philosophy as
`marginalia.py`. **Stateless by design:** the client sends the full turn list on
every request; the server holds no conversation state. This mirrors real LLM APIs
and is itself part of the statelessness lesson.

Endpoints (JSON):

- `GET /api/modes` — registered models + metadata: K/E/H, param count, corpus
  blurb, which acts they serve, availability. A mode whose weights file is absent
  is reported disabled; using it returns 503. Never a crash.
- `POST /api/turn` — request `{mode, turns: [{role, text, excluded}], sampling:
  {temperature, top_k, seed, max_chars, banned_chars}}`. The server builds the
  tape (raw concatenation with role markers; the exact tape string is returned
  verbatim — the tape format is part of the lesson), samples a continuation, and
  returns: the continuation; per-char records (char, true prob, entropy, top-k
  candidates, the literal K-window it saw, slot types, off-map flag); window
  spans over the tape per generated char; `confidence_readout()`; and
  `find_source_echo()` hits. The three-candidate picker is three `/api/turn`
  calls with different temperature/seed — each candidate fully inspectable
  before the user commits one to the transcript.
- `POST /api/saliency` — `{mode, tape, pos}` → occlusion saliency (wraps
  `colophon.context_saliency`).
- Static serving of `incipit/dist/`; if the bundle is missing, a plain page
  explains how to build it.

Errors: unknown mode → 400; absent weights → 503; bad `pos` → 400; tape over cap
(~10,000 chars) → 413. Off-map characters in user input are **not** errors —
they are flagged teaching signals in the response.

CLI: `python incipit.py --port PORT --osai-npz colophon.npz --elements64-npz
elements_k64.npz --dialogue-npz dialogue_k64.npz` — one flag per mode with those
defaults, following Marginalia's `--npz`/`--elements-npz` precedent. Incipit
registers three modes (osai, elements64, dialogue); `elements.npz` (K=12) stays
Marginalia-only.

## 7. Changes to `colophon.py` (small, tested)

- Sampling: thread `top_k` and `banned_ids` (a logit mask applied before
  softmax/sampling) through `generate()` and the sampling path Incipit uses.
  Existing `temp` parameter is already present.
- CLI: `--K`, `--E`, `--H` hyperparameter flags (global flags, before the
  subcommand, like `--src`/`--steps`/`--out`), defaulting to current values.
- No change to the flagship training defaults, the manual-backprop core, or the
  numpy-only requirement.

## 8. Front-end — `incipit/` (Vite + React + TS)

GlassBox's three-panel layout, ported with credit; `geminiService.ts` and
`analysisUtils.ts` get no successor — nothing is simulated.

- **Left — Tape panel** (was ContextPanel): raw transcript-as-text with the
  K-window highlighted and sliding; per-turn include/exclude toggles and
  edit-in-place (context surgery: delete a turn and the "memory" is gone on the
  next generation); honest budget meter (tape length vs K, chars fallen out of
  window); mode switcher + model card (params, K, corpus, openness-scorecard
  link).
- **Center — Conversation** (was ChatInterface): chat bubbles with a
  bubbles ↔ raw-tape toggle; composer with temperature and top-k sliders;
  three-candidate picker; the Act 1/2/3 guided rail with preloaded prompts and
  teaching copy; free-play after.
- **Right — Char inspector** (was TokenInspector): click/hover any generated
  char → true prob, entropy, top-k bars, the exact window it saw, on-demand
  saliency, off-map flag; ban-char button feeding the real logit mask; session
  aggregates (median entropy/confidence, off-map counts per turn).

Slot-type color semantics follow Marginalia's palette (pad / off_map / user turn
/ model turn) so the two UIs read as siblings.

Runtime topology: `npm run build` emits `incipit/dist/`; `python incipit.py`
serves bundle + API from one process/port (no CORS in production). During
front-end development, Vite's dev server proxies `/api` to the Python server.

## 9. Testing (TDD, stdlib conventions)

- `test_incipit.py` — live ephemeral server (pattern of `test_marginalia.py`):
  modes contract + missing-weights degradation; `/api/turn` per-char record
  contract; determinism (identical request + seed → identical response bytes,
  demonstrating statelessness); temperature=0 → greedy; banned char absent from
  output; top-k honored; saliency route; 400/413/503 paths.
- `test_dialogue.py` — corpus pins mirroring `test_elements.py`: byte-for-byte
  regeneration from the builder, entry counts, vocab cleanliness, spot-checked
  facts; behavioral check that a trained dialogue model answers a known question
  (`element 26` → output contains `Iron`), following whatever train-or-load
  convention `test_elements.py` uses.
- `colophon.py` sampling additions get unit tests beside the existing
  gradient-check suite (mask excludes chars; top-k restricts support; flags
  round-trip through the CLI).
- Front-end: vitest on pure logic only (tape assembly, window-span math). No
  component-test suite in v1.

## 10. Docs and hygiene

- README: Incipit section (concept, run instructions, the three acts, honest
  limits — memorization disclosure, "not a chatbot in the capability sense").
- CLAUDE.md: file-map entries (`incipit.py`, `incipit/`, dialogue corpus,
  tests), design-intent notes (statelessness is deliberate; Act 2 must share
  Act 3's hyperparameters; front-end toolchain allowed for Incipit only).
- GlassBox credited in `incipit/README.md` as the design ancestor.
- Attribution: dialogue corpus facts cite IUPAC like elements; OSAI citation
  unchanged (doi:10.5281/zenodo.15386042); code stays MIT.
- Follow-up issues filed for anything scope-cut during implementation.

## 11. Out of scope (v1)

- No cloud-model contrast pane (explicitly rejected).
- No changes to Marginalia beyond none-at-all (it keeps `elements.npz` K=12).
- No torch/transformer path in Incipit; numpy MLP only.
- No modification to the glassboxllm repo.
- No streaming/SSE generation (responses return complete; revisit if latency at
  K=64 warrants it).
- No persistence of conversations server-side (statelessness is the design).
