# Marginalia Context Inspector — design

**Date:** 2026-07-01
**Status:** approved (design), pending implementation plan
**Repo:** Colophon (`marginalia.py`, `colophon.py`)

## Problem

Marginalia today reduces Colophon's white-box signals to a single number. It
computes per-character entropy inside `prompt_confidence()` ([colophon.py:363](../../../colophon.py))
but returns only the *mean*, shows one sampled continuation, an off-map flag, and
the static scorecard. The richest thing Colophon has — that every property people
normally *estimate* about a model is here **ground truth you can audit** — is
invisible in the live UI.

Two sibling projects were reaching for exactly this observability and inform the
upgrade:

- **glassboxllm** — per-token inspection: confidence heatmap, entropy, top-k
  alternatives, attention-by-source, a click-to-inspect panel, session
  aggregates ("anchors"), context-window budget. Its defining limitation: the
  metrics are **simulated** (`analysisUtils.ts`) because Gemini exposes no logits.
- **contextbuddy** — context-hygiene: per-turn scored dimensions, mechanical
  sentinel signals (loop, context-window pressure), and — the transferable core —
  **concrete positional rationales** ("char 47: predicts 'e' (0.31), truth 'a',
  entropy 1.2 bits"), never a bare "low confidence".

The convergent idea: make the model's *use of its context* legible — per-position
confidence, what was in the window, what else it considered, and where it breaks
down. Colophon can do this **for real**: it is genuinely white-box, and its
"context window" is literal — the MLP sees only the last `K=12` characters
([colophon.py:142](../../../colophon.py)).

## Goal

Turn Marginalia from a single-number readout into a **maximal per-position
context inspector**, so that a user demoing Colophon comes away understanding
*viscerally why black-box LLMs are a problem*: every signal a hosted API hides —
per-character confidence, the exact bytes in the context window, the full
next-char distribution it rejected, and which remembered characters actually
drove the prediction — is here, read straight from the weights and auditable.
"Why can't I get this from GPT/Gemini/Claude's API?" is the question the demo
should answer by making the contrast concrete at every view.

All of this while keeping the repo's hard constraint: **stdlib `http.server` + a
single-page vanilla-JS frontend, zero new dependencies.**

The black-box contrast is therefore a **first-class, recurring design element**,
not a footnote: each inspection view carries a paired "what a closed API shows
you here: —" annotation, and a framing banner states that everything below is
read from `colophon.npz` and that this opacity is the problem Colophon exists to
demonstrate.

Non-goals: no build step, no charting library, no framework, no changes to the
NumPy MLP's math or to `prompt_confidence()`'s outputs, no reframing of the
project's honest counter-position (transparency about what a model is ≠ "open
beats closed on every axis" — the scorecard's honest framing stays).

## Approach (chosen: A — additive core)

Add one new function to `colophon.py` that returns per-position records; leave
`prompt_confidence()` byte-for-byte unchanged so the demo, `colophon.json`, and
existing tests do not move. Marginalia grows five views on the existing single
page. Marginalia re-derives no model signals — it renders and aggregates values
the new function returns.

Rejected: **B** (make `prompt_confidence` a wrapper) — touches a function the
demo/tests/`colophon.json` depend on for no user-visible gain. **C** (tabbed
multi-view UI) — more JS, drifts from the single-glanceable-page spirit.

## Architecture & data flow

```
browser (vanilla JS, one page)
  → GET /api/analyze?prompt=...         (debounced 200ms, per keystroke)
    → marginalia.analyze_prompt()       (thin HTTP-layer wrapper)
      → colophon.inspect_prompt()       (SOURCE OF TRUTH: per-position records)
  ← { records:[...], off_map, unknown_chars, ... }
  → JS computes aggregates, focuses the lowest-confidence position, renders views
  → GET /api/saliency?prompt=...&pos=N  (on focus change only, NOT per keystroke)
    → marginalia.context_saliency()     (thin wrapper)
      → colophon.context_saliency()     (occlusion over the K-char window)
  ← { window:[{char, delta}...] }
GET /api/scorecard → colophon.scorecard_section()   (unchanged)
```

Splitting saliency onto its own endpoint keeps the per-keystroke path cheap:
`/api/analyze` stays one forward pass per position, and the K extra occlusion
passes run only for the single focused position, on demand.

## Component: `colophon.inspect_prompt` (new, additive)

```
inspect_prompt(p, stoi, itos, K, text, topk=5, n_continuation=CONTINUATION_LEN,
               seed=0) -> list[record]
```

Teacher-forces through `text`, then appends the model's own sampled continuation,
producing one record per position. Uses the existing `forward()` and the same
normalized-entropy math as `prompt_confidence()` so numbers stay comparable
across the demo, the JSON, and this UI.

Per-position record:

| field             | type              | meaning |
|-------------------|-------------------|---------|
| `char`            | str (len 1)       | the actual next char at this position |
| `display`         | str               | readable form: space→`␣`, newline→`⏎`, tab→`⇥`, else `char` |
| `is_continuation` | bool              | `false` = typed prompt (teacher-forced); `true` = model-sampled |
| `entropy`         | float 0..1        | normalized next-char entropy (same formula as `prompt_confidence`) |
| `top_k`           | list[[str,float]] | up to `topk` `[display_char, prob]`, sorted prob desc |
| `context_window`  | list[str]         | the literal last-`K` chars the model saw; `∅` marks pre-prompt pad (`vocab[0]`) |
| `truth_rank`      | int \| null       | 1-indexed rank of `char` in the full distribution; `null` if off-map |
| `truth_prob`      | float \| null     | prob the model assigned the char that actually came next; `null` if off-map |
| `off_map`         | bool              | `char` never seen in training (not in `stoi`) |

Rules:
- **Context window semantics.** Position `i`'s record describes the prediction
  made *before* consuming `char`; `context_window` is the K-length context used
  for that prediction. Left slots still holding the initial pad render as `∅`
  (they map to `vocab[0]`, the model's blank slate — an honest teaching point,
  not a real character).
- **Off-map positions.** The model still produced a distribution (`top_k` is
  shown), but there is no target embedding for `char`, so `truth_rank`/
  `truth_prob` are `null` and `off_map` is `true`. Context uses `stoi.get(ch,0)`
  exactly as `generate()`/`prompt_confidence()` already do.
- **Continuation.** Sampled via the existing `generate()` path with fixed
  `seed=0` for stable per-keystroke output; each continuation record's `truth_*`
  fields describe the char the model actually emitted.
- **Bounds.** Callers pass `text` already truncated to `MAX_PROMPT_LEN` (500);
  `n_continuation` defaults to `CONTINUATION_LEN` (80). ~580 cheap forward passes
  worst case; `topk=5`.

## Component: `colophon.context_saliency` (new, additive)

```
context_saliency(p, stoi, itos, K, text, pos) -> list[{char, display, delta, is_pad}]
```

The honest, model-derived answer to "which remembered characters actually drove
this prediction" — the real analog of glassboxllm's *simulated* attention-by-
source. It reconstructs the K-char context used at position `pos`, runs the
reference forward pass to get the baseline next-char distribution, then for each
of the K context slots re-runs the forward pass with that slot **occluded**
(replaced by the pad token, `vocab[0]`) and measures how far the distribution
moves. `delta` = total-variation distance (½·Σ|p−p′|) between baseline and
occluded distributions, in [0,1]; larger = that character mattered more. Pad
slots are marked `is_pad` and reported with their delta too (usually ~0 —
another honest teaching point: occluding "nothing" changes nothing).

This is pure NumPy over the existing `forward()`, auditable by eye, and costs K
forward passes per focused position — never on the per-keystroke path.

## Component: `marginalia.analyze_prompt` + `/api/analyze` (revised)

`analyze_prompt()` becomes a thin wrapper returning:

```
{
  "prompt": <str>,
  "records": [ <record>, ... ],       # from inspect_prompt
  "unknown_chars": [ <str>, ... ],    # sorted set of off-map chars (back-compat)
  "off_map": <bool>                   # any off-map char present (back-compat)
}
```

`unknown_chars` / `off_map` are retained so the existing off-map contract (and its
test) still hold. `entropy` mean is no longer a top-level field — the heatmap and
JS-computed median supersede it; the endpoint test is updated accordingly.

`marginalia.context_saliency` + `/api/saliency?prompt=…&pos=N` is a second thin
wrapper over `colophon.context_saliency`, called only when the focused position
changes. Same 503/500 posture as `/api/analyze`. `pos` is clamped to a valid
record index; an out-of-range or missing `pos` returns 400.

## Component: frontend (one page, five regions)

Framing banner at the top: everything below is read straight from
`colophon.npz`; a hosted LLM hides all of it, and that opacity is the problem
Colophon demonstrates. Each region carries a compact **black-box contrast chip**
("closed API here: —") so the lesson lands at every view, not just once.

1. **Entropy heatmap (hero).** Prompt chars then continuation chars, each a
   focusable `<span>` tinted by `entropy` via the existing green→red gradient;
   continuation dimmed/underlined to distinguish typed from generated. Click a
   span to focus that position. *Contrast:* "A closed API returns text; it cannot
   tint one character by the model's confidence — you never see this."
2. **Context rail.** For the focused record, render `context_window` as K cells;
   `∅` pad cells greyed with a "horizon — the model can't see past here" label.
   The literal-context teaching centerpiece. *Contrast:* "You can point to the
   exact bytes the model used; a closed model's context is unverifiable."
3. **Context saliency.** Over the same K cells, an occlusion bar per character
   (`delta` ∝ height/intensity): "which remembered characters actually drove this
   prediction — measured, not guessed." Loaded on focus change from
   `/api/saliency`. *Contrast:* "This is real, per-character context attribution;
   no hosted API exposes it — glassboxllm had to *fake* it."
4. **Inspector.** Focused record's `top_k` as CSS bar rows (width ∝ prob), with a
   **"show full distribution"** expansion revealing all V next-char probabilities
   (maximal inspection — nothing curated away); call out `truth_rank`/`truth_prob`
   in words ("actual next char ranked #2, p=0.22 — right neighborhood" vs "not in
   top-5 — the model was surprised"); off-map badge when `off_map`. *Contrast:*
   "Closed APIs expose at most a truncated logprobs list, often nothing; you can't
   audit the alternatives the model rejected."
5. **Aggregates + scorecard.** JS-computed **median entropy**, top-3 **anchor**
   chars (lowest-entropy positions), **off-map count**; the existing scorecard
   table retained.

**Default focus is the lowest-confidence position** — `argmax(entropy)` over the
records — so the demo lands the user straight on the character where the model
struggled most, the teachable moment. Empty prompt → empty `records`; rail,
saliency, and inspector show a "type to begin" placeholder (mirrors the current
`analyze('')`).

## Error handling

- Unchanged server posture: no trained model → `/api/analyze` and `/api/saliency`
  return 503 with the existing message; `inspect_prompt`/`context_saliency`
  failure → 500 with `analysis failed: …`; the page and scorecard still serve.
- `/api/saliency` with a missing or out-of-range `pos` → 400.
- Empty/whitespace prompt is valid input, not an error.
- Off-map is a normal result, surfaced in the UI — never an error.

## Testing (TDD)

`test_colophon.py` — contract test for `inspect_prompt` on a tiny trained model:
- record count = `len(text) + n_continuation`; each record has the full field set;
- mean of record `entropy` (prompt positions only) equals
  `prompt_confidence(...)[0]` within tolerance (same-source guarantee);
- `top_k` sorted desc, length ≤ `topk`, probs in (0,1];
- `truth_rank`/`truth_prob` correct against a hand-computed softmax on one
  position; `off_map` positions have `null` truth fields;
- an off-map char yields `off_map:true` and appears via the endpoint's
  `unknown_chars`.

`test_colophon.py` — contract test for `context_saliency`:
- returns K entries; each `delta` in [0,1]; occluding a pad slot yields
  `delta ≈ 0`; occluding a real context char yields `delta ≥ 0`;
- deltas match a hand-computed total-variation distance on one occlusion.

`test_marginalia.py` — endpoint faithfulness:
- `analyze_prompt()` returns `records` matching `inspect_prompt` and the
  `unknown_chars`/`off_map` back-compat fields;
- `context_saliency()` wrapper matches `colophon.context_saliency`; `/api/saliency`
  clamps/validates `pos` (out-of-range → 400);
- off-map still fires on an out-of-distribution (e.g. Japanese) prompt.

## Constraints honored

- Pure stdlib server + single vanilla-JS page, **zero new dependencies**.
- No build step, no charting lib (bars are CSS).
- NumPy MLP math and `prompt_confidence()` outputs untouched.
- Honest counter-position preserved; off-map kept categorical and visible.
