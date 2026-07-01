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

Turn Marginalia from a single-number readout into a per-position context
inspector — the honest version of what glassboxllm mocked up — while keeping the
repo's hard constraint: **stdlib `http.server` + a single-page vanilla-JS
frontend, zero new dependencies.**

Non-goals: no build step, no charting library, no framework, no changes to the
NumPy MLP's math or to `prompt_confidence()`'s outputs, no reframing of the
project's honest counter-position.

## Approach (chosen: A — additive core)

Add one new function to `colophon.py` that returns per-position records; leave
`prompt_confidence()` byte-for-byte unchanged so the demo, `colophon.json`, and
existing tests do not move. Marginalia grows four views on the existing single
page. Marginalia re-derives no model signals — it renders and aggregates values
the new function returns.

Rejected: **B** (make `prompt_confidence` a wrapper) — touches a function the
demo/tests/`colophon.json` depend on for no user-visible gain. **C** (tabbed
multi-view UI) — more JS, drifts from the single-glanceable-page spirit.

## Architecture & data flow

```
browser (vanilla JS, one page)
  → GET /api/analyze?prompt=...        (debounced 200ms, per keystroke)
    → marginalia.analyze_prompt()      (thin HTTP-layer wrapper)
      → colophon.inspect_prompt()      (SOURCE OF TRUTH: per-position records)
  ← { records:[...], off_map, unknown_chars, ... }
  → JS computes aggregates from records, renders 4 views
GET /api/scorecard → colophon.scorecard_section()   (unchanged)
```

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

## Component: frontend (one page, four regions)

1. **Entropy heatmap (hero).** Prompt chars then continuation chars, each a
   focusable `<span>` tinted by `entropy` via the existing green→red gradient;
   continuation dimmed/underlined to distinguish typed from generated. Click a
   span to focus that position.
2. **Context rail.** For the focused record, render `context_window` as K cells;
   `∅` pad cells greyed with a "horizon — the model can't see past here" label.
   The literal-context teaching centerpiece; the honest analog of contextbuddy's
   context-pressure sentinel.
3. **Inspector.** Focused record's `top_k` as CSS bar rows (width ∝ prob); call
   out `truth_rank`/`truth_prob` in words ("actual next char ranked #2, p=0.22 —
   right neighborhood" vs "not in top-5 — the model was surprised"); off-map badge
   when `off_map`.
4. **Aggregates + scorecard.** JS-computed **median entropy**, top-3 **anchor**
   chars (lowest-entropy positions), **off-map count**; the existing scorecard
   table retained; a one-line honesty note: *"glassboxllm simulated these numbers;
   here they're read straight from the weights."*

Default focus is the last record. Empty prompt → empty `records`; rail and
inspector show a "type to begin" placeholder (mirrors the current `analyze('')`).

## Error handling

- Unchanged server posture: no trained model → `/api/analyze` returns 503 with the
  existing message; `inspect_prompt` failure → 500 with `analysis failed: …`; the
  page and scorecard still serve.
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

`test_marginalia.py` — endpoint faithfulness:
- `analyze_prompt()` returns `records` matching `inspect_prompt` and the
  `unknown_chars`/`off_map` back-compat fields;
- off-map still fires on an out-of-distribution (e.g. Japanese) prompt.

## Constraints honored

- Pure stdlib server + single vanilla-JS page, **zero new dependencies**.
- No build step, no charting lib (bars are CSS).
- NumPy MLP math and `prompt_confidence()` outputs untouched.
- Honest counter-position preserved; off-map kept categorical and visible.
