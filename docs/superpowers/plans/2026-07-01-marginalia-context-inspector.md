# Marginalia Context Inspector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn Marginalia from a single-number confidence readout into a maximal per-position white-box context inspector that makes a demo user *feel* why black-box LLM opacity is a problem.

**Architecture:** Add three additive, pure-NumPy functions to `colophon.py` (`inspect_prompt`, `context_saliency`, and small helpers) that expose per-position signals from the existing `forward()`; leave `prompt_confidence()` byte-for-byte unchanged. `marginalia.py` gains a rewritten `analyze_prompt()` + `/api/analyze` contract, a new `context_saliency()` wrapper + `/api/saliency` route, and a five-region single-page vanilla-JS frontend. Nothing re-derives model signals in JS — it renders and aggregates values the Python layer returns.

**Tech Stack:** Python 3, NumPy, stdlib `http.server`, `unittest`; single-file vanilla-JS/CSS frontend embedded as `INDEX_HTML`.

## Global Constraints

- Zero new dependencies: stdlib `http.server` server + single-page vanilla-JS frontend only. No build step, no charting library, no framework.
- Do not change the NumPy MLP's math or `prompt_confidence()`'s outputs/signature.
- All per-position numbers must use the same softmax/entropy math as `prompt_confidence()`/`generate()` — those functions stay the source of truth.
- `PAD = "\x00"` is vocab index 0; the initial context is `[0]*K`. Pad slots render as `∅` and are an honest teaching point, not a real character.
- Bounds already in `marginalia.py`: `MAX_PROMPT_LEN = 500`, `CONTINUATION_LEN = 80`. Reuse them; do not raise per-keystroke cost.
- Keep the project's honest counter-position: black-box contrast copy must be factual (e.g. "closed APIs expose at most a truncated logprobs list"), never "open beats closed on every axis".
- Run tests with `python -m unittest test_colophon test_marginalia`. Commit after every green step. Never push to `main`; open a PR.

---

### Task 1: `_display_char` readable-character helper (colophon.py)

**Files:**
- Modify: `colophon.py` (add near `prompt_confidence`, ~line 379)
- Test: `test_colophon.py` (add `DisplayChar` class)

**Interfaces:**
- Consumes: nothing.
- Produces: `_display_char(ch: str) -> str` — maps `"\x00"→"∅"`, `" "→"␣"`, `"\n"→"⏎"`, `"\t"→"⇥"`, else returns `ch` unchanged. Used by Tasks 2 and 3.

- [ ] **Step 1: Write the failing test**

```python
class DisplayChar(unittest.TestCase):
    """Readable glyphs for the inspection UI; every other character is itself."""
    def test_control_and_whitespace_glyphs(self):
        self.assertEqual(C._display_char("\x00"), "∅")
        self.assertEqual(C._display_char(" "), "␣")
        self.assertEqual(C._display_char("\n"), "⏎")
        self.assertEqual(C._display_char("\t"), "⇥")

    def test_ordinary_char_passes_through(self):
        self.assertEqual(C._display_char("a"), "a")
        self.assertEqual(C._display_char(":"), ":")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest test_colophon.DisplayChar -v`
Expected: FAIL with `AttributeError: module 'colophon' has no attribute '_display_char'`

- [ ] **Step 3: Write minimal implementation**

Add to `colophon.py` just above `prompt_confidence`:

```python
# --------------------------------------------------------------------------- #
# White-box inspection helpers (used by marginalia.py; see inspect_prompt).
# --------------------------------------------------------------------------- #

_DISPLAY = {"\x00": "∅", " ": "␣", "\n": "⏎", "\t": "⇥"}


def _display_char(ch):
    """Human-readable form of a single character for the inspection UI.
    PAD/whitespace get glyphs; everything else is shown as itself."""
    return _DISPLAY.get(ch, ch)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest test_colophon.DisplayChar -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add colophon.py test_colophon.py
git commit -m "feat: add _display_char helper for white-box inspection"
```

---

### Task 2: `inspect_prompt` per-position records (colophon.py)

**Files:**
- Modify: `colophon.py` (add after `_display_char`)
- Test: `test_colophon.py` (add `InspectPrompt` class)

**Interfaces:**
- Consumes: `forward`, `generate`, `_display_char`, `PAD`.
- Produces:
  - `_dist(p, ctx_ids) -> np.ndarray` — normalized next-char probs for one K-length int context (same softmax as `prompt_confidence`).
  - `_full_context_ids(p, stoi, itos, K, text, n_continuation, seed) -> (ids: list[int], n_prompt: int, cont: str)` — `[0]*K` + prompt ids (`stoi.get(ch,0)`) + ids of `generate()`'s continuation.
  - `inspect_prompt(p, stoi, itos, K, text, topk=5, n_continuation=80, seed=0) -> list[dict]` — one record per char of `text` then per continuation char. Record keys: `char, display, is_continuation, entropy, top_k (list[[str,float]]), context_window (list[str]), truth_rank (int|None), truth_prob (float|None), off_map (bool)`. Used by Tasks 3, 4.

- [ ] **Step 1: Write the failing test**

```python
class InspectPrompt(unittest.TestCase):
    """Per-position white-box records: entropy must equal prompt_confidence's,
    top_k must be a real sorted distribution, truth-rank must be correct, and
    off-map chars must carry null truth fields."""

    def setUp(self):
        self.text, _ = C.load_corpus(C.DEFAULT_SRC)
        self.chars, self.stoi, self.itos = C.build_vocab(self.text)
        self.p = C.init_params(len(self.chars), 8, 4, 16, seed=0)
        self.K = 4

    def test_record_count_and_shape(self):
        recs = C.inspect_prompt(self.p, self.stoi, self.itos, self.K,
                                "weights", topk=5, n_continuation=6, seed=0)
        self.assertEqual(len(recs), len("weights") + 6)
        r = recs[0]
        for key in ("char", "display", "is_continuation", "entropy", "top_k",
                    "context_window", "truth_rank", "truth_prob", "off_map"):
            self.assertIn(key, r)
        self.assertEqual(len(r["context_window"]), self.K)
        self.assertTrue(all(recs[i]["is_continuation"] is False
                            for i in range(len("weights"))))
        self.assertTrue(recs[-1]["is_continuation"] is True)

    def test_prompt_entropy_mean_matches_prompt_confidence(self):
        text = self.text[:30]
        recs = C.inspect_prompt(self.p, self.stoi, self.itos, self.K, text,
                                n_continuation=0, seed=0)
        mean_ent = sum(r["entropy"] for r in recs) / len(recs)
        ent, _ = C.prompt_confidence(self.p, self.stoi, self.K, text)
        self.assertAlmostEqual(mean_ent, ent, places=9)

    def test_top_k_sorted_and_normalized(self):
        recs = C.inspect_prompt(self.p, self.stoi, self.itos, self.K, "class",
                                topk=5, n_continuation=0)
        top = recs[0]["top_k"]
        self.assertLessEqual(len(top), 5)
        probs = [pr for _, pr in top]
        self.assertEqual(probs, sorted(probs, reverse=True))
        self.assertTrue(all(0.0 < pr <= 1.0 for pr in probs))

    def test_truth_rank_matches_hand_computed(self):
        # Rank the actual next char in position 0's full distribution by hand.
        text = "class"
        recs = C.inspect_prompt(self.p, self.stoi, self.itos, self.K, text,
                                n_continuation=0)
        ctx = np.array([[0, 0, 0, 0]])
        logits, _ = C.forward(self.p, ctx)
        l = logits[0] - logits[0].max()
        pr = np.exp(l); pr /= pr.sum()
        cid = self.stoi[text[0]]
        expected_rank = int((pr > pr[cid]).sum()) + 1
        self.assertEqual(recs[0]["truth_rank"], expected_rank)
        self.assertAlmostEqual(recs[0]["truth_prob"], float(pr[cid]), places=9)

    def test_off_map_char_has_null_truth(self):
        recs = C.inspect_prompt(self.p, self.stoi, self.itos, self.K, "日x",
                                n_continuation=0)
        self.assertTrue(recs[0]["off_map"])
        self.assertIsNone(recs[0]["truth_rank"])
        self.assertIsNone(recs[0]["truth_prob"])
        self.assertFalse(recs[1]["off_map"])  # "x" is ASCII, in vocab

    def test_pad_slots_shown_as_glyph(self):
        recs = C.inspect_prompt(self.p, self.stoi, self.itos, self.K, "ab",
                                n_continuation=0)
        # Position 0 sees an all-pad context.
        self.assertEqual(recs[0]["context_window"], ["∅", "∅", "∅", "∅"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest test_colophon.InspectPrompt -v`
Expected: FAIL with `AttributeError: module 'colophon' has no attribute 'inspect_prompt'`

- [ ] **Step 3: Write minimal implementation**

Add to `colophon.py` immediately after `_display_char`:

```python
def _dist(p, ctx_ids):
    """Raw (temperature-1) next-char probability distribution for a single
    K-length int context -- the same softmax prompt_confidence()/generate() use."""
    logits, _ = forward(p, np.array([ctx_ids]))
    l = logits[0]; l = l - l.max()
    pr = np.exp(l); pr /= pr.sum()
    return pr


def _full_context_ids(p, stoi, itos, K, text, n_continuation, seed):
    """The full int sequence the inspector reasons over: K pad slots, then the
    teacher-forced prompt, then the model's OWN sampled continuation. The
    continuation comes from generate() verbatim, so it is identical to what
    generate() emits -- Marginalia re-derives nothing. Unknown chars map to PAD
    (index 0), exactly as generate()/prompt_confidence() already treat them.
    Returns (ids, n_prompt, cont_chars)."""
    prompt_ids = [stoi.get(ch, 0) for ch in text]
    cont = ""
    if n_continuation > 0:
        cont = generate(p, stoi, itos, K, prompt=text,
                        n=n_continuation, seed=seed)[len(text):]
    ids = [0] * K + prompt_ids + [stoi.get(ch, 0) for ch in cont]
    return ids, len(prompt_ids), cont


def inspect_prompt(p, stoi, itos, K, text, topk=5, n_continuation=80, seed=0):
    """Maximal per-position white-box record over the teacher-forced prompt plus
    the model's sampled continuation. Every number here is read from the weights
    via forward() -- the honest version of what black-box tools can only fake.

    Each record: the actual next char (+ readable display), whether it is the
    typed prompt or the model's own continuation, normalized next-char entropy
    (identical to prompt_confidence's), the top-k next-char distribution, the
    literal K-char context window the model saw (∅ = pad), and where the actual
    next char ranked (truth_rank/truth_prob; null when off-map)."""
    ids, n_prompt, cont = _full_context_ids(p, stoi, itos, K, text,
                                            n_continuation, seed)
    chars = text + cont
    records = []
    for i, ch in enumerate(chars):
        ctx = ids[i:i + K]
        pr = _dist(p, ctx)
        ent = float(-(pr * np.log(pr + 1e-12)).sum() / np.log(len(pr)))
        order = np.argsort(pr)[::-1]
        top = [[_display_char(itos[int(j)]), float(pr[int(j)])]
               for j in order[:topk]]
        if ch in stoi:
            cid = stoi[ch]
            truth_rank = int((pr > pr[cid]).sum()) + 1
            truth_prob = float(pr[cid])
            off = False
        else:
            truth_rank = truth_prob = None
            off = True
        records.append({
            "char": ch,
            "display": _display_char(ch),
            "is_continuation": i >= n_prompt,
            "entropy": ent,
            "top_k": top,
            "context_window": [_display_char(itos[int(c)]) for c in ctx],
            "truth_rank": truth_rank,
            "truth_prob": truth_prob,
            "off_map": off,
        })
    return records
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest test_colophon.InspectPrompt -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add colophon.py test_colophon.py
git commit -m "feat: add inspect_prompt per-position white-box records"
```

---

### Task 3: `context_saliency` occlusion attribution (colophon.py)

**Files:**
- Modify: `colophon.py` (add after `inspect_prompt`)
- Test: `test_colophon.py` (add `ContextSaliency` class)

**Interfaces:**
- Consumes: `_dist`, `_full_context_ids`, `_display_char`.
- Produces: `context_saliency(p, stoi, itos, K, text, pos, n_continuation=80, seed=0) -> dict` with `{"pos": int, "window": [{"char", "display", "delta", "is_pad"} * K]}`. `delta` = total-variation distance in [0,1] between the baseline next-char distribution at `pos` and the distribution with context slot `j` occluded (replaced by PAD). Raises `IndexError` if `pos` is out of range. Used by Task 5.

- [ ] **Step 1: Write the failing test**

```python
class ContextSaliency(unittest.TestCase):
    """Occlusion attribution over the K-char window: real, model-derived
    'which remembered character mattered', not a simulated attention weight."""

    def setUp(self):
        self.text, _ = C.load_corpus(C.DEFAULT_SRC)
        self.chars, self.stoi, self.itos = C.build_vocab(self.text)
        self.p = C.init_params(len(self.chars), 8, 4, 16, seed=0)
        self.K = 4

    def test_window_shape_and_delta_range(self):
        out = C.context_saliency(self.p, self.stoi, self.itos, self.K,
                                 "weights", pos=6, n_continuation=0)
        self.assertEqual(out["pos"], 6)
        self.assertEqual(len(out["window"]), self.K)
        for cell in out["window"]:
            self.assertTrue(0.0 <= cell["delta"] <= 1.0)
            for key in ("char", "display", "delta", "is_pad"):
                self.assertIn(key, cell)

    def test_pad_slot_has_zero_delta(self):
        # Position 0 sees an all-pad context; occluding PAD with PAD changes nothing.
        out = C.context_saliency(self.p, self.stoi, self.itos, self.K,
                                 "weights", pos=0, n_continuation=0)
        for cell in out["window"]:
            self.assertTrue(cell["is_pad"])
            self.assertAlmostEqual(cell["delta"], 0.0, places=12)

    def test_delta_matches_hand_computed_tv_distance(self):
        text = "weights"
        pos = 6
        out = C.context_saliency(self.p, self.stoi, self.itos, self.K, text,
                                 pos=pos, n_continuation=0)
        ids = [0] * self.K + [self.stoi.get(c, 0) for c in text]
        ctx = ids[pos:pos + self.K]
        base = C._dist(self.p, ctx)
        occ = list(ctx); occ[0] = 0
        expected = float(0.5 * np.abs(base - C._dist(self.p, occ)).sum())
        self.assertAlmostEqual(out["window"][0]["delta"], expected, places=12)

    def test_out_of_range_pos_raises(self):
        with self.assertRaises(IndexError):
            C.context_saliency(self.p, self.stoi, self.itos, self.K, "hi",
                               pos=99, n_continuation=0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest test_colophon.ContextSaliency -v`
Expected: FAIL with `AttributeError: module 'colophon' has no attribute 'context_saliency'`

- [ ] **Step 3: Write minimal implementation**

Add to `colophon.py` immediately after `inspect_prompt`:

```python
def context_saliency(p, stoi, itos, K, text, pos, n_continuation=80, seed=0):
    """For the prediction at position `pos`, measure how much each of the K
    remembered characters actually shaped it, by occluding each slot (replace
    with PAD) and taking the total-variation distance between the baseline and
    occluded next-char distributions. Pure NumPy over forward() -- the honest
    analog of the attention weights glassboxllm had to simulate."""
    ids, n_prompt, cont = _full_context_ids(p, stoi, itos, K, text,
                                            n_continuation, seed)
    n = n_prompt + len(cont)
    if not (0 <= pos < n):
        raise IndexError(f"pos {pos} out of range [0, {n})")
    ctx = ids[pos:pos + K]
    base = _dist(p, ctx)
    window = []
    for j in range(K):
        occ = list(ctx); occ[j] = 0
        delta = float(0.5 * np.abs(base - _dist(p, occ)).sum())
        cid = int(ctx[j])
        window.append({
            "char": itos[cid],
            "display": _display_char(itos[cid]),
            "delta": delta,
            "is_pad": cid == 0,
        })
    return {"pos": pos, "window": window}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest test_colophon.ContextSaliency -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add colophon.py test_colophon.py
git commit -m "feat: add context_saliency occlusion attribution"
```

---

### Task 4: Rewrite `analyze_prompt` + `/api/analyze` contract (marginalia.py)

**Files:**
- Modify: `marginalia.py` (`analyze_prompt`, ~lines 56-68; `/api/analyze` handler, ~lines 243-258)
- Test: `test_marginalia.py` (replace `AnalyzePrompt` body; update `HandlerRouting.test_analyze_route`)

**Interfaces:**
- Consumes: `colophon.inspect_prompt`.
- Produces: `analyze_prompt(p, stoi, itos, K, prompt, n=CONTINUATION_LEN, seed=0) -> {"prompt": str, "records": list, "unknown_chars": list[str], "off_map": bool}`. Empty/whitespace-free empty prompt yields `records == []` (no continuation from nothing). Used by the frontend (Task 6).

- [ ] **Step 1: Write the failing test**

Replace the entire `class AnalyzePrompt(...)` block in `test_marginalia.py` with:

```python
class AnalyzePrompt(unittest.TestCase):
    def setUp(self):
        text, _ = C.load_corpus(C.DEFAULT_SRC)
        chars, stoi, itos = C.build_vocab(text)
        self.stoi, self.itos = stoi, itos
        self.p = C.init_params(len(chars), 8, 4, 16, seed=0)
        self.K = 4
        self.native = text[:50]

    def test_records_match_inspect_prompt(self):
        result = M.analyze_prompt(self.p, self.stoi, self.itos, self.K,
                                  self.native, n=10)
        recs = C.inspect_prompt(self.p, self.stoi, self.itos, self.K,
                                self.native, n_continuation=10)
        self.assertEqual(result["records"], recs)
        self.assertEqual(result["prompt"], self.native)
        self.assertFalse(result["off_map"])
        self.assertEqual(result["unknown_chars"], [])

    def test_prompt_entropy_still_matches_prompt_confidence(self):
        result = M.analyze_prompt(self.p, self.stoi, self.itos, self.K,
                                  self.native, n=0)
        mean_ent = sum(r["entropy"] for r in result["records"]) / len(result["records"])
        ent, _ = C.prompt_confidence(self.p, self.stoi, self.K, self.native)
        self.assertAlmostEqual(mean_ent, ent, places=9)

    def test_off_map_true_for_unseen_chars(self):
        result = M.analyze_prompt(self.p, self.stoi, self.itos, self.K, "日本語", n=5)
        self.assertTrue(result["off_map"])
        self.assertEqual(result["unknown_chars"], sorted(set("日本語")))

    def test_empty_prompt_yields_no_records(self):
        result = M.analyze_prompt(self.p, self.stoi, self.itos, self.K, "", n=5)
        self.assertEqual(result["records"], [])
        self.assertEqual(result["unknown_chars"], [])
        self.assertFalse(result["off_map"])
```

Also update `HandlerRouting.test_analyze_route` in `test_marginalia.py` to the new contract:

```python
    def test_analyze_route(self):
        status, headers, body = self.server.get("/api/analyze?prompt=weights")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
        data = json.loads(body)
        self.assertEqual(data["prompt"], "weights")
        self.assertIn("records", data)
        self.assertIn("off_map", data)
        self.assertGreaterEqual(len(data["records"]), len("weights"))
        self.assertIn("entropy", data["records"][0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest test_marginalia.AnalyzePrompt -v`
Expected: FAIL — `analyze_prompt` still returns the old `entropy`/`continuation` shape (KeyError on `records` / assertion failure).

- [ ] **Step 3: Write minimal implementation**

Replace `analyze_prompt` in `marginalia.py` with:

```python
def analyze_prompt(p, stoi, itos, K, prompt, n=CONTINUATION_LEN, seed=0):
    """The per-keystroke call: maximal per-position white-box records from
    colophon.inspect_prompt(), plus the categorical off-map signal. Both come
    from colophon.py's own functions -- nothing is re-derived here. An empty
    prompt yields no records (we do not dream a continuation from nothing)."""
    n_eff = n if prompt else 0
    records = colophon.inspect_prompt(p, stoi, itos, K, prompt,
                                      n_continuation=n_eff, seed=seed)
    unknown = sorted({ch for ch in prompt if ch not in stoi})
    return {
        "prompt": prompt,
        "records": records,
        "unknown_chars": unknown,
        "off_map": bool(unknown),
    }
```

The `/api/analyze` handler body (in `make_handler`) is unchanged — it already calls `analyze_prompt(p, stoi, itos, K, prompt)` and JSON-encodes the result. Confirm it still reads:

```python
            elif parsed.path == "/api/analyze":
                if model is None:
                    self._send_json(
                        {"error": "no trained model found -- run `python colophon.py demo` first"},
                        status=503)
                    return
                qs = urllib.parse.parse_qs(parsed.query)
                prompt = qs.get("prompt", [""])[0][:MAX_PROMPT_LEN]
                p, stoi, itos, K = model
                try:
                    result = analyze_prompt(p, stoi, itos, K, prompt)
                except Exception as e:
                    self._send_json(
                        {"error": f"analysis failed: {e}"}, status=500)
                    return
                self._send_json(result)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest test_marginalia.AnalyzePrompt test_marginalia.HandlerRouting -v`
Expected: PASS (all AnalyzePrompt + HandlerRouting tests)

- [ ] **Step 5: Commit**

```bash
git add marginalia.py test_marginalia.py
git commit -m "feat: analyze_prompt returns per-position records"
```

---

### Task 5: `context_saliency` wrapper + `/api/saliency` route (marginalia.py)

**Files:**
- Modify: `marginalia.py` (add `context_saliency` wrapper near `analyze_prompt`; add `/api/saliency` branch in `do_GET`)
- Test: `test_marginalia.py` (add `SaliencyWrapper` class; add routing tests to `HandlerRouting`)

**Interfaces:**
- Consumes: `colophon.context_saliency`.
- Produces:
  - `context_saliency(p, stoi, itos, K, prompt, pos, n=CONTINUATION_LEN, seed=0) -> dict` — thin wrapper over `colophon.context_saliency`.
  - `GET /api/saliency?prompt=…&pos=N` → 200 JSON `{pos, window}`; 400 on missing/non-integer/out-of-range `pos`; 503 when no model; 500 on unexpected failure.

- [ ] **Step 1: Write the failing test**

Add to `test_marginalia.py`:

```python
class SaliencyWrapper(unittest.TestCase):
    def setUp(self):
        text, _ = C.load_corpus(C.DEFAULT_SRC)
        chars, stoi, itos = C.build_vocab(text)
        self.stoi, self.itos = stoi, itos
        self.p = C.init_params(len(chars), 8, 4, 16, seed=0)
        self.K = 4

    def test_wrapper_matches_colophon(self):
        got = M.context_saliency(self.p, self.stoi, self.itos, self.K,
                                 "weights", pos=6, n=0)
        want = C.context_saliency(self.p, self.stoi, self.itos, self.K,
                                  "weights", pos=6, n_continuation=0)
        self.assertEqual(got, want)
```

Add these methods to `class HandlerRouting`:

```python
    def test_saliency_route_ok(self):
        status, headers, body = self.server.get("/api/saliency?prompt=weights&pos=3")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
        data = json.loads(body)
        self.assertEqual(data["pos"], 3)
        self.assertEqual(len(data["window"]), 4)  # K == 4 in the fixture

    def test_saliency_bad_pos_400(self):
        status, _, body = self.server.get("/api/saliency?prompt=hi&pos=nope")
        self.assertEqual(status, 400)
        self.assertIn("error", json.loads(body))

    def test_saliency_out_of_range_pos_400(self):
        status, _, body = self.server.get("/api/saliency?prompt=hi&pos=999")
        self.assertEqual(status, 400)
        self.assertIn("error", json.loads(body))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest test_marginalia.SaliencyWrapper test_marginalia.HandlerRouting -v`
Expected: FAIL — `M.context_saliency` missing; `/api/saliency` returns 404.

- [ ] **Step 3: Write minimal implementation**

Add the wrapper in `marginalia.py` immediately after `analyze_prompt`:

```python
def context_saliency(p, stoi, itos, K, prompt, pos, n=CONTINUATION_LEN, seed=0):
    """Thin wrapper over colophon.context_saliency for the /api/saliency route.
    Called only when the focused position changes -- never per keystroke."""
    return colophon.context_saliency(p, stoi, itos, K, prompt, pos,
                                     n_continuation=n, seed=seed)
```

Add this branch to `do_GET`, immediately after the `/api/analyze` branch and before the final `else`:

```python
            elif parsed.path == "/api/saliency":
                if model is None:
                    self._send_json(
                        {"error": "no trained model found -- run `python colophon.py demo` first"},
                        status=503)
                    return
                qs = urllib.parse.parse_qs(parsed.query)
                prompt = qs.get("prompt", [""])[0][:MAX_PROMPT_LEN]
                try:
                    pos = int(qs.get("pos", [""])[0])
                except (TypeError, ValueError):
                    self._send_json({"error": "pos must be an integer"}, status=400)
                    return
                p, stoi, itos, K = model
                try:
                    result = context_saliency(p, stoi, itos, K, prompt, pos)
                except IndexError as e:
                    self._send_json({"error": str(e)}, status=400)
                    return
                except Exception as e:
                    self._send_json({"error": f"analysis failed: {e}"}, status=500)
                    return
                self._send_json(result)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest test_marginalia.SaliencyWrapper test_marginalia.HandlerRouting -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add marginalia.py test_marginalia.py
git commit -m "feat: add /api/saliency route and context_saliency wrapper"
```

---

### Task 6: Five-region inspector frontend (marginalia.py `INDEX_HTML`)

**Files:**
- Modify: `marginalia.py` (`INDEX_HTML`, lines 71-217)
- Test: `test_marginalia.py` (add `IndexHtmlContract` class); manual browser verification

**Interfaces:**
- Consumes: `/api/analyze` (`records`, `off_map`, `unknown_chars`), `/api/saliency` (`window`), `/api/scorecard`.
- Produces: a single self-contained HTML page with element ids `heatmap`, `rail`, `saliency`, `inspector`, `aggregates`, `scorecard`, `bb-banner` (black-box framing).

> Note on testing: the repo has no JS test harness, so the JS logic is verified by (a) a Python contract test asserting the page ships the required region ids/markers and (b) manual browser verification (CLAUDE.md: open HTML and confirm it renders). Do not add a JS test framework — that would violate the zero-dependency constraint.

- [ ] **Step 1: Write the failing test**

Add to `test_marginalia.py`:

```python
class IndexHtmlContract(unittest.TestCase):
    """The single-page inspector must ship all five regions + the black-box
    framing banner, and must not smuggle in an external dependency."""

    def test_regions_present(self):
        html = M.INDEX_HTML
        for marker in ('id="heatmap"', 'id="rail"', 'id="saliency"',
                       'id="inspector"', 'id="aggregates"', 'id="scorecard"',
                       'id="bb-banner"'):
            self.assertIn(marker, html)

    def test_calls_both_apis(self):
        html = M.INDEX_HTML
        self.assertIn("/api/analyze", html)
        self.assertIn("/api/saliency", html)

    def test_no_external_dependencies(self):
        html = M.INDEX_HTML
        self.assertNotIn("http://", html.replace("http://127.0.0.1", ""))
        self.assertNotIn("https://", html)
        self.assertNotIn("cdn", html.lower())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest test_marginalia.IndexHtmlContract -v`
Expected: FAIL (`id="heatmap"` etc. not found in the current page).

- [ ] **Step 3: Write minimal implementation**

Replace the entire `INDEX_HTML = """..."""` assignment in `marginalia.py` with:

```python
INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Marginalia -- live white-box inspection for Colophon</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: ui-monospace, Menlo, Consolas, monospace; max-width: 900px;
         margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }
  h1 { font-size: 1.3rem; } h1 .glyph { opacity: .6; }
  textarea { width: 100%; min-height: 4rem; font: inherit; padding: .5rem;
             box-sizing: border-box; }
  .panel { border: 1px solid #8884; border-radius: 6px; padding: .75rem 1rem;
           margin: 1rem 0; }
  .muted { opacity: .6; font-size: .85rem; }
  .bb { color: #b60; font-size: .8rem; margin-top: .4rem; }
  #bb-banner { border-left: 3px solid #b60; padding: .5rem .75rem; background: #b6010a10;
               font-size: .85rem; }
  #heatmap { font-size: 1.1rem; line-height: 1.9; word-break: break-all; }
  #heatmap span { padding: 0 1px; border-radius: 2px; cursor: pointer; }
  #heatmap span.cont { text-decoration: underline dotted; opacity: .85; }
  #heatmap span.focus { outline: 2px solid #06f; }
  #heatmap span.off { outline: 2px solid #d33; }
  .cell { display: inline-block; min-width: 1.4rem; text-align: center;
          border: 1px solid #8884; border-radius: 3px; margin: 1px; padding: .1rem .2rem; }
  .cell.pad { opacity: .4; }
  .salbar { height: 6px; background: #06f; border-radius: 3px; margin-top: 2px; }
  .barrow { display: flex; align-items: center; gap: .5rem; margin: 2px 0; }
  .barrow .lab { min-width: 2rem; text-align: right; }
  .barrow .bar { height: 12px; background: linear-gradient(90deg,#2a6,#6cf); border-radius: 3px; }
  table { border-collapse: collapse; width: 100%; font-size: .85rem; }
  th, td { border: 1px solid #8884; padding: .3rem .5rem; text-align: left; }
  td.open { color: #2a6; } td.partial { color: #c90; } td.closed { color: #d33; }
  .error { color: #d33; }
</style>
</head>
<body>
<h1><span class="glyph">&#10087;</span> Marginalia</h1>
<div id="bb-banner">Every number below is read straight from <code>colophon.npz</code> --
the model's own weights. A hosted LLM (GPT, Gemini, Claude via API) hides all of it;
that opacity is the problem Colophon exists to demonstrate. glassboxllm had to
<em>simulate</em> these signals -- here they are ground truth.</div>

<textarea id="prompt" placeholder="e.g. weights_basemodel:&#10;    class:   or   &#26085;&#26412;&#35486;&#12391;&#26360;&#12367;" autofocus></textarea>

<div class="panel">
  <div class="muted">confidence heatmap -- each character tinted by the model's own
  next-char entropy (green = certain, red = no idea). Click any character to inspect it.</div>
  <div id="heatmap">&nbsp;</div>
  <div class="bb">Closed API here: returns text only -- it cannot tint a single character by confidence. You never see this.</div>
</div>

<div class="panel">
  <div class="muted">context window -- the literal last-K characters the model saw when
  predicting the focused character (<span id="focus-label">--</span>). <span class="pad">Greyed</span> = pad; the model cannot see past the horizon.</div>
  <div id="rail">&nbsp;</div>
  <div class="bb">Closed API here: you cannot verify which bytes were actually in context.</div>
</div>

<div class="panel">
  <div class="muted">context saliency -- occluding each remembered character and
  measuring how far the prediction moves (real per-character attribution).</div>
  <div id="saliency">&nbsp;</div>
  <div class="bb">Closed API here: no attribution at all -- glassboxllm had to fake this as "attention".</div>
</div>

<div class="panel">
  <div class="muted">inspector -- what the model predicted for the focused position.</div>
  <div id="inspector">&nbsp;</div>
  <div class="bb">Closed API here: at most a truncated logprobs list, often nothing. You can't audit the rejected alternatives.</div>
</div>

<div class="panel" id="aggregates"><div class="muted">session signals</div></div>

<div class="panel" id="scorecard-panel">
  <div class="muted">OSAI openness scorecard</div>
  <table id="scorecard"><tbody></tbody></table>
</div>

<div id="error" class="error"></div>

<script>
const $ = id => document.getElementById(id);
const promptEl = $('prompt'), errorEl = $('error');
let records = [], focus = 0, timer = null;

function entColor(e) {  // 0 (green) -> 1 (red)
  const h = Math.round(140 * (1 - Math.max(0, Math.min(1, e))));
  return `hsl(${h} 60% 50% / .35)`;
}

function renderHeatmap() {
  const hm = $('heatmap'); hm.innerHTML = '';
  if (!records.length) { hm.textContent = 'type to begin'; return; }
  records.forEach((r, i) => {
    const s = document.createElement('span');
    s.textContent = r.display;
    s.style.background = entColor(r.entropy);
    if (r.is_continuation) s.classList.add('cont');
    if (r.off_map) s.classList.add('off');
    if (i === focus) s.classList.add('focus');
    s.title = `entropy ${r.entropy.toFixed(3)}` +
      (r.truth_rank ? `, rank #${r.truth_rank}, p=${r.truth_prob.toFixed(3)}` : ', off-map');
    s.onclick = () => { focus = i; renderFocus(); fetchSaliency(); };
    hm.appendChild(s);
  });
}

function renderFocus() {
  document.querySelectorAll('#heatmap span').forEach((s, i) =>
    s.classList.toggle('focus', i === focus));
  const r = records[focus];
  $('focus-label').textContent = r ? `'${r.display}'` : '--';

  const rail = $('rail'); rail.innerHTML = '';
  if (r) r.context_window.forEach(c => {
    const d = document.createElement('span');
    d.className = 'cell' + (c === '∅' ? ' pad' : '');
    d.textContent = c; rail.appendChild(d);
  });

  const ins = $('inspector'); ins.innerHTML = '';
  if (!r) { ins.textContent = 'type to begin'; return; }
  const truth = document.createElement('div');
  truth.textContent = r.off_map
    ? 'you typed a character the model has never seen -- it has no representation for it.'
    : (r.truth_rank <= 5
        ? `actual next char '${r.display}' ranked #${r.truth_rank}, p=${r.truth_prob.toFixed(3)} -- the right neighborhood.`
        : `actual next char '${r.display}' ranked #${r.truth_rank} (p=${r.truth_prob.toFixed(3)}) -- the model was surprised.`);
  ins.appendChild(truth);
  const max = Math.max(...r.top_k.map(t => t[1]), 1e-9);
  r.top_k.forEach(([ch, pr]) => {
    const row = document.createElement('div'); row.className = 'barrow';
    const lab = document.createElement('span'); lab.className = 'lab'; lab.textContent = ch;
    const bar = document.createElement('span'); bar.className = 'bar';
    bar.style.width = (100 * pr / max) + 'px';
    const val = document.createElement('span'); val.className = 'muted'; val.textContent = pr.toFixed(3);
    row.append(lab, bar, val); ins.appendChild(row);
  });
  const note = document.createElement('div'); note.className = 'muted';
  note.textContent = `top ${r.top_k.length} of the full distribution (hover the heatmap for any position).`;
  ins.appendChild(note);
}

function renderAggregates() {
  const el = $('aggregates'); el.innerHTML = '<div class="muted">session signals</div>';
  if (!records.length) return;
  const ents = records.map(r => r.entropy).slice().sort((a, b) => a - b);
  const median = ents[Math.floor(ents.length / 2)];
  const offCount = records.filter(r => r.off_map).length;
  const anchors = records.slice().sort((a, b) => a.entropy - b.entropy)
    .slice(0, 3).map(r => r.display).join(' ');
  const div = document.createElement('div');
  div.innerHTML = `median entropy <b>${median.toFixed(3)}</b> &middot; ` +
    `most-confident chars (anchors): <b>${anchors}</b> &middot; ` +
    `off-map characters: <b>${offCount}</b>`;
  el.appendChild(div);
}

function renderSaliency(data) {
  const el = $('saliency'); el.innerHTML = '';
  const max = Math.max(...data.window.map(c => c.delta), 1e-9);
  data.window.forEach(c => {
    const wrap = document.createElement('span'); wrap.className = 'cell' + (c.is_pad ? ' pad' : '');
    wrap.textContent = c.display;
    const bar = document.createElement('div'); bar.className = 'salbar';
    bar.style.width = Math.round(100 * c.delta / max) + '%';
    bar.title = `delta ${c.delta.toFixed(3)}`;
    wrap.appendChild(bar); el.appendChild(wrap);
  });
}

async function fetchSaliency() {
  if (!records.length) { $('saliency').textContent = ' '; return; }
  try {
    const res = await fetch('/api/saliency?pos=' + focus +
      '&prompt=' + encodeURIComponent(promptEl.value));
    if (res.ok) renderSaliency(await res.json());
  } catch (e) { /* saliency is best-effort; heatmap already rendered */ }
}

async function analyze(prompt) {
  try {
    const res = await fetch('/api/analyze?prompt=' + encodeURIComponent(prompt));
    const data = await res.json();
    if (!res.ok) { errorEl.textContent = data.error || ('error ' + res.status); return; }
    errorEl.textContent = '';
    records = data.records;
    // Default focus: the lowest-confidence (highest-entropy) position.
    focus = records.reduce((best, r, i, a) => r.entropy > a[best].entropy ? i : best, 0);
    renderHeatmap(); renderFocus(); renderAggregates(); fetchSaliency();
  } catch (e) { errorEl.textContent = String(e); }
}

promptEl.addEventListener('input', () => {
  clearTimeout(timer);
  timer = setTimeout(() => analyze(promptEl.value), 200);
});

function renderScorecard(sc) {
  const tbody = document.querySelector('#scorecard tbody'); tbody.innerHTML = '';
  const header = document.createElement('tr');
  ['dimension', 'colophon', 'typical closed', 'note'].forEach(h => {
    const th = document.createElement('th'); th.textContent = h; header.appendChild(th);
  });
  tbody.appendChild(header);
  sc.dimensions.forEach(d => {
    const tr = document.createElement('tr');
    [d.dimension, d.colophon, d.typical_closed, d.note].forEach((val, i) => {
      const td = document.createElement('td'); td.textContent = val;
      if (i === 1) td.className = d.colophon;
      if (i === 2) td.className = d.typical_closed;
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  const caption = document.createElement('div'); caption.className = 'muted';
  caption.textContent = `${sc.colophon_open}/${sc.scored} open (this artifact) vs ` +
    `${sc.typical_closed_open}/${sc.scored} open (typical closed model)`;
  $('scorecard-panel').appendChild(caption);
}

fetch('/api/scorecard').then(r => r.json()).then(renderScorecard)
  .catch(e => { errorEl.textContent = String(e); });

analyze('');
</script>
</body>
</html>
"""
```

- [ ] **Step 4: Run the contract test to verify it passes**

Run: `python -m unittest test_marginalia.IndexHtmlContract -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Manual browser verification (CLAUDE.md: confirm HTML renders)**

```bash
python colophon.py demo          # writes colophon.npz
python marginalia.py &           # serves at http://127.0.0.1:8765
sleep 1
curl -s "http://127.0.0.1:8765/api/analyze?prompt=weights_basemodel:" | python -m json.tool | head -40
curl -s "http://127.0.0.1:8765/api/saliency?prompt=weights_basemodel:&pos=5" | python -m json.tool
```

Then open `http://127.0.0.1:8765` in a browser and confirm:
- typing `weights_basemodel:` colors the heatmap and auto-focuses the reddest (least-confident) char;
- clicking a character updates the context rail, saliency bars, and inspector;
- typing `日本語` shows red off-map outlines and the "never seen" inspector message;
- the scorecard still renders.
Stop the server: `kill %1`.

- [ ] **Step 6: Commit**

```bash
git add marginalia.py test_marginalia.py
git commit -m "feat: five-region white-box context inspector frontend"
```

---

### Task 7: Documentation + full-suite gate

**Files:**
- Modify: `CLAUDE.md` (File map `marginalia.py` bullet; the `Done: Marginalia` line)
- Modify: `README.md` (Marginalia description)
- Test: full suite

**Interfaces:**
- Consumes: everything above. Produces: updated docs; no code.

- [ ] **Step 1: Update `CLAUDE.md`**

Replace the `marginalia.py` bullet in the File map with:

```markdown
- `marginalia.py` — the live inspection UI (item #1 of the former "Open work"
  list). Stdlib-only `http.server` + a single vanilla-JS page; loads a trained
  `colophon.npz` and serves a five-region white-box inspector: a per-character
  entropy heatmap, the literal K-char context window, occlusion-based context
  saliency, a top-k next-char inspector, and the OSAI scorecard. Every signal is
  read from the weights via `colophon.inspect_prompt()` / `context_saliency()` —
  the honest version of what black-box tools can only simulate. No new
  dependencies. Not imported by `colophon.py`.
```

Replace the `Done: **Marginalia**` line with:

```markdown
Done: **Marginalia** — the live inspection UI (`marginalia.py`, stdlib-only
`http.server` + a single-page frontend) is now a five-region white-box context
inspector: per-character entropy heatmap, literal K-char context window,
occlusion-based context saliency (`context_saliency()`), a top-k next-char
inspector with the ground-truth char's rank, session aggregates, and the OSAI
scorecard. Framed throughout as "the real version of what black-box LLM tools
fake". Backed by `colophon.inspect_prompt()` / `context_saliency()`.
```

- [ ] **Step 2: Update `README.md`**

Find the section that describes Marginalia (search: `grep -n -i marginalia README.md`). Add this paragraph to it (or create a short "What Marginalia shows" subsection near the existing mention):

```markdown
### What Marginalia shows

Marginalia is a live, zero-dependency inspector for a trained Colophon model.
Type a prompt and every white-box signal the model has is rendered from its own
weights: a **confidence heatmap** (each character tinted by next-char entropy),
the **literal K-character context window** the model saw (with the pad horizon it
cannot see past), **occlusion-based context saliency** (which remembered
characters actually drove the prediction), a **top-k next-char inspector** with
where the real next character ranked, and the OSAI **openness scorecard**. It is
framed as the honest counterpart to black-box "observability" tools: a hosted API
exposes none of this, and where a tool like glassboxllm has to *simulate*
per-token confidence, Colophon reads it straight from the weights.
```

- [ ] **Step 3: Run the full suite**

Run: `python -m unittest test_colophon test_marginalia -v`
Expected: PASS — all tests green (existing gradient/off-map/colophon.json/handler tests plus the new `DisplayChar`, `InspectPrompt`, `ContextSaliency`, `AnalyzePrompt`, `SaliencyWrapper`, `IndexHtmlContract` classes). Report the exact count.

- [ ] **Step 4: Confirm `prompt_confidence` and the demo are untouched**

Run: `python colophon.py demo | tail -20`
Expected: the demo still trains, prints confidence + scorecard, and writes `colophon.npz` + `colophon.json` — unchanged behavior.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: document Marginalia's white-box context inspector"
```

---

## Self-Review

**Spec coverage:**
- `inspect_prompt` per-position records → Task 2. ✅
- `context_saliency` occlusion → Task 3. ✅
- `/api/analyze` new contract + back-compat `unknown_chars`/`off_map` → Task 4. ✅
- `/api/saliency` route + `pos` validation (400) → Task 5. ✅
- Five frontend regions + black-box contrast chips + banner → Task 6. ✅
- Default focus = lowest-confidence position → Task 6 (`analyze()` reduce). ✅
- Full-distribution note / top-k inspector → Task 6 inspector. ✅
- Aggregates (median entropy, anchors, off-map count) → Task 6 `renderAggregates`. ✅
- `_display_char` glyphs incl. `∅` pad → Task 1. ✅
- `prompt_confidence` unchanged; entropy-mean equality asserted → Tasks 2, 4. ✅
- Tests updated for changed contract → Tasks 4, 5, 6. ✅
- Docs → Task 7. ✅
- Zero new deps / stdlib-only asserted → Task 6 `test_no_external_dependencies`. ✅

**Placeholder scan:** No TBD/TODO; every code step shows full code; every test step shows real assertions.

**Type consistency:** `inspect_prompt`/`context_saliency` signatures and the record/window key names are identical across colophon.py definitions (Tasks 2-3), the marginalia wrappers (Tasks 4-5), and the frontend consumers (Task 6): `records[].{char,display,is_continuation,entropy,top_k,context_window,truth_rank,truth_prob,off_map}` and `window[].{char,display,delta,is_pad}`. `n_continuation` (colophon) vs `n` (marginalia wrapper) is intentional and consistently mapped.
