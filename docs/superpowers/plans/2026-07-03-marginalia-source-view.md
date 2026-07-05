# Marginalia Source View Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the source-echo panel's `file:line` label a real link that opens the exact training file the model saw, server-rendered from the corpus already loaded in memory.

**Architecture:** One new module-level renderer `source_page()` in `marginalia.py`, one new `GET /source` route in `make_handler()`'s `do_GET`, per-mode provenance copy in `MODE_META`, and a one-function frontend change (`renderSource()`). The file lookup is exact-name matching against the in-memory `(name, text)` list — no filesystem access at request time, so path traversal is impossible by construction.

**Tech Stack:** Python stdlib only (`http.server`, `html`, `urllib.parse`), vanilla JS, `unittest` with the existing `_ServerFixture` live-server pattern in `test_marginalia.py`.

**Spec:** `docs/superpowers/specs/2026-07-03-marginalia-source-view-design.md`

## Global Constraints

- Stdlib only — `requirements.txt` stays numpy-only; Marginalia adds zero runtime dependencies. The new page ships **zero JavaScript**.
- All corpus text, filenames, labels, and notes are HTML-escaped with `html.escape` before entering markup. Non-negotiable.
- Error semantics mirror existing routes: unknown mode → 400, known mode with absent model → 503, file not in corpus → 404, non-integer `line` → 400, out-of-range/missing `line` → 200 with no highlight. `/source` errors are **HTML** pages (it's a browser tab), `/api/*` errors stay JSON.
- One missing corpus never takes the page down (same degradation contract as every other route).
- TDD: every task writes its failing test first, runs it to see it fail, implements, runs it green, commits with a conventional prefix.
- Run the full suite (`python -m unittest test_marginalia -v`) before each commit; report exact pass counts.
- All work happens in `marginalia.py` and `test_marginalia.py` (plus README/CLAUDE.md in the docs task). `colophon.py` is untouched.

---

### Task 1: `source_page()` renderer

The pure-function core: given a label, filename, file text, optional highlight line, and provenance strings, return the complete HTML page. No server involvement — unit-testable directly.

**Files:**
- Modify: `marginalia.py` (imports at line 28; new constant + function after `corpus_sha256()`, which ends near line 173)
- Test: `test_marginalia.py` (new test class, add after `EmbeddingsWrapper`, near line 213)

**Interfaces:**
- Consumes: nothing new — `html` (stdlib) added to imports.
- Produces: `source_page(label, filename, text, line=None, note="", url="", sha="") -> str` (complete HTML document). Line rows render exactly as `<tr id="L{i}">` (no highlight) or `<tr id="L{i}" class="hit">` (highlight) — Task 2's route and tests rely on this exact markup.

- [ ] **Step 1: Write the failing tests**

Add to `test_marginalia.py` (after the `EmbeddingsWrapper` class):

```python
class SourcePageRender(unittest.TestCase):
    """source_page() renders one training file: numbered anchored lines,
    optional highlight, escaped everything, provenance footer."""

    def test_numbers_anchors_and_content(self):
        page = M.source_page("Periodic table", "018_argon.yaml",
                             "number: 18\nsymbol: Ar\n")
        self.assertIn('<tr id="L1">', page)
        self.assertIn('<tr id="L2">', page)
        self.assertIn("number: 18", page)
        self.assertIn("018_argon.yaml", page)
        self.assertIn("Periodic table", page)

    def test_highlight_only_requested_line(self):
        page = M.source_page("x", "f.yaml", "a: 1\nb: 2\n", line=2)
        self.assertIn('<tr id="L2" class="hit">', page)
        self.assertNotIn('<tr id="L1" class="hit">', page)

    def test_no_line_means_no_highlight(self):
        page = M.source_page("x", "f.yaml", "a: 1\n")
        self.assertNotIn('class="hit"', page)

    def test_corpus_text_is_escaped(self):
        page = M.source_page("x", "f.yaml", '<b>&"</b>\n')
        self.assertNotIn("<b>", page)
        self.assertIn("&lt;b&gt;", page)

    def test_label_and_filename_are_escaped(self):
        page = M.source_page("<lab>", "<f>.yaml", "a\n")
        self.assertNotIn("<lab>", page)
        self.assertIn("&lt;lab&gt;", page)
        self.assertNotIn("<f>.yaml", page)

    def test_footer_shows_note_url_and_sha(self):
        page = M.source_page("x", "f.yaml", "a\n", note="CC BY 4.0",
                             url="https://example.org/idx", sha="ab12cd")
        self.assertIn("CC BY 4.0", page)
        self.assertIn('href="https://example.org/idx"', page)
        self.assertIn("ab12cd", page)

    def test_page_ships_zero_javascript(self):
        page = M.source_page("x", "f.yaml", "a\n")
        self.assertNotIn("<script", page)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest test_marginalia.SourcePageRender -v`
Expected: 7 ERRORs with `AttributeError: module 'marginalia' has no attribute 'source_page'`

- [ ] **Step 3: Implement `source_page()`**

In `marginalia.py`, change the import line (line 28) from:

```python
import argparse, glob, hashlib, http.server, json, os, urllib.parse
```

to:

```python
import argparse, glob, hashlib, html, http.server, json, os, urllib.parse
```

Then add after `corpus_sha256()` (before `find_source_echo()`):

```python
_SOURCE_CSS = """
  :root { color-scheme: light dark; }
  body { font-family: ui-monospace, Menlo, Consolas, monospace; max-width: 900px;
         margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }
  h1 { font-size: 1.1rem; } h1 .glyph { opacity: .6; }
  .muted { opacity: .6; font-size: .85rem; }
  table { border-collapse: collapse; width: 100%; }
  td { padding: 0 .5rem; white-space: pre-wrap; word-break: break-all;
       font-size: .9rem; }
  td.num { text-align: right; opacity: .5; user-select: none; width: 1%; }
  tr.hit td { background: #2a62; }
  footer { margin-top: 1.5rem; }
"""


def source_page(label, filename, text, line=None, note="", url="", sha=""):
    """Render the /source view: the exact text of one training file, straight
    from the (name, text) corpus pairs already in memory. Numbered rows are
    anchored id="L<n>" so the main page's #L<n> fragment scrolls natively --
    this page ships zero JavaScript. `line` (if it matches a row) gets a
    highlight class; everything user- or corpus-derived is html.escape()d."""
    rows = []
    for i, raw in enumerate(text.splitlines(), 1):
        hit = ' class="hit"' if i == line else ""
        rows.append(f'<tr id="L{i}"{hit}><td class="num">{i}</td>'
                    f'<td>{html.escape(raw)}</td></tr>')
    head = f"{html.escape(label)} &mdash; {html.escape(filename)}"
    footer_bits = []
    if note:
        footer_bits.append(html.escape(note))
    if url:
        footer_bits.append(f'<a href="{html.escape(url)}">{html.escape(url)}</a>')
    if sha:
        footer_bits.append(f"corpus sha256 (PAD-joined snapshot): <code>{sha}</code>")
    footer = " &middot; ".join(footer_bits)
    return (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        f"<title>{head}</title>\n<style>{_SOURCE_CSS}</style>\n</head>\n<body>\n"
        f'<h1><span class="glyph">&#10087;</span> {head}</h1>\n'
        '<p class="muted">The exact text of one training file, served from the '
        "corpus copy this model actually trained on &mdash; ground truth, not a "
        "link out.</p>\n"
        f"<table>\n{''.join(rows)}\n</table>\n"
        f'<footer class="muted">{footer}</footer>\n</body>\n</html>\n'
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest test_marginalia.SourcePageRender -v`
Expected: 7 tests, all PASS

- [ ] **Step 5: Run the full suite, then commit**

Run: `python -m unittest test_marginalia -v` — expected: all pass, 0 failures.

```bash
git add marginalia.py test_marginalia.py
git commit -m "feat: add source_page() renderer for the Marginalia source view"
```

---

### Task 2: `GET /source` route

Wire the renderer into `make_handler()`: HTML error variants of the existing mode resolution, strict `line` parsing, in-memory file lookup, and the dispatch branch. All verified end-to-end through the live `_ServerFixture`.

**Files:**
- Modify: `marginalia.py` — `_mode_cfg()` (line ~900), new `_send_html_error()` beside `_send_json()` (line ~896), new branch in `do_GET()` before the final `else` (line ~973)
- Test: `test_marginalia.py` (new test class after `HandlerDegraded`, near line 386)

**Interfaces:**
- Consumes: `source_page(label, filename, text, line=None, note="", url="", sha="")` from Task 1; `corpus_sha256(files)` (existing, line 168).
- Produces: `GET /source?mode=<id>&file=<name>&line=<n>` → 200 `text/html`; errors 400/503/404 as HTML. `_mode_cfg(qs, html_errors=False)` — the existing call sites keep JSON behavior by default. Reads optional `cfg["source_note"]` / `cfg["source_url"]` via `.get()`, so modes without them (all of them until Task 3, and the test fixtures forever) render an emptier footer rather than erroring.

- [ ] **Step 1: Write the failing tests**

Add to `test_marginalia.py` (after `HandlerDegraded`). `_make_modes(model, files=...)` already accepts a `files` tuple — no fixture changes needed.

```python
SOURCE_FILES = (("entry.yaml", "class: open\nlicense: mit\n"),
                ("evil.yaml", '<script>alert("x")</script>\n'))


class SourceRoute(unittest.TestCase):
    """GET /source serves one training file from the in-memory corpus as an
    HTML page. Lookup is by exact name against (name, text) pairs -- no
    filesystem access at request time, so traversal is impossible by
    construction (the ../ test below 404s without touching disk)."""

    @classmethod
    def setUpClass(cls):
        cls.server = _ServerFixture(_make_modes(_make_model(), files=SOURCE_FILES))

    @classmethod
    def tearDownClass(cls):
        cls.server.close()

    def test_serves_file_with_highlight(self):
        status, headers, body = self.server.get("/source?file=entry.yaml&line=2")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
        page = body.decode("utf-8")
        self.assertIn("license: mit", page)
        self.assertIn('<tr id="L2" class="hit">', page)
        self.assertIn('<tr id="L1">', page)

    def test_missing_line_renders_without_highlight(self):
        status, _, body = self.server.get("/source?file=entry.yaml")
        self.assertEqual(status, 200)
        self.assertNotIn(b'class="hit"', body)

    def test_out_of_range_line_renders_without_highlight(self):
        status, _, body = self.server.get("/source?file=entry.yaml&line=99")
        self.assertEqual(status, 200)
        self.assertNotIn(b'class="hit"', body)

    def test_non_integer_line_400(self):
        status, headers, _ = self.server.get("/source?file=entry.yaml&line=nope")
        self.assertEqual(status, 400)
        self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")

    def test_unknown_file_404(self):
        status, headers, _ = self.server.get("/source?file=../colophon.py")
        self.assertEqual(status, 404)
        # Our handler's header, not stdlib send_error()'s "text/html;charset=utf-8"
        # (no space) -- this pins that the 404 came from the route's own lookup.
        self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")

    def test_unknown_mode_400_as_html(self):
        status, headers, _ = self.server.get("/source?mode=nope&file=entry.yaml")
        self.assertEqual(status, 400)
        self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")

    def test_absent_mode_503_as_html(self):
        server = _ServerFixture(_make_modes(None, files=SOURCE_FILES))
        try:
            status, headers, _ = server.get("/source?file=entry.yaml")
            self.assertEqual(status, 503)
            self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
        finally:
            server.close()

    def test_api_errors_stay_json(self):
        # The html_errors switch must not leak into the /api/ routes.
        status, headers, body = self.server.get("/api/analyze?mode=nope&prompt=x")
        self.assertEqual(status, 400)
        self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
        self.assertIn("error", json.loads(body))

    def test_corpus_text_is_escaped_end_to_end(self):
        status, _, body = self.server.get("/source?file=evil.yaml")
        self.assertEqual(status, 200)
        self.assertNotIn(b"<script", body)  # the page ships zero JS at all
        self.assertIn(b"&lt;script&gt;", body)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest test_marginalia.SourceRoute -v`
Expected: `test_api_errors_stay_json` PASSes (existing behavior); the other 8 FAIL — every /source request falls through to stdlib `send_error(404)`, so the status asserts fail on the 200/400/503 cases, and `test_unknown_file_404` fails on Content-Type (stdlib sends `text/html;charset=utf-8`, no space).

- [ ] **Step 3: Implement the route**

In `marginalia.py`, add `_send_html_error` right after `_send_json` (inside the `Handler` class):

```python
        def _send_html_error(self, status, msg):
            body = ('<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">'
                    f"<title>{status}</title></head><body>"
                    f"<p>{html.escape(msg)}</p></body></html>\n").encode("utf-8")
            self._send(status, "text/html; charset=utf-8", body)
```

Replace `_mode_cfg` with (same resolution, selectable error format):

```python
        def _mode_cfg(self, qs, html_errors=False):
            """Resolve ?mode= to a usable mode config, or send the right error
            (400 unknown / 503 absent -- JSON for the /api/ routes, HTML for
            /source, which renders in a browser tab) and return None."""
            mode = qs.get("mode", [default_mode])[0]
            if mode not in modes:
                msg = f"unknown mode: {mode!r}"
                if html_errors:
                    self._send_html_error(400, msg)
                else:
                    self._send_json({"error": msg}, status=400)
                return None
            cfg = modes[mode]
            if cfg.get("model") is None:
                hint = cfg.get("train_hint", "python colophon.py demo")
                msg = f"no trained model for '{mode}' -- run `{hint}` first"
                if html_errors:
                    self._send_html_error(503, msg)
                else:
                    self._send_json({"error": msg}, status=503)
                return None
            return cfg
```

Add the route branch in `do_GET`, between the `/api/saliency` branch and the final `else`:

```python
            elif parsed.path == "/source":
                qs = urllib.parse.parse_qs(parsed.query)
                cfg = self._mode_cfg(qs, html_errors=True)
                if cfg is None:
                    return
                line_raw = qs.get("line", [None])[0]
                line = None
                if line_raw is not None:
                    try:
                        line = int(line_raw)
                    except ValueError:
                        self._send_html_error(400, "line must be an integer")
                        return
                filename = qs.get("file", [""])[0]
                # Exact-name lookup in the in-memory corpus; never touches disk,
                # so a path-traversal filename can only ever 404.
                text = next((t for name, t in cfg.get("files", ())
                             if name == filename), None)
                if text is None:
                    self._send_html_error(
                        404, f"no file named {filename!r} in this mode's corpus")
                    return
                files = cfg.get("files", ())
                body = source_page(cfg.get("label", ""), filename, text,
                                   line=line,
                                   note=cfg.get("source_note", ""),
                                   url=cfg.get("source_url", ""),
                                   sha=corpus_sha256(files) if files else "")
                self._send(200, "text/html; charset=utf-8", body.encode("utf-8"))
```

Also update `make_handler`'s docstring (line ~871) — replace the sentence beginning "The scorecard and page serve regardless..." with:

```python
    The scorecard and page serve regardless of which models loaded; /api/analyze,
    /api/saliency, and /source report 400 for an unknown mode and 503 for a known
    mode whose model is absent, so one missing corpus never takes the page down.
    /source additionally serves a mode's training files by exact in-memory name
    lookup (404 otherwise) as server-rendered HTML.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest test_marginalia.SourceRoute -v`
Expected: 9 tests, all PASS

- [ ] **Step 5: Run the full suite, then commit**

Run: `python -m unittest test_marginalia -v` — expected: all pass (the existing `_mode_cfg` call sites are unchanged by the default `html_errors=False`).

```bash
git add marginalia.py test_marginalia.py
git commit -m "feat: serve training files at GET /source from the in-memory corpus"
```

---

### Task 3: Provenance notes in `MODE_META`

Real copy for the footer. `main()` already spreads `**meta` into each mode config (line 1045), so adding keys to `MODE_META` reaches `cfg` with no plumbing. The OSAI note must stay honest: that mode trains on either the bundled fictional `sample_data/` or the real CC-BY index, and the note must not claim index provenance for the sample files.

**Files:**
- Modify: `marginalia.py` — the three `MODE_META` entries (lines 47–90)
- Test: `test_marginalia.py` (new test class after `SourceRoute`)

**Interfaces:**
- Consumes: Task 2's route already reads `cfg.get("source_note")` / `cfg.get("source_url")`.
- Produces: `MODE_META[<mode>]["source_note"]` (non-empty str, all three modes) and `MODE_META["osai"]["source_url"]` (the Codeberg URL). Exact strings below.

- [ ] **Step 1: Write the failing tests**

```python
class SourceProvenance(unittest.TestCase):
    """Every mode documents where its corpus comes from, and the note reaches
    the served /source page. The OSAI note stays honest about sample_data
    being original stand-ins, not index entries (keep-it-honest rule)."""

    def test_every_mode_has_a_source_note(self):
        for mid, meta in M.MODE_META.items():
            self.assertTrue(meta.get("source_note", "").strip(), mid)

    def test_osai_cites_the_index_and_flags_the_sample(self):
        note = M.MODE_META["osai"]["source_note"]
        self.assertIn("CC BY 4.0", note)
        self.assertIn("10.5281/zenodo.15386042", note)
        self.assertIn("sample_data", note)
        self.assertIn("codeberg.org", M.MODE_META["osai"]["source_url"])

    def test_generated_corpora_cite_their_generators(self):
        self.assertIn("build_elements.py", M.MODE_META["elements"]["source_note"])
        self.assertIn("build_kana.py", M.MODE_META["kana"]["source_note"])

    def test_note_reaches_the_served_page(self):
        modes = {"osai": {"model": _make_model(),
                          "files": (("entry.yaml", "class: open\n"),),
                          **M.MODE_META["osai"]}}
        server = _ServerFixture(modes)
        try:
            status, _, body = server.get("/source?file=entry.yaml")
            self.assertEqual(status, 200)
            self.assertIn(b"10.5281/zenodo.15386042", body)
            self.assertIn(b"codeberg.org", body)
        finally:
            server.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest test_marginalia.SourceProvenance -v`
Expected: 4 FAILures (`source_note` missing from every `MODE_META` entry)

- [ ] **Step 3: Add the notes to `MODE_META`**

In the `"osai"` entry, after the `"train_hint"` line, add:

```python
        "source_note": ("OSAI-schema YAML. Trained on the real European Open "
                        "Source AI Index, this corpus is CC BY 4.0 -- cite "
                        "doi:10.5281/zenodo.15386042. The bundled sample_data "
                        "files are original, fictional stand-ins, not index "
                        "entries."),
        "source_url": "https://codeberg.org/AI-Technology-Assessment/main-database",
```

In the `"elements"` entry, after its `"train_hint"`, add:

```python
        "source_note": ("118 undisputed IUPAC facts, generated byte-for-byte "
                        "by the committed teaching_data/build_elements.py."),
```

In the `"kana"` entry, after its `"train_hint"`, add:

```python
        "source_note": ("71 hiragana with traditional Hepburn romaji, generated "
                        "byte-for-byte by the committed teaching_data/build_kana.py."),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest test_marginalia.SourceProvenance -v`
Expected: 4 tests, all PASS

- [ ] **Step 5: Run the full suite, then commit**

Run: `python -m unittest test_marginalia -v` — expected: all pass. (`/api/modes` serializes only explicit fields, so the new keys change no existing payloads.)

```bash
git add marginalia.py test_marginalia.py
git commit -m "feat: per-mode provenance notes on the Marginalia source view"
```

---

### Task 4: Frontend link in `renderSource()`

The only UI change: wrap the `file:line` label in an anchor to `/source` with the `#L<n>` fragment. DOM-built (no innerHTML with data), `target="_blank" rel="noopener"`.

**Files:**
- Modify: `marginalia.py` — the JS `renderSource()` function inside `INDEX_HTML` (line ~497)
- Test: `test_marginalia.py` — extend the `IndexHtmlContract` class (line ~388)

**Interfaces:**
- Consumes: the `/source` URL shape from Task 2; JS globals `activeMode` (line ~759), `sourceLabelEl` (line ~458), and `analyze` data `source.file` / `source.line` from `find_source_echo()`.
- Produces: nothing later tasks use.

- [ ] **Step 1: Write the failing tests**

Add two methods to the existing `IndexHtmlContract` class:

```python
    def test_source_match_links_to_source_view(self):
        html = M.INDEX_HTML
        self.assertIn("'/source?mode='", html)
        self.assertIn("noopener", html)

    def test_source_link_carries_line_fragment(self):
        self.assertIn("'#L'", M.INDEX_HTML)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest test_marginalia.IndexHtmlContract -v`
Expected: the 2 new tests FAIL; the existing ones PASS

- [ ] **Step 3: Update `renderSource()`**

In `INDEX_HTML`'s script, replace the current `renderSource` (which sets `sourceLabelEl.textContent = `${source.file}:${source.line}``) with:

```javascript
function renderSource(source) {
  sourceSnippetEl.innerHTML = '';
  sourceLabelEl.textContent = '';
  if (!source || !source.matched) {
    sourceLabelEl.textContent = 'no match -- this context does not appear verbatim in the training corpus';
    return;
  }
  const a = document.createElement('a');
  a.href = '/source?mode=' + encodeURIComponent(activeMode) +
           '&file=' + encodeURIComponent(source.file) +
           '&line=' + source.line + '#L' + source.line;
  a.target = '_blank';
  a.rel = 'noopener';
  a.title = 'open this training file (served from the same in-memory corpus)';
  a.textContent = `${source.file}:${source.line}`;
  sourceLabelEl.appendChild(a);
  const preSpan = document.createElement('span');
  preSpan.className = 'prompt-part';
  preSpan.textContent = source.pre;
  const matchSpan = document.createElement('span');
  matchSpan.className = 'cont-part';
  matchSpan.textContent = source.match;
  const postSpan = document.createElement('span');
  postSpan.className = 'prompt-part';
  postSpan.textContent = source.post;
  sourceSnippetEl.append(preSpan, matchSpan, postSpan);
}
```

(The only changes from the original: `sourceLabelEl.textContent = ''` reset at the top, and the anchor replacing the plain-text label. The snippet spans are unchanged.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest test_marginalia.IndexHtmlContract -v`
Expected: all PASS — including `test_no_external_dependencies` (the new URL is relative, no `http://`)

- [ ] **Step 5: Manual end-to-end verification**

The repo has a trained `colophon.npz` checked in beside `marginalia.py`.

Run: `python marginalia.py --port 8765` then open `http://127.0.0.1:8765`, type `weights_basemodel:` and confirm: (1) the source panel's label is now a link; (2) clicking it opens a new tab showing a `sample_data/*.yaml` file with the matched line highlighted and scrolled to; (3) the footer shows the OSAI note, doi, Codeberg link, and corpus sha; (4) hand-editing the tab's URL to `file=../colophon.py` returns a 404 page. Stop the server with Ctrl+C.

- [ ] **Step 6: Run the full suite, then commit**

Run: `python -m unittest test_marginalia -v` — expected: all pass.

```bash
git add marginalia.py test_marginalia.py
git commit -m "feat: link the source-echo match to the /source training-file view"
```

---

### Task 5: Documentation

Record the feature where the repo documents Marginalia: the README's "What Marginalia shows" section and CLAUDE.md's file-map bullets.

**Files:**
- Modify: `README.md` (the "What Marginalia shows" paragraph, lines 157–169)
- Modify: `CLAUDE.md` (the `marginalia.py` and `test_marginalia.py` file-map bullets)

**Interfaces:**
- Consumes: everything shipped in Tasks 1–4.
- Produces: nothing — docs only.

- [ ] **Step 1: Update README.md**

In the "What Marginalia shows" paragraph, extend the sentence describing the signals. After the words "and the OSAI **openness scorecard**" insert a new sentence so the passage reads:

```markdown
with click-to-see nearest neighbors by cosine similarity), and the OSAI
**openness scorecard**. When a prompt matches the corpus verbatim, the source
panel's `file:line` label links to a **served view of the actual training
file** — rendered from the same in-memory corpus the model trained on, matched
line highlighted, with a per-corpus provenance footer (for the real index:
CC BY 4.0, doi:10.5281/zenodo.15386042). It is framed as the honest counterpart
to black-box "observability" tools: a hosted API exposes none of this, and
where a tool like glassboxllm has to *simulate* per-token confidence, Colophon
reads it straight from the weights.
```

- [ ] **Step 2: Update CLAUDE.md**

In the `marginalia.py` file-map bullet, after the sentence ending "and a literal source-in-training-data match (`find_source_echo()`)." add:

```markdown
  The matched `file:line` links to `GET /source` — a zero-JS server-rendered
  view of that training file from the in-memory corpus (exact-name lookup, so
  no filesystem access at request time), matched line highlighted, with a
  per-mode provenance footer (`source_note`/`source_url` in `MODE_META`; OSAI
  cites CC-BY + doi) and the corpus sha256.
```

In the `test_marginalia.py` bullet, after "(incl. 400 on a bad `pos`)," add:

```markdown
  that `source_page()` / the `/source` route serve a training file from memory
  with escaping and highlight (400/503/404 on bad mode/absent model/unknown
  file, HTML errors not JSON), that every mode carries a provenance note,
```

- [ ] **Step 3: Run the full suite one last time**

Run: `python -m unittest test_marginalia test_colophon test_elements test_kana -v`
Expected: all pass, 0 failures — report the exact count.

- [ ] **Step 4: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: document the Marginalia source view"
```
