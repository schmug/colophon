# Corpus Browser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a promptless "read the entire corpus" browse view to both Marginalia and Incipit so a reader can list and read every training file independent of what they typed.

**Architecture:** One shared renderer (`corpus_index_page`) and one shared error renderer (`html_error_page`) in `marginalia.py`, called by thin `/corpus` (both servers) and `/source` (new in Incipit) route handlers. Incipit already `import marginalia`, so it reuses `source_page`/`corpus_index_page` verbatim and holds all three corpora — including `dialogue` — in memory. Lists files and drills into `/source`; never inlines file contents.

**Tech Stack:** Python stdlib `http.server` + NumPy (servers); vanilla JS (Marginalia page); Vite + React + TypeScript (Incipit front-end only); `unittest` (Python tests); vitest (front-end tests).

## Global Constraints

Every task's requirements implicitly include these (verbatim from the spec):

- **Servers are stdlib + NumPy only; no new dependencies.** Node/npm/Vite/React are build-time only, confined to `incipit/`.
- **Served pages ship zero JavaScript** (`/source`, `/corpus`, error pages). The Marginalia main page's existing vanilla JS is the only exception.
- **`/source` uses exact-name in-memory lookup** — no filesystem access at request time; a `../` filename can only 404.
- **Everything corpus/user-derived is `html.escape()`d** (filenames, counts shown as text, note, url, sha, file contents).
- **Gating via `_mode_cfg`:** unknown mode → 400, absent model → 503, both as **HTML** for the browser-facing routes (JSON stays on `/api/` routes). An absent mode is unreachable from the UI (disabled toggle).
- **Char total is the canonical PAD-joined length** — `len(("\n" + colophon.PAD + "\n").join(texts))`, equal to `colophon.json`'s `num_characters`; strictly greater than the sum of per-file char counts by `3 × (N−1)` boundary tokens.
- **Line count is `len(text.splitlines())`** — matches the numbered rows `source_page` renders.
- **Incipit stays stateless** — new routes are read-only GETs over the already-loaded corpus; no session store.

---

### Task 1: Extract the shared HTML error renderer

**Files:**
- Modify: `marginalia.py` (add module-level `html_error_page`; rewrite `_send_html_error` at `marginalia.py:969-973` to call it)
- Test: `test_marginalia.py` (add `HtmlErrorPage` class)

**Interfaces:**
- Produces: `marginalia.html_error_page(status: int, msg: str) -> bytes` — the `<!DOCTYPE html>…` error body, `msg` html-escaped, `<title>{status}</title>`. Used by both servers' `_send_html_error`.

- [ ] **Step 1: Write the failing test**

Add to `test_marginalia.py` (after the `SourcePageRender` class, before `_make_model`):

```python
class HtmlErrorPage(unittest.TestCase):
    """The error body is factored to a module-level helper so incipit.py can
    reuse the exact same renderer. This pins its shape."""

    def test_returns_escaped_bytes_with_status_title(self):
        body = M.html_error_page(404, "no file named <x>")
        self.assertIsInstance(body, bytes)
        page = body.decode("utf-8")
        self.assertIn("<title>404</title>", page)
        self.assertIn("no file named &lt;x&gt;", page)
        self.assertNotIn("<x>", page)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest test_marginalia.HtmlErrorPage -v`
Expected: FAIL with `AttributeError: module 'marginalia' has no attribute 'html_error_page'`

- [ ] **Step 3: Add the module-level helper**

In `marginalia.py`, immediately **before** `def source_page(` (line 202), add:

```python
def html_error_page(status, msg):
    """The tiny zero-JS HTML error body shared by both servers'
    _send_html_error. `msg` is html-escaped; the page ships no JavaScript."""
    return ('<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">'
            f"<title>{status}</title></head><body>"
            f"<p>{html.escape(msg)}</p></body></html>\n").encode("utf-8")
```

- [ ] **Step 4: Rewrite `_send_html_error` to call it**

Replace the method body at `marginalia.py:969-973`:

```python
        def _send_html_error(self, status, msg):
            self._send(status, "text/html; charset=utf-8",
                       html_error_page(status, msg))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m unittest test_marginalia -v`
Expected: PASS — the new test passes and every existing `SourceRoute`/`ModeRouting` test (which exercise `_send_html_error` through the live server) still passes, proving the refactor is behavior-preserving.

- [ ] **Step 6: Commit**

```bash
git add marginalia.py test_marginalia.py
git commit -m "refactor: extract html_error_page for cross-server reuse

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: The shared corpus-index renderer

**Files:**
- Modify: `marginalia.py` (add `corpus_index_page` after `source_page`, ~line 232)
- Test: `test_marginalia.py` (add `CorpusIndexPage` class)

**Interfaces:**
- Consumes: `_SOURCE_CSS`, `colophon.PAD`, `html`, `urllib.parse` (all already in `marginalia.py`).
- Produces: `marginalia.corpus_index_page(label: str, files: list[tuple[str, str]], mode: str, note: str = "", url: str = "", sha: str = "") -> str` — a full zero-JS HTML page listing every file (name link → `/source?mode=&file=`, line count, char count), a "N files · M characters" summary where M is the canonical PAD-joined length, and the shared provenance footer.

- [ ] **Step 1: Write the failing tests**

Add to `test_marginalia.py` (after the `HtmlErrorPage` class from Task 1):

```python
class CorpusIndexPage(unittest.TestCase):
    files = [("a.yaml", "x\ny\n"), ("b.yaml", "zz\n")]

    def test_lists_files_with_source_links(self):
        page = M.corpus_index_page("Openness index", self.files, "osai")
        self.assertIn('href="/source?mode=osai&amp;file=a.yaml"', page)
        self.assertIn(">a.yaml</a>", page)
        self.assertIn(">b.yaml</a>", page)

    def test_summary_uses_canonical_padjoined_total(self):
        page = M.corpus_index_page("L", self.files, "osai")
        joined = ("\n" + C.PAD + "\n").join(t for _, t in self.files)
        self.assertIn("2 files", page)
        self.assertIn(f"{len(joined)} characters", page)
        # canonical total exceeds the naive per-file char sum by the boundaries
        self.assertGreater(len(joined), sum(len(t) for _, t in self.files))

    def test_total_matches_colophon_num_characters(self):
        # /corpus must not print a number that disagrees with colophon.json.
        text = ("\n" + C.PAD + "\n").join(t for _, t in self.files)
        chars, _, _ = C.build_vocab(text)
        manifest = C.data_manifest(text, ["a.yaml", "b.yaml"], chars)
        page = M.corpus_index_page("L", self.files, "osai")
        self.assertIn(f"{manifest['num_characters']} characters", page)

    def test_per_row_line_and_char_counts(self):
        page = M.corpus_index_page("L", self.files, "osai")
        # a.yaml: 2 lines (splitlines), 4 chars; b.yaml: 1 line, 3 chars
        self.assertIn('<td class="num">2</td><td class="num">4</td>', page)
        self.assertIn('<td class="num">1</td><td class="num">3</td>', page)

    def test_escapes_names_and_note(self):
        page = M.corpus_index_page("L", [("<b>.yaml", "x\n")], "osai",
                                   note="<i>n</i>")
        self.assertNotIn("<b>.yaml", page)
        self.assertIn("&lt;b&gt;.yaml", page)
        self.assertNotIn("<i>n</i>", page)
        self.assertIn("&lt;i&gt;n&lt;/i&gt;", page)

    def test_ships_no_javascript(self):
        page = M.corpus_index_page("L", self.files, "osai")
        self.assertNotIn("<script", page)

    def test_empty_corpus_is_honest_not_a_crash(self):
        page = M.corpus_index_page("L", [], "osai")
        self.assertIn("0 files", page)
        self.assertIn("0 characters", page)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest test_marginalia.CorpusIndexPage -v`
Expected: FAIL with `AttributeError: module 'marginalia' has no attribute 'corpus_index_page'`

- [ ] **Step 3: Implement the renderer**

In `marginalia.py`, immediately **after** the `source_page` function (after line 231, before `def find_source_echo`), add:

```python
def corpus_index_page(label, files, mode, note="", url="", sha=""):
    """Render GET /corpus: a promptless index of every training file in one
    mode's corpus, each linking to /source. Zero JS, same CSS and provenance
    footer as source_page(). `files` is the list of (name, text) pairs already
    in memory. Contents are NOT inlined here (that is /source's job) -- only
    names, line counts (len(splitlines), matching source_page's rows), and raw
    per-file char counts. The summary total is the canonical PAD-joined length
    load_corpus feeds the model (== colophon.json num_characters), so it is
    slightly larger than the sum of the per-row char counts -- the difference
    is the boundary tokens the model genuinely sees. Everything name/count/
    footer-derived is html.escape()d."""
    rows = []
    for name, text in files:
        n_lines = len(text.splitlines())
        n_chars = len(text)
        href = ("/source?mode=" + urllib.parse.quote(mode, safe="")
                + "&file=" + urllib.parse.quote(name, safe=""))
        rows.append(
            f'<tr><td><a href="{html.escape(href)}">{html.escape(name)}</a></td>'
            f'<td class="num">{n_lines}</td>'
            f'<td class="num">{n_chars}</td></tr>')
    total_chars = len(("\n" + colophon.PAD + "\n").join(t for _, t in files)) \
        if files else 0
    head = f"{html.escape(label)} &mdash; corpus"
    footer_bits = []
    if note:
        footer_bits.append(html.escape(note))
    if url:
        footer_bits.append(f'<a href="{html.escape(url)}">{html.escape(url)}</a>')
    if sha:
        footer_bits.append(
            f"corpus sha256 (PAD-joined snapshot): <code>{html.escape(sha)}</code>")
    footer = " &middot; ".join(footer_bits)
    return (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        f"<title>{head}</title>\n<style>{_SOURCE_CSS}</style>\n</head>\n<body>\n"
        f'<h1><span class="glyph">&#10087;</span> {head}</h1>\n'
        '<p class="muted">Every file the model actually trained on, served from '
        "the in-memory corpus &mdash; ground truth, not a link out.</p>\n"
        f'<p class="muted">{len(files)} files &middot; {total_chars} characters. '
        "Totals count the corpus as the model sees it, including the "
        "<code>\\n&#9216;\\n</code> boundary token between entries; per-file "
        "counts are raw file lengths.</p>\n"
        '<table>\n'
        '<tr><td class="muted">file</td>'
        '<td class="num muted">lines</td>'
        '<td class="num muted">chars</td></tr>\n'
        f"{''.join(rows)}\n</table>\n"
        f'<footer class="muted">{footer}</footer>\n</body>\n</html>\n')
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest test_marginalia.CorpusIndexPage -v`
Expected: PASS (all seven)

- [ ] **Step 5: Commit**

```bash
git add marginalia.py test_marginalia.py
git commit -m "feat: add corpus_index_page shared renderer

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Marginalia `/corpus` route + entry-point link

**Files:**
- Modify: `marginalia.py` (add `/corpus` branch in `do_GET` after the `/source` branch, ~line 1083; add the browse link to `INDEX_HTML` source panel at `marginalia.py:433-437`; wire its href in the JS near `activeMode = mode.id`, ~line 863)
- Test: `test_marginalia.py` (add `CorpusRoute` class; add one assertion to `IndexHtmlContract`)

**Interfaces:**
- Consumes: `corpus_index_page` (Task 2), `corpus_sha256` (`marginalia.py:179`), `_mode_cfg` (`marginalia.py:975`).

- [ ] **Step 1: Write the failing route tests**

Add to `test_marginalia.py` (after the `SourceProvenance` class):

```python
class CorpusRoute(unittest.TestCase):
    """GET /corpus lists a mode's whole corpus as a zero-JS HTML page, each
    file linking to /source. Promptless: no verbatim match required."""

    @classmethod
    def setUpClass(cls):
        modes = {"osai": {"model": _make_model(), "files": SOURCE_FILES,
                          "label": "Openness index",
                          "source_note": "cite the source",
                          "source_url": "https://example.org/db"}}
        cls.server = _ServerFixture(modes)

    @classmethod
    def tearDownClass(cls):
        cls.server.close()

    def test_lists_every_file_as_html(self):
        status, headers, body = self.server.get("/corpus")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
        page = body.decode("utf-8")
        for name, _ in SOURCE_FILES:
            self.assertIn(f"file={name}", page)

    def test_links_round_trip_to_source(self):
        for name, _ in SOURCE_FILES:
            status, _, _ = self.server.get(f"/source?mode=osai&file={name}")
            self.assertEqual(status, 200)

    def test_footer_carries_provenance(self):
        _, _, body = self.server.get("/corpus")
        page = body.decode("utf-8")
        self.assertIn("cite the source", page)
        self.assertIn("https://example.org/db", page)

    def test_unknown_mode_400_html(self):
        status, headers, _ = self.server.get("/corpus?mode=nope")
        self.assertEqual(status, 400)
        self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")

    def test_absent_model_503_html(self):
        server = _ServerFixture({"osai": {"model": None, "files": SOURCE_FILES,
                                          "label": "x"}})
        try:
            status, headers, _ = server.get("/corpus")
            self.assertEqual(status, 503)
            self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
        finally:
            server.close()
```

Add this method inside the existing `IndexHtmlContract` class:

```python
    def test_index_has_browse_corpus_link(self):
        status, _, body = self.server.get("/")
        self.assertEqual(status, 200)
        self.assertIn(b'id="browse-corpus"', body)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest test_marginalia.CorpusRoute test_marginalia.IndexHtmlContract.test_index_has_browse_corpus_link -v`
Expected: FAIL — `/corpus` returns 404 (no branch yet); the index link assertion fails (no element yet).

- [ ] **Step 3: Add the `/corpus` route**

In `marginalia.py` `do_GET`, immediately **after** the `/source` branch ends (after line 1083, before `else: self.send_error(404)`), add:

```python
            elif parsed.path == "/corpus":
                qs = urllib.parse.parse_qs(parsed.query)
                cfg = self._mode_cfg(qs, html_errors=True)
                if cfg is None:
                    return
                mode = qs.get("mode", [default_mode])[0]
                files = cfg.get("files", ())
                body = corpus_index_page(
                    cfg.get("label", ""), files, mode,
                    note=cfg.get("source_note", ""),
                    url=cfg.get("source_url", ""),
                    sha=corpus_sha256(files) if files else "")
                self._send(200, "text/html; charset=utf-8", body.encode("utf-8"))
```

- [ ] **Step 4: Add the entry-point link to `INDEX_HTML`**

Replace the source panel at `marginalia.py:433-437` with (adds one line before `</div>`):

```html
<div class="panel">
  <div class="muted">source in training data (literal longest-suffix match, ground truth):</div>
  <div id="source-label" class="muted">&nbsp;</div>
  <div id="source-snippet" class="continuation"></div>
  <div class="muted" style="margin-top:.35rem"><a id="browse-corpus" href="/corpus" target="_blank" rel="noopener">browse the full corpus &rarr;</a> &mdash; read every training file, no prompt needed</div>
</div>
```

- [ ] **Step 5: Wire the link's href to the active mode**

In the JS, near the other element handles (`marginalia.py:516`), add after the `sourceLabelEl`/`sourceSnippetEl` line:

```javascript
const browseCorpusEl = $('browse-corpus');
```

Then find where the active mode is set (`marginalia.py:863`, `activeMode = mode.id;`) and add immediately after it:

```javascript
  if (browseCorpusEl) browseCorpusEl.href = '/corpus?mode=' + encodeURIComponent(activeMode);
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m unittest test_marginalia -v`
Expected: PASS (full suite, including `CorpusRoute` and the index link assertion)

- [ ] **Step 7: Manually verify the entry point renders**

Run: `python marginalia.py` (needs `colophon.npz`; run `python colophon.py demo` first if absent), open `http://127.0.0.1:8765`, confirm the "browse the full corpus →" link appears in the source panel and opens `/corpus?mode=osai` listing every file, each opening in `/source`. Switch modes and confirm the link's `?mode=` follows.

- [ ] **Step 8: Commit**

```bash
git add marginalia.py test_marginalia.py
git commit -m "feat: add /corpus browse route + entry point to Marginalia

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Incipit `/source` route + provenance + error plumbing

**Files:**
- Modify: `incipit.py` (add `source_note`/`source_url` to `MODE_META` entries at `incipit.py:58-92`; add `_send_html_error`; add `html_errors=` param to `_mode_cfg` at `incipit.py:253`; add `/source` branch in `do_GET` at `incipit.py:354-363`)
- Test: `test_incipit.py` (add `ModeProvenance` and `IncipitSourceRoute` classes + `INCIPIT_SOURCE_FILES` fixture)

**Interfaces:**
- Consumes: `marginalia.source_page`, `marginalia.corpus_sha256`, `marginalia.html_error_page` (Tasks 1). `_make_modes(model, files=())` spreads `**I.MODE_META[...]`, so it picks up the new `source_note` automatically.
- Produces: `GET /source?mode=&file=&line=` on Incipit, HTML errors, mirroring Marginalia's. `_mode_cfg(mode, html_errors=False)`.

- [ ] **Step 1: Write the failing tests**

Add to `test_incipit.py` (after the `_turn_body` helper, before `ServerRouting`):

```python
INCIPIT_SOURCE_FILES = (("q.txt", "user: hi\nmodel: hello\n"),
                        ("evil.txt", '<script>bad()</script>\n'))


class ModeProvenance(unittest.TestCase):
    def test_every_mode_documents_its_source(self):
        for mid, meta in I.MODE_META.items():
            self.assertTrue(meta.get("source_note"),
                            f"{mid} is missing a source_note")


class IncipitSourceRoute(unittest.TestCase):
    """GET /source serves one training file from a mode's in-memory corpus --
    Incipit's own route (it links to no other server). Covers the dialogue-
    shaped corpus that has no home in Marginalia."""

    @classmethod
    def setUpClass(cls):
        cls.server = _ServerFixture(
            _make_modes(_make_model(), files=INCIPIT_SOURCE_FILES))

    @classmethod
    def tearDownClass(cls):
        cls.server.close()

    def test_serves_dialogue_shaped_file_with_highlight(self):
        status, headers, body = self.server.get(
            "/source?mode=elements64&file=q.txt&line=2")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
        page = body.decode("utf-8")
        self.assertIn("model: hello", page)
        self.assertIn('<tr id="L2" class="hit">', page)

    def test_traversal_name_404_html(self):
        status, headers, _ = self.server.get(
            "/source?mode=elements64&file=../incipit.py")
        self.assertEqual(status, 404)
        self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")

    def test_bad_line_400_html(self):
        status, headers, _ = self.server.get(
            "/source?mode=elements64&file=q.txt&line=nope")
        self.assertEqual(status, 400)
        self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")

    def test_absent_model_503_html(self):
        status, headers, _ = self.server.get("/source?mode=dialogue&file=q.txt")
        self.assertEqual(status, 503)
        self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")

    def test_unknown_mode_400_html(self):
        status, headers, _ = self.server.get("/source?mode=nope&file=q.txt")
        self.assertEqual(status, 400)
        self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")

    def test_contents_escaped(self):
        _, _, body = self.server.get("/source?mode=elements64&file=evil.txt")
        self.assertNotIn(b"<script", body)
        self.assertIn(b"&lt;script&gt;", body)

    def test_not_swallowed_by_static(self):
        # /source is matched before _serve_static; the fixture's dist_dir does
        # not exist, so a mis-ordered route would return the build-help page.
        _, _, body = self.server.get("/source?mode=elements64&file=q.txt")
        self.assertNotIn(b"npm run build", body)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest test_incipit.ModeProvenance test_incipit.IncipitSourceRoute -v`
Expected: FAIL — `ModeProvenance` fails (no `source_note` yet); `/source` returns the build-help page or 404.

- [ ] **Step 3: Add provenance metadata to `MODE_META`**

In `incipit.py`, add these keys to the existing `MODE_META` entries (inside each dict, alongside `label`/`blurb`/`train_hint`):

`elements64` (after its `train_hint`, `incipit.py:69`):

```python
        "source_note": ("118 undisputed IUPAC facts as YAML, generated "
                        "byte-for-byte by the committed "
                        "teaching_data/build_elements.py."),
```

`dialogue` (after its `train_hint`, `incipit.py:81`):

```python
        "source_note": ("The same 118 IUPAC element facts, reformatted as "
                        "user:/model: Q&A turns, generated byte-for-byte by "
                        "the committed teaching_data/build_dialogue.py."),
```

`osai` (after its `train_hint`, `incipit.py:91`):

```python
        "source_note": ("OSAI-schema YAML. Trained on the real European Open "
                        "Source AI Index, this corpus is CC BY 4.0 -- cite "
                        "doi:10.5281/zenodo.15386042. The bundled sample_data "
                        "files are original, fictional stand-ins."),
        "source_url": "https://codeberg.org/AI-Technology-Assessment/main-database",
```

- [ ] **Step 4: Add `_send_html_error` and extend `_mode_cfg`**

In `incipit.py`, add a method to the `Handler` class (after `_send_json`, `incipit.py:251`):

```python
        def _send_html_error(self, status, msg):
            self._send(status, "text/html; charset=utf-8",
                       marginalia.html_error_page(status, msg))
```

Replace `_mode_cfg` (`incipit.py:253-267`) with the HTML-capable version:

```python
        def _mode_cfg(self, mode, html_errors=False):
            """Resolve a mode id to a usable config, or send the right error
            (400 unknown / 503 absent) and return None. JSON errors for the
            /api/ routes; HTML for the browser-facing /corpus and /source."""
            if mode not in modes:
                msg = f"unknown mode: {mode!r}"
                if html_errors:
                    self._send_html_error(400, msg)
                else:
                    self._send_json({"error": msg}, status=400)
                return None
            cfg = modes[mode]
            if cfg.get("model") is None:
                hint = cfg.get("train_hint", "")
                msg = f"no trained model for '{mode}' -- run `{hint}` first"
                if html_errors:
                    self._send_html_error(503, msg)
                else:
                    self._send_json({"error": msg}, status=503)
                return None
            return cfg
```

(Existing callers pass a positional `mode` and get JSON errors — unchanged.)

- [ ] **Step 5: Add the `/source` route**

In `incipit.py` `do_GET` (`incipit.py:354`), add the branch **before** `elif parsed.path.startswith("/api/")`:

```python
            elif parsed.path == "/source":
                qs = urllib.parse.parse_qs(parsed.query)
                mode = qs.get("mode", [default_mode])[0]
                cfg = self._mode_cfg(mode, html_errors=True)
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
                files = cfg.get("files", ())
                text = next((t for name, t in files if name == filename), None)
                if text is None:
                    self._send_html_error(
                        404, f"no file named {filename!r} in this mode's corpus")
                    return
                body = marginalia.source_page(
                    cfg.get("label", ""), filename, text, line=line,
                    note=cfg.get("source_note", ""),
                    url=cfg.get("source_url", ""),
                    sha=marginalia.corpus_sha256(files) if files else "")
                self._send(200, "text/html; charset=utf-8", body.encode("utf-8"))
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m unittest test_incipit -v`
Expected: PASS (full suite, including `ModeProvenance` and `IncipitSourceRoute`)

- [ ] **Step 7: Commit**

```bash
git add incipit.py test_incipit.py
git commit -m "feat: add /source route + corpus provenance to Incipit

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Incipit `/corpus` route

**Files:**
- Modify: `incipit.py` (add `/corpus` branch in `do_GET`, before the `/source` branch from Task 4)
- Test: `test_incipit.py` (add `IncipitCorpusRoute` class)

**Interfaces:**
- Consumes: `marginalia.corpus_index_page` (Task 2), `marginalia.corpus_sha256`, `_mode_cfg(html_errors=)` (Task 4), `INCIPIT_SOURCE_FILES` (Task 4).

- [ ] **Step 1: Write the failing tests**

Add to `test_incipit.py` (after `IncipitSourceRoute`):

```python
class IncipitCorpusRoute(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = _ServerFixture(
            _make_modes(_make_model(), files=INCIPIT_SOURCE_FILES))

    @classmethod
    def tearDownClass(cls):
        cls.server.close()

    def test_lists_files_as_html(self):
        status, headers, body = self.server.get("/corpus?mode=elements64")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
        page = body.decode("utf-8")
        self.assertIn("file=q.txt", page)
        self.assertIn("file=evil.txt", page)

    def test_footer_has_provenance_note(self):
        _, _, body = self.server.get("/corpus?mode=elements64")
        self.assertIn(b"build_elements.py", body)

    def test_unknown_mode_400_html(self):
        status, headers, _ = self.server.get("/corpus?mode=nope")
        self.assertEqual(status, 400)
        self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")

    def test_absent_model_503_html(self):
        status, headers, _ = self.server.get("/corpus?mode=dialogue")
        self.assertEqual(status, 503)
        self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")

    def test_not_swallowed_by_static(self):
        _, _, body = self.server.get("/corpus?mode=elements64")
        self.assertNotIn(b"npm run build", body)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest test_incipit.IncipitCorpusRoute -v`
Expected: FAIL — `/corpus` hits `_serve_static` and returns the build-help page / 404.

- [ ] **Step 3: Add the `/corpus` route**

In `incipit.py` `do_GET`, add **before** the `/source` branch (Task 4) so both sit before `startswith("/api/")`:

```python
            elif parsed.path == "/corpus":
                qs = urllib.parse.parse_qs(parsed.query)
                mode = qs.get("mode", [default_mode])[0]
                cfg = self._mode_cfg(mode, html_errors=True)
                if cfg is None:
                    return
                files = cfg.get("files", ())
                body = marginalia.corpus_index_page(
                    cfg.get("label", ""), files, mode,
                    note=cfg.get("source_note", ""),
                    url=cfg.get("source_url", ""),
                    sha=marginalia.corpus_sha256(files) if files else "")
                self._send(200, "text/html; charset=utf-8", body.encode("utf-8"))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest test_incipit -v`
Expected: PASS (full suite)

- [ ] **Step 5: Commit**

```bash
git add incipit.py test_incipit.py
git commit -m "feat: add /corpus browse route to Incipit

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Incipit front-end entry points

**Files:**
- Modify: `incipit/src/components/CharInspector.tsx` (add `mode` prop; link the `file:line`; add a browse link)
- Modify: `incipit/src/App.tsx:153-157` (pass `mode={mode}` to `CharInspector`)

**Interfaces:**
- Consumes: the `/source` and `/corpus` routes (Tasks 4, 5); `ModeInfo` type from `../types`; the `mode` value already computed in `App.tsx` and passed to `TapePanel` (`App.tsx:139`).

- [ ] **Step 1: Add the `mode` prop and links in `CharInspector.tsx`**

Update the import at `CharInspector.tsx:1`:

```tsx
import type { CharRecord, ModeInfo, SaliencyCell, Sampling, TurnResponse } from '../types'
```

Add `mode` to the props type (`CharInspector.tsx:13-20`):

```tsx
export function CharInspector(props: {
  record: CharRecord | null
  saliency: SaliencyCell[] | null
  response: TurnResponse | null
  recordSets: CharRecord[][]
  sampling: Sampling
  onSampling: (s: Sampling) => void
  mode: ModeInfo | null
}) {
  const { record, saliency, response, mode } = props
```

Replace the source-echo block (`CharInspector.tsx:104-111`):

```tsx
      {response?.source.matched && (
        <>
          <h3>Source echo (ground truth)</h3>
          <p className="muted">
            {mode ? (
              <a
                href={`/source?mode=${encodeURIComponent(mode.id)}&file=${encodeURIComponent(response.source.file)}&line=${response.source.line}#L${response.source.line}`}
                target="_blank"
                rel="noopener"
              >
                {response.source.file}:{response.source.line}
              </a>
            ) : (
              <>{response.source.file}:{response.source.line}</>
            )}
            {' — '}“…{response.source.pre}<b>{response.source.match}</b>
            {response.source.post}…”
          </p>
          {mode && (
            <p className="muted">
              <a href={`/corpus?mode=${encodeURIComponent(mode.id)}`} target="_blank" rel="noopener">
                browse the full corpus →
              </a>{' '}— read every training file, no prompt needed
            </p>
          )}
        </>
      )}
```

- [ ] **Step 2: Pass `mode` from `App.tsx`**

Update the `CharInspector` invocation (`App.tsx:153-157`):

```tsx
      <CharInspector
        record={focusedRecord} saliency={saliency} response={focusedResponse}
        recordSets={Object.values(modelData).map(d => d.records)}
        sampling={sampling} onSampling={setSampling} mode={mode}
      />
```

- [ ] **Step 3: Typecheck + build**

Run: `cd incipit && npm run build`
Expected: `tsc --noEmit` passes (no type errors — confirms `ModeInfo` has an `id` field and `mode` is threaded), then `vite build` writes `incipit/dist/`.

- [ ] **Step 4: Run front-end unit tests**

Run: `cd incipit && npm run test`
Expected: PASS — `tapeUtils` tests unaffected (no logic changed).

- [ ] **Step 5: Manually verify in the running app**

Run (from repo root): `python incipit.py` (needs `elements_k64.npz`; the app degrades gracefully otherwise). Open `http://127.0.0.1:8790`, send a turn, click a generated character, and confirm the "Source echo" `file:line` is now a link opening `/source`, and the "browse the full corpus →" link opens `/corpus?mode=<active>`.

- [ ] **Step 6: Commit**

```bash
git add incipit/src/components/CharInspector.tsx incipit/src/App.tsx
git commit -m "feat: link source echo + corpus browser in Incipit UI

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Documentation reconciliation

**Files:**
- Modify: `README.md` (Marginalia section ~`README.md:166`; Incipit section)
- Modify: `CLAUDE.md` (`marginalia.py` and `incipit.py` file-map entries)

**Interfaces:** none (docs only).

- [ ] **Step 1: Update the README Marginalia section**

In `README.md`, in the paragraph describing Marginalia's source view (~line 166, near "When a prompt matches the corpus verbatim, the source file**"), append a sentence documenting the promptless browser:

```markdown
A persistent **"browse the full corpus"** link opens `/corpus` — a zero-JS,
per-mode index of every training file (name, line count, char count), each
drilling into `/source`. This makes the "read the entire corpus" claim literal
in the tool: you no longer need a verbatim prompt match to reach a file.
```

- [ ] **Step 2: Update the README Incipit section**

In the Incipit section, add a sentence noting Incipit now serves its own browser:

```markdown
Incipit serves its own `/source` and `/corpus` routes (reusing Marginalia's
renderers) so all three of its corpora — including the dialogue corpus that has
no Marginalia mode — are browsable in-app.
```

- [ ] **Step 3: Update the CLAUDE.md file map**

In `CLAUDE.md`, extend the `marginalia.py` entry to mention: the shared
`corpus_index_page` renderer, the `/corpus` promptless browse route, and the
persistent "browse the full corpus" entry point. Extend the `incipit.py` entry
to mention: its new `/source` and `/corpus` routes (reusing
`marginalia.source_page` / `corpus_index_page`), and the `source_note`/
`source_url` added to its `MODE_META`. Add one line under "do NOT fix these" or
the file map noting the char-total convention: `/corpus`'s total is the
canonical PAD-joined length (matches `colophon.json` `num_characters`), so it
reads slightly higher than the sum of per-row counts — the boundary tokens are
real training input, not a bug.

- [ ] **Step 4: Verify the full gate suite**

Run:
```bash
python -m unittest test_marginalia test_incipit test_colophon test_elements test_kana test_dialogue
cd incipit && npm run test && npm run build && cd ..
```
Expected: all Python suites pass (report the counts explicitly); vitest passes; build succeeds.

- [ ] **Step 5: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: document the corpus browser in README + CLAUDE.md

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notes for the implementer

- **Run tests before every commit** and report counts explicitly ("N passing, 0 failing"), per the repo's discipline — not "tests pass" by hand-wave.
- **Do not push to `main`.** Work stays on the current feature branch; open a PR at the end.
- **Verify branch/worktree first:** `git rev-parse --abbrev-ref HEAD && pwd`.
- Tasks 1→2→3 are Marginalia; Task 4 depends on Task 1 (`html_error_page`); Task 5 depends on Tasks 2 + 4; Task 6 depends on Tasks 4 + 5; Task 7 is last. Implement in order.
