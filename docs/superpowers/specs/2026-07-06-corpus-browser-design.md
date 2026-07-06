# Corpus browser — a promptless "read the entire corpus" view

**Date:** 2026-07-06
**Status:** design approved, ready for implementation plan
**Scope:** Marginalia + Incipit (both servers), shared renderer in `marginalia.py`

## Problem

Colophon's central pitch, stated three times in the README, is that a reader can
**read the entire corpus**:

- *"Here you can **read the entire corpus**. Nothing else exists to the model."*
  (`README.md:18`)
- *"Training-data composition — Known exactly — **you can read the corpus**"*
  (`README.md:44`)
- *"The contrast **is** the pitch."*

But the shipped tools don't actually deliver that in-product. The only path to a
corpus file is `GET /source?mode=&file=<exact-name>` (`marginalia.py:1055`), and
the UI only ever *emits* that link when `find_source_echo()` (`marginalia.py:234`)
matches what the user typed **verbatim** into a file. No match → no link. There
is **no index, no listing, no browse affordance**. A user cannot enumerate or
read the corpus unless they type text that already appears in it.

Incipit is worse off: it reuses `marginalia.find_source_echo` for the echo
snippet (`incipit.py:194`) but has **no `/source` route at all** — the matched
`file:line` in `CharInspector.tsx:107` is inert text.

So "read the entire corpus" is true at the filesystem level (`--src` is right
there) but false in the tool whose entire job is to make the ground truth
legible. That gap lands squarely on the project's thesis.

## Goal

Add a promptless, per-corpus **browse** view to both servers so a reader can list
and read every training file the model actually trained on — independent of what
they typed. Close the README promise-vs-product gap. Keep every existing
invariant (stdlib + NumPy servers, zero-JS served pages, exact-name in-memory
lookup, escaping, honest provenance).

### Non-goals (YAGNI)

- **No full-text content search.** It overlaps `find_source_echo` and adds attack
  surface for little gain.
- **No inlining of file contents on the index page.** The index lists files and
  drills into the existing `/source`; the 196-file / 579 KB real index stays a
  light page.
- **No cross-app linking.** Incipit gets its own routes (see "Rejected
  alternatives"); it does not link across to Marginalia.
- **No `/api/modes` payload change.** Front-ends construct `/corpus?mode=<active>`
  directly from the mode they already track.

## Approach (chosen: A — one shared renderer)

One new rendering function in `marginalia.py`, called by thin route handlers in
both servers. This leans on the fact that `incipit.py` already does
`import marginalia` (`incipit.py:42`) and already reuses `find_source_echo` /
`source_page`.

### Rejected alternatives

- **B — overload `/source` (no `file` ⇒ index).** Conflates "one file" and "the
  whole list" on one path and changes today's semantics (empty `file=` currently
  404s). More surprising, harder to test. Rejected.
- **C — Marginalia-only page, Incipit cross-links ("links over").** The two apps
  share neither mode ids nor corpora: Incipit's `dialogue` corpus
  (`teaching_data/dialogue`) is **not loaded in Marginalia at all**, and it is
  Act 3's entire point. A cross-link would break exactly where it matters most,
  and would couple Incipit (:8790) to Marginalia (:8765) being up. Rejected in
  favor of Incipit growing its own routes — cheap, because `marginalia` is
  already imported and Incipit already holds all three corpora in memory.

## Design

### 1. Shared renderer

Add to `marginalia.py`, as a sibling of `source_page()`:

```
corpus_index_page(label, files, mode, note="", url="", sha="") -> str
```

- Zero-JS server-rendered HTML. Reuses the existing `_SOURCE_CSS` and the same
  provenance-footer construction as `source_page()`.
- Header mirrors `source_page`: `❧ {label} — corpus`, with a muted intro:
  *"Every file the model actually trained on, served from the in-memory corpus —
  ground truth, not a link out."*
- Body: one table row per file (files arrive already sorted by
  `load_corpus_files()`), each row showing:
  - **filename** as a link → `/source?mode={mode}&file={name}`
  - **line count** (`len(text.splitlines())` — matches the numbered rows
    `source_page` renders when you drill in, `marginalia.py:209`)
  - **char count** (raw per-file `len(text)`)
- A summary line: **"N files · M characters"** where **M is the canonical
  count** — `len(("\n" + colophon.PAD + "\n").join(texts))`, the same
  PAD-joined length `load_corpus` feeds the model and that `colophon.json`
  records as `num_characters` (`colophon.py:73`, `:96`), and the same basis as
  the sha256 in the footer (`corpus_sha256`). So `/corpus`, `colophon.json`, and
  the footer sha all describe one corpus string and cross-check cleanly.
  **Important:** M is therefore *slightly larger* than the sum of the per-row
  char counts — by the `\n␀\n` boundary tokens the model sees between entries
  (`3 × (N−1)` chars). A one-line note on the page states this: *"Totals count
  the corpus as the model sees it, including the `\n␀\n` boundary token between
  entries; per-file counts are raw file lengths."* The row-vs-total seam is a
  real teaching point — the boundaries are genuine training input — not a bug.
- Footer: identical to `source_page`'s (`note` / `url` / `sha`). For the real
  index that is the CC-BY note + codeberg URL + corpus sha256.
- **Everything corpus/user-derived is `html.escape()`d**: filenames, counts,
  note, url, sha. (`corpus_index_page` never renders file *contents* — that's
  `/source`'s job — but filenames are still escaped.)
- **Empty corpus (`files == []`) → an honest empty page** ("0 files · 0
  characters"), never a crash.

### 2. Shared error page

Marginalia's `_send_html_error` (`marginalia.py:969`) currently builds the error
HTML inline. Extract its body into a module-level helper so both servers share
one error renderer:

```
html_error_page(status, msg) -> bytes
```

Marginalia's `_send_html_error` method then calls it (no behavior change —
covered by a regression test). Incipit gains a thin `_send_html_error` that calls
the same helper.

### 3. Routes

A new `GET /corpus?mode=` in **both** servers, plus a new `GET /source` in
Incipit (Marginalia already has one).

**Marginalia** (`do_GET`, alongside the existing `/source` branch):
- `/corpus`: resolve mode via `_mode_cfg(qs, html_errors=True)`, then
  `corpus_index_page(cfg["label"], cfg["files"], mode, note, url,
  sha=corpus_sha256(files) if files else "")` using `MODE_META`'s
  `source_note` / `source_url`.

**Incipit** (`do_GET`, added **before** the `startswith("/api/")` /
`_serve_static` fallthrough — `/corpus` and `/source` are not under `/api/`, so
ordering matters or static serving swallows them):
- `_mode_cfg` gains an `html_errors=False` param mirroring Marginalia's.
- `/corpus`: parse `?mode=`, resolve with `html_errors=True`, call
  `marginalia.corpus_index_page(...)` over the mode's in-memory `files`.
- `/source`: parse `?mode=`, `?file=`, `?line=`; resolve with
  `html_errors=True`; exact-name in-memory lookup (`next((t for name, t in
  files if name == filename), None)`); 404 HTML if absent; call
  `marginalia.source_page(...)`. Mirrors Marginalia's `/source` handler
  (`marginalia.py:1055`) — including the `dialogue` corpus.

**Gating (deliberate):** both new routes reuse `_mode_cfg`, so they 400 on an
unknown mode and **503 on an absent model** — matching today's `/source` and the
"absent weights → disabled toggle + 503" convention. An absent mode is
unreachable from the UI anyway, so gating on the model (rather than adding a
files-only gate) keeps one code path. Empty corpus (files loaded, but none) →
honest empty page, not an error.

### 4. Provenance metadata

Incipit's `MODE_META` (`incipit.py:58`) has no `source_note` / `source_url`
today. Add them for all three modes so the footer is honest — the `dialogue`
mode especially needs one:
- `osai` → CC-BY note + codeberg URL (mirror Marginalia's `osai`).
- `elements64` → generator + IUPAC note (mirror Marginalia's `elements`).
- `dialogue` → generator note (`teaching_data/build_dialogue.py`; same 118 IUPAC
  facts reformatted as `user:`/`model:` turns).

### 5. Entry points

- **Marginalia** (vanilla JS / `INDEX_HTML` string): a persistent
  **"browse the full corpus →"** link in the source-in-training-data panel
  (`marginalia.py:433`) — shown always, not just on a verbatim match — carrying
  the active mode via the existing `activeMode` JS. HTML/JS-string edit; no
  build.
- **Incipit** (React): turn the inert `file:line` in
  `CharInspector.tsx:107` into a link to Incipit's own `/source`, and add a
  **"browse the full corpus →"** link to `/corpus?mode=<active>`. Requires
  `npm run build`; confined entirely to `incipit/src` (the sanctioned front-end —
  no violation of "the toolchain stops at `incipit/`"; `incipit/dist` is
  gitignored, so nothing committed churns).

## Data flow

```
User (browser tab)
  └─ GET /corpus?mode=elements
       └─ _mode_cfg → cfg["files"]  (already in memory; no disk read)
            └─ corpus_index_page(label, files, mode, note, url, sha)
                 └─ HTML list; each filename → /source?mode=elements&file=<name>

  └─ GET /source?mode=elements&file=element-026.yaml
       └─ _mode_cfg → cfg["files"]
            └─ exact-name lookup (traversal can only 404)
                 └─ source_page(label, filename, text, line, note, url, sha)
```

Both servers, same renderers. No new state; read-only GETs over the corpus
already loaded at startup by `marginalia._load_mode`.

## Error handling

| Condition | Response |
|---|---|
| Unknown `?mode=` | 400, **HTML** (renders in a browser tab), not JSON |
| Known mode, model absent | 503, HTML, with the mode's `train_hint` |
| `/source` unknown/traversal `file=` | 404, HTML |
| `/source` non-integer `line=` | 400, HTML |
| Empty corpus (files loaded, none present) | 200, honest "0 files" index |
| Incipit `/corpus`,`/source` before static fallthrough | matched, not a static 404 / missing-dist page |

## Security & invariants (explicitly preserved)

- **Path traversal impossible:** `/source` keeps exact-name in-memory lookup (no
  disk access at request time); `/corpus` only ever emits links to names that
  exist in memory. Both servers.
- **Escaping:** filenames, counts, and footer are `html.escape()`d; `/source`
  already escapes file contents. The real OSAI index is external CC-BY data, so
  this matters (global rule: never inline untrusted content without escaping).
- **No new dependencies:** stdlib + NumPy servers, zero JS on served pages. Node
  stays confined to `incipit/src`.
- **Attribution, not vendoring:** footer carries CC-BY note + codeberg URL +
  corpus sha256; files are read from `--src` at runtime.
- **Incipit statelessness untouched:** new routes are read-only GETs over the
  in-memory corpus; no session state added.

## Testing (TDD — failing tests first)

`test_marginalia.py` (stdlib `unittest`, live ephemeral server — existing style):
- `corpus_index_page()` unit: lists files with `/source?mode=&file=` links;
  "N files · M chars" summary where **M is the canonical PAD-joined length**
  (equals `colophon.data_manifest(...)["num_characters"]` for the same corpus,
  and is strictly greater than the sum of per-row char counts by `3 × (N−1)`);
  escapes filenames/footer; zero JS; empty corpus → honest empty page.
- `/corpus` route: 200 lists every file; `?mode=` selects the right corpus
  (osai vs elements vs kana); unknown mode → 400 HTML; absent model → 503 HTML;
  **round-trip** — every emitted `/source?…&file=` link returns 200.
- `html_error_page()` unit: regression guard that the extracted helper emits the
  same error body as before the refactor.

`test_incipit.py` (live ephemeral server):
- `/corpus` parity: 200 lists files for `elements64` / `dialogue` / `osai`;
  unknown → 400 HTML; absent → 503 HTML with train hint.
- `/source` (new): serves a file by exact name **including the `dialogue`
  corpus**; 404 on unknown/traversal name; 400 on bad `line`; 503 on absent
  mode; all HTML.
- Route ordering: `/corpus` and `/source` are not swallowed by `_serve_static`.
- Each mode's `/corpus` footer carries a provenance note.

`incipit/` (vitest): `tapeUtils` unchanged, so front-end unit tests are
unaffected; the CharInspector link is covered by `npm run build`
(`tsc --noEmit` + `vite build`) passing.

## Docs reconciliation

- **README**: the "read the entire corpus" claim (`README.md:18`, `:44`) becomes
  literally true in-product. Update the Marginalia section (~`README.md:166`) to
  document the `/corpus` browser + entry points, and the Incipit section to note
  its new `/source` + `/corpus`. Adding the view is the honest fix — do **not**
  soften the claim.
- **CLAUDE.md file map**: extend the `marginalia.py` and `incipit.py` entries to
  describe the shared `corpus_index_page`, the `/corpus` route in both servers,
  and Incipit's new `/source`.
- **colophon.json**: unchanged; the browse summary corroborates the existing
  datasheet aggregate.

## Build sequence (for the implementation plan)

1. `test_marginalia.py`: failing tests for `corpus_index_page`, `/corpus`,
   `html_error_page` regression.
2. `marginalia.py`: extract `html_error_page`; add `corpus_index_page`; add the
   `/corpus` route; add the Marginalia entry-point link. → tests green.
3. `test_incipit.py`: failing tests for `/corpus`, `/source`, ordering,
   provenance.
4. `incipit.py`: add `source_note`/`source_url` to `MODE_META`; add
   `_send_html_error` + `_mode_cfg(html_errors=)`; add `/corpus` and `/source`
   routes. → tests green.
5. `incipit/src/components/CharInspector.tsx`: link the `file:line`; add the
   browse link. `npm run build`.
6. Docs: README + CLAUDE.md updates.
7. Full gates: `python -m unittest`, `npm run test`, `npm run build`.

## Open questions

None outstanding — the dialogue-corpus mapping and the Incipit-route decision
were resolved during brainstorming (Incipit gets its own routes).
