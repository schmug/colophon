# Plan: Hiragana kana corpus + Marginalia third mode

Slug: `kana-teaching-corpus`
Status: approved (recommendations accepted 2026-07-03: を="wo", 71 kana, "Kana chart" label, steps measured during build)

## Problem

Both existing corpora (OSAI index, periodic table) are Latin-only, so the
flagship out-of-distribution demo — the Japanese prompt `日本語で書いてください` —
can only ever show characters *no* model in the suite has seen. A reader
can't see that "off-map" is a fact about the model–data pairing rather than
about the text itself. A third corpus in a disjoint script gives the suite a
model for which that same prompt is (partly) home turf: the four kanji stay
off-map while the six hiragana light up as known, per character, in
Marginalia's heatmap.

## Decisions (proposed)

1. **Corpus: 71 hiragana** — the 46 basic gojūon (incl. ん, を) **plus the 25
   dakuten/handakuten voiced kana** (が–ぽ). The voiced set is required for the
   flagship demo: で and だ in the canonical prompt are voiced; with basic-46
   only they'd read "unknown" and muddy the kanji-vs-kana split that is the
   whole payload. Excluded as archaic/out-of-scope: ゐ, ゑ, small kana
   (ゃゅょっ), katakana, kanji.
2. **Schema: two fields only** — `kana:` + `romaji:` (Hepburn). No `number:`
   / `row:` / `column:`: keeps digits out of the vocab entirely (maximal
   distinctness from the other corpora — shared tokens reduce to
   `[a-z :\n-]`), avoids fake overlap with elements' `number:` prompts, and
   sidesteps ん having no grid cell. Example file:

   ```yaml
   kana: し
   romaji: shi
   ```

3. **Layout mirrors `teaching_data/elements/`:** committed generator
   `teaching_data/build_kana.py` with one embedded 71-row chart-order table →
   per-kana YAML under `teaching_data/kana/`, files checked in and
   regenerable byte-for-byte. Filenames `NNN-<romaji>.yaml`; Hepburn
   collisions (じ/ぢ → ji, ず/づ → zu) keep identical romaji *values* (that is
   the Hepburn fact) and disambiguate in the filename only — filenames never
   enter the corpus.
4. **No colophon.py changes.** The existing `--out` plumbing (built for
   elements) trains to `kana.npz` without clobbering anything:
   `python colophon.py --src teaching_data/kana --out kana.npz --steps 4000 train`.
5. **Marginalia: paired flags, no generalization.** Add `--kana-npz` /
   `--kana-src` mirroring the elements flags, a third `MODE_META` entry, and a
   third entry in the sources dict (marginalia.py:1016). The toggle and
   example chips already render list-driven from `/api/modes`, so the
   frontend needs no structural change. OSAI stays the default mode.

## Scope

**In:** kana corpus + committed generator; `kana.npz` training path (via
existing flags); Marginalia kana mode (flags, `MODE_META`, routing, absent →
503/disabled toggle); `test_kana.py` + mode-routing additions to
`test_marginalia.py`; README "kana mode" section with Hepburn citation;
CLAUDE.md file-map update.

**Out:** no change to the MLP / transformer / confidence math or the OSAI
scorecard; no colophon.py changes; no katakana/kanji/small-kana corpus; no
N-mode CLI generalization in Marginalia; `colophon.py demo` stays OSAI-only;
no fourth corpus.

## Constraints

- **Dependencies:** numpy-only `requirements.txt`; Marginalia stays
  stdlib-only; torch never on the default path.
- **Encoding:** corpus loader and Marginalia file reads are already
  `encoding="utf-8"` (colophon.py:68, marginalia.py:148). Python strings are
  per-codepoint, so kana are single vocab entries for free.
- **Absent-mode invariant:** a mode whose weights are missing degrades to a
  disabled toggle + 503, never a crash — must hold for kana exactly as for
  elements.
- **Vocab hygiene:** no provenance comments inside data files (they pollute
  the tiny vocab) — attribution lives in the generator + README, per the
  elements precedent.
- **Undisputed facts only:** one romanization standard (Hepburn), cited in
  README the way IUPAC is; archaic kana excluded.
- **On-theme overfitting:** the corpus is ~1.5 KB; the model will memorize
  it. Do not add regularization — same stance as the 3-file sample.
- **Keep it honest:** present kana mode as "a model whose chart these
  characters are on," never "the model knows Japanese." The lesson is that
  off-map is relative, not that competence transferred.
- **Lesson from the elements spec (dropped criterion):** do not write
  entropy-magnitude acceptance criteria — the original
  `entropy(26) < entropy(250)` criterion contradicted the thesis and passed
  by luck. All kana criteria key on the categorical off-map flag and literal
  fact reproduction, not entropy comparisons.
- **Process:** TDD; branch + PR, never push to main; conventional commits.

## Acceptance criteria

1. **Corpus contract:** `python colophon.py --src teaching_data/kana prepare`
   loads 71 entries; vocab is exactly the 71 kana plus a subset of
   `[a-z :\n-]` (no digits, no uppercase, no CJK beyond the 71); data section
   records num_files / num_characters / sha256. Running
   `teaching_data/build_kana.py` reproduces the checked-in files
   byte-for-byte. (test)
2. **Fact reproduction:** a low-temperature continuation from `kana: し\n` on
   the trained kana model contains `romaji: shi` — checkable on any chart.
   (test)
3. **The flagship split:** on the kana model,
   `analyze_prompt("日本語で書いてください")` sets the off-map flag with unknown
   characters exactly {日, 本, 語, 書}, while で・い・て・く・だ・さ are
   in-vocab; the same prompt on the OSAI model reports 10 distinct unknown
   characters. (test)
4. **Isolation + routing:** training with `--out kana.npz` leaves
   `colophon.npz` and `elements.npz` untouched; `/api/modes` lists kana with
   availability; `/api/analyze?mode=kana` and `/api/saliency?mode=kana` route
   to the kana model; absent kana weights → 503 with a clear message and a
   disabled toggle; unknown mode → 400 unchanged. (test)
5. **Suites green:** full `python -m unittest` passes — existing
   `test_colophon.py` / `test_marginalia.py` / `test_elements.py` plus new
   `test_kana.py`; report exact counts.

## Open questions

1. **を romaji:** "wo" (traditional chart form, what a learner's chart says)
   vs "o" (modified Hepburn, particle pronunciation). Recommend **"wo"** with
   a one-line README note.
2. **71 vs purist 46:** recommended 71 (above). The 46-only alternative makes
   で/だ read as unknown — a subtler three-way lesson (kanji unknown, voiced
   kana unknown, basic kana known) that risks confusing more than it teaches.
   Confirm 71.
3. **Mode copy:** proposed label "Kana chart"; example chips: a real lookup
   (`kana: し`), a reverse prompt (`romaji: a`), and the canonical mixed
   prompt labeled "Half on the chart, half off" (`日本語で書いてください`).
   Wordsmithing open.
4. **Training steps for the hint:** elements uses `--steps 4000`; kana corpus
   is smaller, so 4000 is likely fine — measure during implementation and put
   the measured number in `train_hint` and the README.

## Build sequence (TDD)

1. `teaching_data/build_kana.py` from an embedded, Hepburn-cited 71-row
   table → per-kana YAML. Test: 71 files, schema, exact vocab, byte-for-byte
   regeneration.
2. Train `kana.npz` via existing `--out`; pin criteria 2–3 in `test_kana.py`
   (fact reproduction, flagship split, isolation).
3. Marginalia: flags + `MODE_META` + sources entry; extend
   `test_marginalia.py` mode-routing tests (kana routes, 503 when absent,
   400 unknown unchanged).
4. Frontend sanity pass (toggle should pick up the third mode with no
   structural change — verify).
5. README kana-mode section + Hepburn citation; CLAUDE.md file map.
6. Full `python -m unittest`; report counts.
