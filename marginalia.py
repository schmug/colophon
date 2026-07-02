#!/usr/bin/env python3
"""
marginalia.py -- Marginalia, the live inspection UI for Colophon.

A small local web page: type a prompt, watch the model's own white-box signals
react in real time -- next-char entropy, the categorical off-map/unknown-char
flag, and the OSAI openness scorecard -- against a trained `colophon.npz`. It
makes the auditability `colophon.py demo` already prints interactive instead
of a one-shot report.

Marginalia adds ZERO runtime dependencies: the server is Python's stdlib
`http.server`, and the frontend is a single vanilla-JS page with no build step
or CDN script. `prompt_confidence()` and `scorecard_section()` from colophon.py
are the source of truth -- nothing here re-derives those signals.

Usage:
  python colophon.py demo          # train a model first (writes colophon.npz)
  python marginalia.py             # serve the UI at http://127.0.0.1:8765
  python marginalia.py --port 9000 --npz /path/to/colophon.npz
"""

from __future__ import annotations
import argparse, glob, hashlib, http.server, json, os, urllib.parse
import numpy as np

import colophon

MAX_PROMPT_LEN = 500     # bound the per-keystroke forward-pass cost
CONTINUATION_LEN = 80    # sampled chars shown after the typed prompt
SOURCE_SUFFIX_FLOOR = 4  # shortest suffix worth searching for (else everything matches)
SOURCE_SUFFIX_CAP = 64   # longest suffix worth searching for (bounds per-keystroke cost)
SOURCE_CONTEXT_CHARS = 40  # chars of corpus context shown on each side of a match


REQUIRED_KEYS = ("chars", "C", "W1", "b1", "W2", "b2")

# The two corpora the page can switch between. OSAI is the flagship (carries the
# self-referential openness argument); elements is the layperson on-ramp (ground
# truth already in the reader's head, so the signals can be checked against what
# they already know). Copy lives here, server-side, so /api/modes drives the
# page and nothing is re-authored in JavaScript.
MODE_META = {
    "osai": {
        "label": "Openness index",
        "blurb": ("The flagship corpus — the European Open Source AI Index. Real "
                  "and rich, but jargon you can't grade by eye, so you have to "
                  "take the confidence signals on faith."),
        "examples": [
            ("Something it trained on", "weights_basemodel:"),
            ("Characters it’s never seen", "日本語"),
            ("A topic outside its data", "the 2027 election"),
        ],
        "train_hint": "python colophon.py demo",
    },
    "elements": {
        "label": "Periodic table",
        "blurb": ("A teaching corpus of facts you already know: 118 elements. Ask "
                  "for a real one and check the answer yourself; ask for a made-up "
                  "“number: 250” and watch it stay just as “sure” "
                  "— the confidence number can't tell you it's inventing. Only "
                  "the off-map flag can."),
        "examples": [
            ("A real element (26 = Iron)", "number: 26\n"),
            ("A made-up element", "number: 250\n"),
            ("Characters it’s never seen", "日本語"),
        ],
        "train_hint": ("python colophon.py --src teaching_data/elements "
                       "--out elements.npz --steps 4000 train"),
    },
}
DEFAULT_MODE = "osai"


def load_model(npz_path: str):
    """Mirrors the .npz load path in colophon.cmd_generate().

    Raises KeyError if the archive is missing a required array and ValueError
    if the embedding width is zero (which would make the K inference divide by
    zero). main() degrades gracefully on both, like the missing-file case."""
    d = np.load(npz_path, allow_pickle=True)
    missing = [k for k in REQUIRED_KEYS if k not in d.files]
    if missing:
        raise KeyError(f"{npz_path} is missing required array(s): {', '.join(missing)}")
    chars = list(d["chars"])
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    p = {k: d[k] for k in ("C", "W1", "b1", "W2", "b2")}
    embed_dim = p["C"].shape[1] if p["C"].ndim == 2 else 0
    if embed_dim == 0:
        raise ValueError(f"{npz_path} has a zero-width embedding (C shape {p['C'].shape})")
    K = p["W1"].shape[0] // embed_dim
    return p, stoi, itos, K


def confidence_readout(entropy, unknown, has_prompt=True):
    """Translate colophon's raw normalized entropy into a layperson reading.

    This is a *presentation* transform of the same white-box signal, not a new
    or independent estimate -- the raw entropy is still returned alongside so
    the page can show it underneath. Entropy runs the opposite way from
    confidence (0 = certain, 1 = no idea), so confidence% is (1 - entropy)*100.

    Two things are deliberate and load-bearing (see CLAUDE.md):
      * Empty prompt -> no number. entropy is 0.0 for an empty prompt, which
        would otherwise read as a misleading "100% sure".
      * The off-map flag, NOT the friendly percentage, is the trustworthy tell.
        A prompt of never-seen characters can still report a moderate
        confidence because the entropy signal under-reacts out of distribution,
        so the verdict overrides it and tells the reader to ignore the number.
    """
    if not has_prompt:
        return {"confidence_pct": None, "verdict_level": "none",
                "verdict": "Type something to see how sure the model is."}

    pct = max(0, min(100, round((1.0 - entropy) * 100)))
    if unknown:
        n = len(unknown)
        return {
            "confidence_pct": pct,
            "verdict_level": "off-map",
            "verdict": (f"It reads {pct}% “sure” — but it has never "
                        f"seen {n} character{'s' if n != 1 else ''} here, so ignore "
                        f"that number: this is off the map and it’s guessing blind."),
        }
    if pct >= 75:
        level, msg = "confident", "Confident — this looks like the data it was trained on."
    elif pct >= 50:
        level, msg = "unsure", "Unsure — this is only loosely like its training data."
    else:
        level, msg = "struggling", "Struggling — this is unlike most of what it saw in training."
    return {"confidence_pct": pct, "verdict_level": level,
            "verdict": f"{pct}% sure. {msg}"}


def load_corpus_files(src_dir: str):
    """Mirrors colophon.load_corpus()'s file discovery but keeps each file's
    raw text separate (never PAD-joined), so the suffix search in
    find_source_echo() can never match across an entry boundary."""
    paths = sorted(glob.glob(os.path.join(src_dir, "*.yaml")) +
                   glob.glob(os.path.join(src_dir, "*.yml")))
    files = []
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as f:
            files.append((os.path.basename(path), f.read()))
    return files


def corpus_sha256(files):
    """The same PAD-joined hash colophon.data_manifest() records in
    colophon.json, computed from the per-file texts load_corpus_files()
    already read (no re-globbing, no re-reading the source directory)."""
    text = ("\n" + colophon.PAD + "\n").join(text for _, text in files)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def find_source_echo(files, prompt, floor=SOURCE_SUFFIX_FLOOR, cap=SOURCE_SUFFIX_CAP,
                      context=SOURCE_CONTEXT_CHARS):
    """Literal longest-matching-suffix search over `files` (a list of
    (name, text) pairs, one per corpus entry). Backs off from the longest
    suffix of `prompt` (capped at `cap` chars) down to `floor` chars, and
    returns the first hit -- the longest suffix that appears verbatim in any
    file, searched file-by-file so a match can never span an entry boundary.
    Returns {"matched": False} if nothing >= floor chars matches."""
    max_len = min(len(prompt), cap)
    for length in range(max_len, floor - 1, -1):
        suffix = prompt[-length:]
        for name, text in files:
            idx = text.find(suffix)
            if idx == -1:
                continue
            line = text.count("\n", 0, idx) + 1
            return {
                "matched": True,
                "file": name,
                "line": line,
                "pre": text[max(0, idx - context):idx],
                "match": suffix,
                "post": text[idx + length:idx + length + context],
            }
    return {"matched": False}


def analyze_prompt(p, stoi, itos, K, prompt, files=(), n=CONTINUATION_LEN, seed=0):
    """The one function the HTTP layer calls: entropy + off-map signal from
    prompt_confidence(), a sampled continuation from generate(), and a
    source-echo match from find_source_echo(), and a layperson confidence
    readout. The first two are colophon.py's own functions, called as-is;
    confidence_readout() only reframes entropy for display."""
    entropy, unknown = colophon.prompt_confidence(p, stoi, K, prompt)
    full = colophon.generate(p, stoi, itos, K, prompt=prompt, n=n, seed=seed)
    return {
        "prompt": prompt,
        "entropy": entropy,
        "unknown_chars": unknown,
        "off_map": bool(unknown),
        "continuation": full[len(prompt):],
        **confidence_readout(entropy, unknown, has_prompt=bool(prompt)),
        "source": find_source_echo(files, prompt),
    }


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Marginalia -- live inspection UI for Colophon</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: ui-monospace, Menlo, Consolas, monospace; max-width: 860px;
         margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }
  h1 { font-size: 1.3rem; }
  h1 .glyph { opacity: .6; }
  textarea { width: 100%; min-height: 4rem; font: inherit; padding: .5rem;
             box-sizing: border-box; }
  .panel { border: 1px solid #8884; border-radius: 6px; padding: .75rem 1rem;
           margin: 1rem 0; }
  .panel h2 { font-size: .95rem; margin: 0 0 .4rem; }
  .conf-pct { font-size: 1.6rem; font-weight: bold; }
  .verdict { margin: .35rem 0 .1rem; }
  .verdict.confident { color: #2a6; }
  .verdict.unsure { color: #c90; }
  .verdict.struggling, .verdict.off-map { color: #d33; }
  .verdict.none { opacity: .6; }
  .conf-bar { height: 10px; border-radius: 5px; background: #8882;
              overflow: hidden; margin-top: .4rem; }
  .conf-fill { height: 100%; background: linear-gradient(90deg, #d63, #2a6);
               transition: width .15s ease; }
  .raw-entropy { margin-top: .35rem; }
  .offmap { font-weight: bold; }
  .offmap.ok { color: #2a6; }
  .offmap.flagged { color: #d33; }
  .mode-toggle { display: flex; flex-wrap: wrap; gap: .4rem; margin: .6rem 0 .2rem; }
  .mode-toggle button { font: inherit; font-size: .85rem; cursor: pointer;
                        border: 1px solid #8886; border-radius: 6px;
                        padding: .3rem .8rem; background: #8881; color: inherit; }
  .mode-toggle button.active { background: #2a63; border-color: #2a6;
                               font-weight: bold; }
  .mode-toggle button:disabled { opacity: .45; cursor: not-allowed; }
  .examples { margin: .5rem 0 0; display: flex; flex-wrap: wrap; gap: .4rem; }
  .examples button { font: inherit; font-size: .8rem; cursor: pointer;
                     border: 1px solid #8886; border-radius: 999px;
                     padding: .2rem .7rem; background: #8881; color: inherit; }
  .examples button:hover { background: #8883; }
  .continuation .prompt-part { opacity: .55; }
  .continuation .cont-part { font-weight: bold; }
  table { border-collapse: collapse; width: 100%; font-size: .85rem; }
  th, td { border: 1px solid #8884; padding: .3rem .5rem; text-align: left; }
  td.open { color: #2a6; } td.partial { color: #c90; } td.closed { color: #d33; }
  .error { color: #d33; }
  .muted { opacity: .6; font-size: .85rem; }
</style>
</head>
<body>
<h1><span class="glyph">&#10087;</span> Marginalia</h1>
<p class="muted">A tiny language model whose every honesty signal &mdash; how sure it
is, whether it has ever seen these characters, what it would write next &mdash; is
computed by its own weights, not guessed at in JavaScript. Pick a corpus:</p>

<div class="mode-toggle" id="mode-toggle"></div>
<p class="muted" id="mode-blurb">&nbsp;</p>

<textarea id="prompt" aria-label="Prompt" placeholder="Type here, or tap an example below&hellip;" autofocus></textarea>

<div class="examples" id="examples"></div>

<div class="panel">
  <h2>How sure is the model about what comes next?</h2>
  <div><span class="conf-pct" id="conf-pct">--</span></div>
  <div class="verdict none" id="verdict">Type something to see how sure the model is.</div>
  <div class="conf-bar"><div class="conf-fill" id="conf-fill" style="width:0%"></div></div>
  <div class="raw-entropy muted">raw signal: entropy <b id="entropy-val">--</b>
    (0&nbsp;=&nbsp;certain, 1&nbsp;=&nbsp;no idea) &mdash; the number under the hood, shown so you can audit it</div>
</div>

<div class="panel">
  <h2>Has the model seen these characters before?</h2>
  <div id="offmap" class="offmap ok">no off-map characters</div>
</div>

<div class="panel">
  <h2>What the model writes if it keeps going</h2>
  <div id="continuation" class="continuation">&nbsp;</div>
  <div class="muted" style="margin-top:.35rem">sampled from the model's own weights (a little randomness, so it varies)</div>
</div>

<div class="panel">
  <div class="muted">source in training data (literal longest-suffix match, ground truth):</div>
  <div id="source-label" class="muted">&nbsp;</div>
  <div id="source-snippet" class="continuation"></div>
</div>

<div class="panel" id="scorecard-panel">
  <h2>What this model does and doesn't disclose</h2>
  <p class="muted" style="margin:.2rem 0 .6rem">The openness index scores AI systems
  on what they reveal. Here's how this artifact grades against a typical closed model
  &mdash; not "open is better," just what is and isn't on the table.</p>
  <table id="scorecard"><tbody></tbody></table>
</div>

<div id="error" class="error"></div>

<script>
const promptEl = document.getElementById('prompt');
const confPctEl = document.getElementById('conf-pct');
const verdictEl = document.getElementById('verdict');
const confFill = document.getElementById('conf-fill');
const entropyVal = document.getElementById('entropy-val');
const offmapEl = document.getElementById('offmap');
const continuationEl = document.getElementById('continuation');
const sourceLabelEl = document.getElementById('source-label');
const sourceSnippetEl = document.getElementById('source-snippet');
const errorEl = document.getElementById('error');

const VERDICT_LEVELS = ['confident', 'unsure', 'struggling', 'off-map', 'none'];

let debounceTimer = null;

function renderAnalysis(data) {
  errorEl.textContent = '';

  const pct = data.confidence_pct;
  confPctEl.textContent = pct === null ? '--' : pct + '% sure';
  confFill.style.width = (pct === null ? 0 : Math.min(100, Math.max(0, pct))) + '%';
  verdictEl.textContent = data.verdict;
  VERDICT_LEVELS.forEach(l => verdictEl.classList.toggle(l, l === data.verdict_level));
  entropyVal.textContent = data.entropy.toFixed(3);

  offmapEl.classList.toggle('ok', !data.off_map);
  offmapEl.classList.toggle('flagged', data.off_map);
  offmapEl.textContent = data.off_map
    ? `Never seen before: ${data.unknown_chars.length} character(s) that were not in its training data: ${data.unknown_chars.join(' ')}`
    : 'Yes -- every character you typed appeared in its training data.';

  continuationEl.innerHTML = '';
  const promptSpan = document.createElement('span');
  promptSpan.className = 'prompt-part';
  promptSpan.textContent = data.prompt;
  const contSpan = document.createElement('span');
  contSpan.className = 'cont-part';
  contSpan.textContent = data.continuation;
  continuationEl.appendChild(promptSpan);
  continuationEl.appendChild(contSpan);

  renderSource(data.source);
}

function renderSource(source) {
  sourceSnippetEl.innerHTML = '';
  if (!source || !source.matched) {
    sourceLabelEl.textContent = 'no match -- this context does not appear verbatim in the training corpus';
    return;
  }
  sourceLabelEl.textContent = `${source.file}:${source.line}`;
  const preSpan = document.createElement('span');
  preSpan.className = 'prompt-part';
  preSpan.textContent = source.pre;
  const matchSpan = document.createElement('span');
  matchSpan.className = 'cont-part';
  matchSpan.textContent = source.match;
  const postSpan = document.createElement('span');
  postSpan.className = 'prompt-part';
  postSpan.textContent = source.post;
  sourceSnippetEl.appendChild(preSpan);
  sourceSnippetEl.appendChild(matchSpan);
  sourceSnippetEl.appendChild(postSpan);
}

let activeMode = null;

async function analyze(prompt) {
  if (!activeMode) return;
  try {
    const res = await fetch('/api/analyze?mode=' + encodeURIComponent(activeMode) +
                            '&prompt=' + encodeURIComponent(prompt));
    const data = await res.json();
    if (!res.ok) {
      errorEl.textContent = data.error || ('error ' + res.status);
      return;
    }
    renderAnalysis(data);
  } catch (e) {
    errorEl.textContent = String(e);
  }
}

promptEl.addEventListener('input', () => {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => analyze(promptEl.value), 200);
});

const modeToggleEl = document.getElementById('mode-toggle');
const modeBlurbEl = document.getElementById('mode-blurb');
const examplesEl = document.getElementById('examples');

function renderExamples(mode) {
  examplesEl.innerHTML = '';
  mode.examples.forEach(ex => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.textContent = ex.label;
    btn.addEventListener('click', () => {
      promptEl.value = ex.prompt;
      promptEl.focus();
      clearTimeout(debounceTimer);
      analyze(promptEl.value);
    });
    examplesEl.appendChild(btn);
  });
}

function applyMode(mode) {
  activeMode = mode.id;
  modeBlurbEl.textContent = mode.blurb;
  modeToggleEl.querySelectorAll('button').forEach(b =>
    b.classList.toggle('active', b.dataset.mode === mode.id));
  renderExamples(mode);
  errorEl.textContent = '';
  analyze(promptEl.value);
}

async function loadModes() {
  try {
    const res = await fetch('/api/modes');
    const data = await res.json();
    const byId = {};
    data.modes.forEach(m => { byId[m.id] = m; });
    modeToggleEl.innerHTML = '';
    data.modes.forEach(m => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.dataset.mode = m.id;
      btn.textContent = m.available ? m.label : m.label + ' (not trained)';
      btn.disabled = !m.available;
      btn.addEventListener('click', () => applyMode(m));
      modeToggleEl.appendChild(btn);
    });
    const start = byId[data.default] && byId[data.default].available
      ? byId[data.default]
      : data.modes.find(m => m.available);
    if (start) applyMode(start);
    else errorEl.textContent = 'No trained model found -- run `python colophon.py demo` first.';
  } catch (e) {
    errorEl.textContent = String(e);
  }
}

function renderScorecard(sc) {
  const tbody = document.querySelector('#scorecard tbody');
  tbody.innerHTML = '';
  const header = document.createElement('tr');
  ['dimension', 'colophon', 'typical closed', 'note'].forEach(h => {
    const th = document.createElement('th');
    th.textContent = h;
    header.appendChild(th);
  });
  tbody.appendChild(header);
  sc.dimensions.forEach(d => {
    const tr = document.createElement('tr');
    const cells = [d.dimension, d.colophon, d.typical_closed, d.note];
    cells.forEach((val, i) => {
      const td = document.createElement('td');
      td.textContent = val;
      if (i === 1) td.className = d.colophon;
      if (i === 2) td.className = d.typical_closed;
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  const caption = document.createElement('div');
  caption.className = 'muted';
  caption.textContent = `${sc.colophon_open}/${sc.scored} open (this artifact) vs ${sc.typical_closed_open}/${sc.scored} open (typical closed model)`;
  document.getElementById('scorecard-panel').appendChild(caption);
}

fetch('/api/scorecard').then(r => r.json()).then(renderScorecard).catch(e => {
  errorEl.textContent = String(e);
});

loadModes();
</script>
</body>
</html>
"""


def make_handler(modes, default_mode=DEFAULT_MODE):
    """modes maps a mode id (e.g. "osai", "elements") to a config dict:
        {"model": (p, stoi, itos, K) or None,   # None if that npz was absent
         "files": [(name, text), ...],           # corpus for the source panel
         "label": str, "blurb": str,             # page copy
         "examples": [(label, prompt), ...],
         "train_hint": str}                      # shown in the 503 message
    The scorecard and page serve regardless of which models loaded; /api/analyze
    reports 400 for an unknown mode and 503 for a known mode whose model is
    absent, so one missing corpus never takes the page down."""

    def _modes_payload():
        return {
            "default": default_mode,
            "modes": [
                {"id": mid, "label": cfg.get("label", mid),
                 "blurb": cfg.get("blurb", ""),
                 "available": cfg.get("model") is not None,
                 "examples": [{"label": lbl, "prompt": pr}
                              for lbl, pr in cfg.get("examples", [])]}
                for mid, cfg in modes.items()
            ],
        }

    class Handler(http.server.BaseHTTPRequestHandler):
        def _send(self, status, content_type, body: bytes):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, obj, status=200):
            self._send(status, "application/json; charset=utf-8",
                       json.dumps(obj).encode("utf-8"))

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8", INDEX_HTML.encode("utf-8"))
            elif parsed.path == "/api/scorecard":
                self._send_json(colophon.scorecard_section())
            elif parsed.path == "/api/modes":
                self._send_json(_modes_payload())
            elif parsed.path == "/api/analyze":
                qs = urllib.parse.parse_qs(parsed.query)
                mode = qs.get("mode", [default_mode])[0]
                if mode not in modes:
                    self._send_json({"error": f"unknown mode: {mode!r}"}, status=400)
                    return
                cfg = modes[mode]
                if cfg.get("model") is None:
                    hint = cfg.get("train_hint", "python colophon.py demo")
                    self._send_json(
                        {"error": f"no trained model for '{mode}' -- run `{hint}` first"},
                        status=503)
                    return
                prompt = qs.get("prompt", [""])[0][:MAX_PROMPT_LEN]
                p, stoi, itos, K = cfg["model"]
                try:
                    result = analyze_prompt(p, stoi, itos, K, prompt,
                                            files=cfg.get("files", ()))
                except Exception as e:
                    self._send_json(
                        {"error": f"analysis failed: {e}"}, status=500)
                    return
                self._send_json(result)
            else:
                self.send_error(404)

    return Handler


def _load_mode(mode_id, npz_path, src_dir):
    """Load one mode's model + corpus, print the same graceful warnings as
    before, and verify the corpus against the colophon.json that pairs with
    this npz. Returns (model_or_None, files)."""
    model = None
    try:
        model = load_model(npz_path)
    except FileNotFoundError:
        print(f"warning [{mode_id}]: no trained model at {npz_path} -- this mode "
              f"will be offered but unavailable until you train it")
    except (OSError, ValueError, KeyError) as e:
        print(f"warning [{mode_id}]: could not load model at {npz_path} ({e}) -- "
              f"this mode will be unavailable; the page still works")

    files = load_corpus_files(src_dir)
    if not files:
        print(f"warning [{mode_id}]: no .yaml/.yml files in {src_dir} -- the "
              f"source-echo panel will report every prompt as absent")
    else:
        json_path = colophon.colophon_json_path(npz_path)
        try:
            with open(json_path) as f:
                recorded_sha = json.load(f).get("data", {}).get("sha256")
        except (OSError, json.JSONDecodeError):
            recorded_sha = None
        if recorded_sha and corpus_sha256(files) != recorded_sha:
            print(f"warning [{mode_id}]: corpus at {src_dir} does not match "
                  f"{os.path.basename(json_path)}'s recorded sha256 -- it's a "
                  f"snapshot; retrain to refresh it")
    return model, files


def main():
    default_elements_npz = os.path.join(colophon.HERE, "elements.npz")
    default_elements_src = os.path.join(colophon.HERE, "teaching_data", "elements")
    ap = argparse.ArgumentParser(
        description="Marginalia -- live inspection UI for Colophon.")
    ap.add_argument("--npz", default=os.path.join(colophon.HERE, colophon.WEIGHTS_FILE),
                    help="trained OSAI colophon.npz (default: bundled location)")
    ap.add_argument("--src", default=colophon.DEFAULT_SRC,
                    help="directory of OSAI-index .yaml files (default: bundled sample)")
    ap.add_argument("--elements-npz", default=default_elements_npz,
                    help="trained periodic-table elements.npz for the teaching mode "
                         "(default: elements.npz beside colophon.py)")
    ap.add_argument("--elements-src", default=default_elements_src,
                    help="directory of the periodic-table .yaml files "
                         "(default: teaching_data/elements)")
    ap.add_argument("--host", default="127.0.0.1", help="local-only by default")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    sources = {"osai": (args.npz, args.src),
               "elements": (args.elements_npz, args.elements_src)}
    modes = {}
    for mode_id, meta in MODE_META.items():
        npz_path, src_dir = sources[mode_id]
        model, files = _load_mode(mode_id, npz_path, src_dir)
        modes[mode_id] = {"model": model, "files": files, **meta}

    # Open on an available mode: prefer the flagship, fall back to whatever loaded.
    default_mode = DEFAULT_MODE
    if modes[default_mode]["model"] is None:
        available = [mid for mid, cfg in modes.items() if cfg["model"] is not None]
        default_mode = available[0] if available else DEFAULT_MODE

    httpd = http.server.HTTPServer((args.host, args.port),
                                   make_handler(modes, default_mode=default_mode))
    print(f"Marginalia serving at http://{args.host}:{args.port}  (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
