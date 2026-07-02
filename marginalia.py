#!/usr/bin/env python3
"""
marginalia.py -- Marginalia, the live inspection UI for Colophon.

A small local web page: type a prompt and watch the model's own white-box
signals react in real time, against a trained `colophon.npz`. It layers a
layperson-facing read (a plain "how sure is it" confidence readout, an off-map
flag, and a literal source-in-training-data match) over a maximal per-position
inspector (an entropy heatmap, the literal K-char context window, occlusion-based
context saliency, and the top-k next-char distribution) plus the OSAI scorecard.
It makes the auditability `colophon.py demo` prints interactive instead of a
one-shot report.

Marginalia adds ZERO runtime dependencies: the server is Python's stdlib
`http.server`, and the frontend is a single vanilla-JS page with no build step
or CDN script. `inspect_prompt()`, `context_saliency()`, `prompt_confidence()`,
and `scorecard_section()` from colophon.py are the source of truth;
`confidence_readout()` and `find_source_echo()` only reframe or locate those
signals for display -- nothing here re-derives the model's numbers.

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
    """The one call the per-keystroke path makes. It combines the layperson
    signals -- a plain confidence readout and a literal source-in-training-data
    match -- with the maximal per-position white-box records the inspector
    renders. Every number comes from colophon.py's own functions
    (prompt_confidence / inspect_prompt) or a presentation transform of them
    (confidence_readout, find_source_echo); nothing is re-derived. An empty
    prompt yields no records (we do not dream a continuation from nothing)."""
    entropy, unknown = colophon.prompt_confidence(p, stoi, K, prompt)
    n_eff = n if prompt else 0
    records = colophon.inspect_prompt(p, stoi, itos, K, prompt,
                                      n_continuation=n_eff, seed=seed)
    return {
        "prompt": prompt,
        "entropy": entropy,
        "records": records,
        "unknown_chars": unknown,
        "off_map": bool(unknown),
        **confidence_readout(entropy, unknown, has_prompt=bool(prompt)),
        "source": find_source_echo(files, prompt),
    }


def context_saliency(p, stoi, itos, K, prompt, pos, n=CONTINUATION_LEN, seed=0):
    """Thin wrapper over colophon.context_saliency for the /api/saliency route.
    Called only when the focused position changes -- never per keystroke."""
    return colophon.context_saliency(p, stoi, itos, K, prompt, pos,
                                     n_continuation=n, seed=seed)


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
  .panel h2 { font-size: .95rem; margin: 0 0 .4rem; }
  .muted { opacity: .6; font-size: .85rem; }
  .bb { color: #b60; font-size: .8rem; margin-top: .4rem; }
  #bb-banner { border-left: 3px solid #b60; padding: .5rem .75rem; background: #b6010a10;
               font-size: .85rem; }
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
  .examples { margin: .5rem 0 0; display: flex; flex-wrap: wrap; gap: .4rem; }
  .examples button { font: inherit; font-size: .8rem; cursor: pointer;
                     border: 1px solid #8886; border-radius: 999px;
                     padding: .2rem .7rem; background: #8881; color: inherit; }
  .examples button:hover { background: #8883; }
  .continuation .prompt-part { opacity: .55; }
  .continuation .cont-part { font-weight: bold; }
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

<p class="muted">This is a tiny language model trained only on the open-source AI
index. Type something and watch its own honesty signals react &mdash; how sure it
is, whether it has ever seen these characters, what it would write next, and
(further down) a per-character inspector. Every signal is computed by the model's
own weights, not guessed at in JavaScript.</p>

<textarea id="prompt" aria-label="Prompt" placeholder="Type here, or tap an example below&hellip;" autofocus></textarea>

<div class="examples">
  <button type="button" data-prompt="weights_basemodel:">Something it trained on</button>
  <button type="button" data-prompt="&#26085;&#26412;&#35486;&#12391;&#26360;&#12356;&#12390;&#12367;&#12384;&#12373;&#12356;">Characters it&rsquo;s never seen</button>
  <button type="button" data-prompt="the 2027 election">A topic outside its data</button>
</div>

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
  <h2>What this model does and doesn't disclose</h2>
  <p class="muted" style="margin:.2rem 0 .6rem">The openness index scores AI systems
  on what they reveal. Here's how this artifact grades against a typical closed model
  &mdash; not "open is better," just what is and isn't on the table.</p>
  <table id="scorecard"><tbody></tbody></table>
</div>

<div id="error" class="error"></div>

<script>
const $ = id => document.getElementById(id);
const promptEl = $('prompt'), errorEl = $('error');
const confPctEl = $('conf-pct'), verdictEl = $('verdict'), confFill = $('conf-fill');
const entropyVal = $('entropy-val'), offmapEl = $('offmap');
const continuationEl = $('continuation');
const sourceLabelEl = $('source-label'), sourceSnippetEl = $('source-snippet');

const VERDICT_LEVELS = ['confident', 'unsure', 'struggling', 'off-map', 'none'];

let records = [], focus = 0, timer = null;

function entColor(e) {  // 0 (green) -> 1 (red)
  const h = Math.round(140 * (1 - Math.max(0, Math.min(1, e))));
  return `hsl(${h} 60% 50% / .35)`;
}

function renderConfidence(data) {
  const pct = data.confidence_pct;
  confPctEl.textContent = pct === null ? '--' : pct + '% sure';
  confFill.style.width = (pct === null ? 0 : Math.min(100, Math.max(0, pct))) + '%';
  verdictEl.textContent = data.verdict;
  VERDICT_LEVELS.forEach(l => verdictEl.classList.toggle(l, l === data.verdict_level));
  entropyVal.textContent = data.entropy.toFixed(3);
}

function renderOffmap(data) {
  offmapEl.classList.toggle('ok', !data.off_map);
  offmapEl.classList.toggle('flagged', data.off_map);
  offmapEl.textContent = data.off_map
    ? `Never seen before: ${data.unknown_chars.length} character(s) that were not in its training data: ${data.unknown_chars.join(' ')}`
    : 'Yes -- every character you typed appeared in its training data.';
}

function renderContinuation(data) {
  continuationEl.innerHTML = '';
  const promptSpan = document.createElement('span');
  promptSpan.className = 'prompt-part';
  promptSpan.textContent = data.prompt;
  const contSpan = document.createElement('span');
  contSpan.className = 'cont-part';
  contSpan.textContent = records.filter(r => r.is_continuation).map(r => r.char).join('');
  continuationEl.append(promptSpan, contSpan);
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
  sourceSnippetEl.append(preSpan, matchSpan, postSpan);
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
  note.textContent = `top ${r.top_k.length} of the model's full next-char distribution.`;
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
  const medianB = document.createElement('b');
  medianB.textContent = median.toFixed(3);
  const anchorsB = document.createElement('b');
  anchorsB.textContent = anchors;
  const offCountB = document.createElement('b');
  offCountB.textContent = String(offCount);
  div.append(
    document.createTextNode('median entropy '),
    medianB,
    document.createTextNode(' · most-confident chars (anchors): '),
    anchorsB,
    document.createTextNode(' · off-map characters: '),
    offCountB
  );
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

function renderAnalysis(data) {
  errorEl.textContent = '';
  records = data.records;
  renderConfidence(data);
  renderOffmap(data);
  renderContinuation(data);
  renderSource(data.source);
  // Default focus: the lowest-confidence (highest-entropy) position.
  focus = records.length
    ? records.reduce((best, r, i, a) => r.entropy > a[best].entropy ? i : best, 0)
    : 0;
  renderHeatmap(); renderFocus(); renderAggregates(); fetchSaliency();
}

async function analyze(prompt) {
  try {
    const res = await fetch('/api/analyze?prompt=' + encodeURIComponent(prompt));
    const data = await res.json();
    if (!res.ok) { errorEl.textContent = data.error || ('error ' + res.status); return; }
    renderAnalysis(data);
  } catch (e) { errorEl.textContent = String(e); }
}

promptEl.addEventListener('input', () => {
  clearTimeout(timer);
  timer = setTimeout(() => analyze(promptEl.value), 200);
});

document.querySelectorAll('.examples button').forEach(btn => {
  btn.addEventListener('click', () => {
    promptEl.value = btn.dataset.prompt;
    promptEl.focus();
    clearTimeout(timer);
    analyze(promptEl.value);
  });
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


def make_handler(model, files=()):
    """model is (p, stoi, itos, K) if a trained colophon.npz was found, else
    None -- the scorecard and page still serve either way; /api/analyze and
    /api/saliency report 503 until a model exists. files is the corpus loaded
    by load_corpus_files(), used for the source-echo panel; an empty corpus
    just makes every prompt report as absent from the training data."""

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

        def _no_model(self):
            self._send_json(
                {"error": "no trained model found -- run `python colophon.py demo` first"},
                status=503)

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8", INDEX_HTML.encode("utf-8"))
            elif parsed.path == "/api/scorecard":
                self._send_json(colophon.scorecard_section())
            elif parsed.path == "/api/analyze":
                if model is None:
                    self._no_model()
                    return
                qs = urllib.parse.parse_qs(parsed.query)
                prompt = qs.get("prompt", [""])[0][:MAX_PROMPT_LEN]
                p, stoi, itos, K = model
                try:
                    result = analyze_prompt(p, stoi, itos, K, prompt, files=files)
                except Exception as e:
                    self._send_json(
                        {"error": f"analysis failed: {e}"}, status=500)
                    return
                self._send_json(result)
            elif parsed.path == "/api/saliency":
                if model is None:
                    self._no_model()
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
            else:
                self.send_error(404)

    return Handler


def main():
    ap = argparse.ArgumentParser(
        description="Marginalia -- live inspection UI for Colophon.")
    ap.add_argument("--npz", default=os.path.join(colophon.HERE, colophon.WEIGHTS_FILE),
                    help="path to a trained colophon.npz (default: bundled location)")
    ap.add_argument("--src", default=colophon.DEFAULT_SRC,
                    help="directory of OSAI-index .yaml files for the live "
                         "source-echo panel (default: bundled sample)")
    ap.add_argument("--host", default="127.0.0.1", help="local-only by default")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    model = None
    try:
        model = load_model(args.npz)
    except FileNotFoundError:
        print(f"warning: no trained model at {args.npz} -- entropy/off-map will "
              f"be unavailable until you run `python colophon.py demo` (or train)")
    except (OSError, ValueError, KeyError) as e:
        print(f"warning: could not load model at {args.npz} ({e}) -- entropy/"
              f"off-map will be unavailable; the scorecard and page still work")

    files = load_corpus_files(args.src)
    if not files:
        print(f"warning: no .yaml/.yml files found in {args.src} -- the "
              f"source-echo panel will report every prompt as absent")
    else:
        colophon_json_path = os.path.join(colophon.HERE, colophon.COLOPHON_FILE)
        try:
            with open(colophon_json_path) as f:
                recorded_sha = json.load(f).get("data", {}).get("sha256")
        except (OSError, json.JSONDecodeError):
            recorded_sha = None
        if recorded_sha:
            live_sha = corpus_sha256(files)
            if live_sha != recorded_sha:
                print(f"warning: corpus at {args.src} (sha256 {live_sha[:12]}...) "
                      f"does not match {colophon.COLOPHON_FILE}'s recorded sha256 "
                      f"({recorded_sha[:12]}...) -- colophon.json is a snapshot; "
                      f"run `python colophon.py demo` to refresh it")

    httpd = http.server.HTTPServer((args.host, args.port), make_handler(model, files))
    print(f"Marginalia serving at http://{args.host}:{args.port}  (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
