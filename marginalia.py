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
or CDN script. `inspect_prompt()`, `context_saliency()`, and
`scorecard_section()` from colophon.py are the source of truth -- nothing here
re-derives those signals.

Usage:
  python colophon.py demo          # train a model first (writes colophon.npz)
  python marginalia.py             # serve the UI at http://127.0.0.1:8765
  python marginalia.py --port 9000 --npz /path/to/colophon.npz
"""

from __future__ import annotations
import argparse, http.server, json, os, urllib.parse
import numpy as np

import colophon

MAX_PROMPT_LEN = 500     # bound the per-keystroke forward-pass cost
CONTINUATION_LEN = 80    # sampled chars shown after the typed prompt


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


def make_handler(model):
    """model is (p, stoi, itos, K) if a trained colophon.npz was found, else
    None -- the scorecard and page still serve either way; /api/analyze
    reports 503 until a model exists."""

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
            else:
                self.send_error(404)

    return Handler


def main():
    ap = argparse.ArgumentParser(
        description="Marginalia -- live inspection UI for Colophon.")
    ap.add_argument("--npz", default=os.path.join(colophon.HERE, colophon.WEIGHTS_FILE),
                    help="path to a trained colophon.npz (default: bundled location)")
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

    httpd = http.server.HTTPServer((args.host, args.port), make_handler(model))
    print(f"Marginalia serving at http://{args.host}:{args.port}  (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
