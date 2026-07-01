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

    pct = round((1.0 - entropy) * 100)
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


def analyze_prompt(p, stoi, itos, K, prompt, n=CONTINUATION_LEN, seed=0):
    """The one function the HTTP layer calls: entropy + off-map signal from
    prompt_confidence(), a sampled continuation from generate(), and a
    layperson confidence readout. The signals are colophon.py's own functions,
    called as-is; confidence_readout() only reframes entropy for display."""
    entropy, unknown = colophon.prompt_confidence(p, stoi, K, prompt)
    full = colophon.generate(p, stoi, itos, K, prompt=prompt, n=n, seed=seed)
    return {
        "prompt": prompt,
        "entropy": entropy,
        "unknown_chars": unknown,
        "off_map": bool(unknown),
        "continuation": full[len(prompt):],
        **confidence_readout(entropy, unknown, has_prompt=bool(prompt)),
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
<p class="muted">This is a tiny language model trained only on the open-source AI
index. Type something and watch its own honesty signals react &mdash; how sure it
is, whether it has ever seen these characters, and what it would write next. Every
signal is computed by the model's own weights, not guessed at in JavaScript.</p>

<textarea id="prompt" placeholder="Type here, or tap an example below&hellip;" autofocus></textarea>

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
}

async function analyze(prompt) {
  try {
    const res = await fetch('/api/analyze?prompt=' + encodeURIComponent(prompt));
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

document.querySelectorAll('.examples button').forEach(btn => {
  btn.addEventListener('click', () => {
    promptEl.value = btn.dataset.prompt;
    promptEl.focus();
    clearTimeout(debounceTimer);
    analyze(promptEl.value);
  });
});

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
