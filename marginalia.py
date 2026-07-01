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
  .entropy-bar { height: 10px; border-radius: 5px; background: #8882;
                 overflow: hidden; margin-top: .25rem; }
  .entropy-fill { height: 100%; background: linear-gradient(90deg, #2a6, #d63); }
  .offmap { font-weight: bold; }
  .offmap.ok { color: #2a6; }
  .offmap.flagged { color: #d33; }
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
<p class="muted">Type a prompt. Entropy, the off-map/unknown-character flag, and a
sampled continuation are computed live by Colophon's own white-box signals --
nothing here is re-derived in JavaScript.</p>

<textarea id="prompt" placeholder="e.g. availability_weights_endmodel_class:  or  &#26085;&#26412;&#35486;&#12391;&#26360;&#12356;&#12390;&#12367;&#12384;&#12373;&#12356;" autofocus></textarea>

<div class="panel">
  <div>next-char entropy (normalized, 0 = certain, 1 = no idea): <b id="entropy-val">--</b></div>
  <div class="entropy-bar"><div class="entropy-fill" id="entropy-fill" style="width:0%"></div></div>
</div>

<div class="panel">
  <div id="offmap" class="offmap ok">no off-map characters</div>
</div>

<div class="panel">
  <div class="muted">sampled continuation (model's own weights, temp 0.8):</div>
  <div id="continuation" class="continuation">&nbsp;</div>
</div>

<div class="panel" id="scorecard-panel">
  <div class="muted">OSAI openness scorecard</div>
  <table id="scorecard"><tbody></tbody></table>
</div>

<div id="error" class="error"></div>

<script>
const promptEl = document.getElementById('prompt');
const entropyVal = document.getElementById('entropy-val');
const entropyFill = document.getElementById('entropy-fill');
const offmapEl = document.getElementById('offmap');
const continuationEl = document.getElementById('continuation');
const errorEl = document.getElementById('error');

let debounceTimer = null;

function renderAnalysis(data) {
  errorEl.textContent = '';
  entropyVal.textContent = data.entropy.toFixed(3);
  entropyFill.style.width = Math.min(100, Math.max(0, data.entropy * 100)) + '%';

  offmapEl.classList.toggle('ok', !data.off_map);
  offmapEl.classList.toggle('flagged', data.off_map);
  offmapEl.textContent = data.off_map
    ? `off-map: ${data.unknown_chars.length} character(s) never seen in training: ${data.unknown_chars.join(' ')}`
    : 'no off-map characters -- every character was seen in training';

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
