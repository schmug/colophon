#!/usr/bin/env python3
"""incipit.py -- Incipit, the multiturn glass-box chat front-end for Colophon.

An incipit is a book's opening words -- the complement of a colophon. Where
Marginalia inspects one prompt, Incipit inspects a CONVERSATION: chat bubbles
on top, and underneath them the truth -- one growing text tape, the literal
K-character window the model re-reads, and per-character records (probability,
entropy, top-k, occlusion saliency) computed from the weights via
colophon.inspect_prompt(). glassboxllm (2025) mocked this UI with simulated
metrics; every number here is real.

The server is STATELESS BY DESIGN: the client sends the full turn list on
every request and the server holds no conversation. That mirrors how real LLM
APIs work and is itself part of the lesson -- "memory" is just text re-sent
and re-read. It is also why /api/turn responses are deterministic given the
same request + seed.

Three modes, three acts (the teaching sequence lives in the front-end; the
copy here drives the mode switcher):
  elements64  YAML-trained periodic-table model, K=64  (Acts 1-2)
  dialogue    dialogue-trained periodic-table model, K=64  (Act 3)
  osai        the K=12 flagship, for tiny-window drama (free play)
Act 2 and Act 3 share hyperparameters on purpose: the only variable between
"chat format fails" and "chat format works" is the training data's format.

Zero new runtime dependencies: stdlib http.server + numpy, same as Marginalia.
The React front-end in incipit/ is a build-time artifact; this server serves
its incipit/dist/ output, so production runs with Python alone.

Usage:
  python colophon.py --src teaching_data/elements --out elements_k64.npz \
      --steps 30000 --K 64 --E 64 --H 512 train
  python colophon.py --src teaching_data/dialogue --out dialogue_k64.npz \
      --steps 30000 --K 64 --E 64 --H 512 train
  (cd incipit && npm install && npm run build)     # once, for the UI
  python incipit.py                                 # http://127.0.0.1:8790
"""
from __future__ import annotations
import argparse, http.server, json, mimetypes, os, urllib.parse

import colophon
import marginalia

HERE = os.path.dirname(os.path.abspath(__file__))
DIST_DIR = os.path.join(HERE, "incipit", "dist")

MAX_TAPE_CHARS = 10_000      # bound the per-request forward-pass cost
MAX_CONTINUATION = 400       # hard cap on sampled chars per turn
DEFAULT_CONTINUATION = 160
MAX_TURNS = 200
ROLES = ("user", "model")
FORMATS = ("raw", "chat")

# Copy lives server-side (Marginalia's convention) so /api/modes drives the
# mode switcher and nothing is re-authored in JavaScript. The front-end's
# three-act rail copy lives client-side in incipit/src/acts.ts -- it is
# teaching narrative, not mode metadata.
MODE_META = {
    "elements64": {
        "label": "Periodic table (YAML, K=64)",
        "blurb": ("Trained on 118 element facts as YAML. Completion works; "
                  "chat format doesn't -- not because the window is small "
                  "(it is the same K=64 as the dialogue model) but because "
                  "its training data contains no conversations."),
        "acts": [1, 2],
        "format_default": "raw",
        "train_hint": ("python colophon.py --src teaching_data/elements "
                       "--out elements_k64.npz --steps 30000 --K 64 --E 64 "
                       "--H 512 train"),
    },
    "dialogue": {
        "label": "Periodic table (dialogue, K=64)",
        "blurb": ("Same architecture, same facts -- but the training data is "
                  "user:/model: conversations, so chat format works. A "
                  "chatbot is a completion model whose corpus contains "
                  "dialogue."),
        "acts": [3],
        "format_default": "chat",
        "train_hint": ("python colophon.py --src teaching_data/dialogue "
                       "--out dialogue_k64.npz --steps 30000 --K 64 --E 64 "
                       "--H 512 train"),
    },
    "osai": {
        "label": "Openness index (K=12)",
        "blurb": ("The flagship Colophon model. Its tiny 12-character window "
                  "makes the sliding-context lesson dramatic: watch it "
                  "forget almost immediately."),
        "acts": [],
        "format_default": "raw",
        "train_hint": "python colophon.py demo",
    },
}
DEFAULT_MODE = "elements64"  # Incipit is the teaching app; Act 1 starts here


def build_tape(turns, fmt):
    """The exact string the model is conditioned on -- returned to the client
    verbatim, because "the conversation IS this string" is the lesson.

    raw : active turn texts concatenated directly (completion mode).
    chat: each active turn as 'role: text\\n', then 'model: ' so generation
          begins where the model's reply would -- byte-identical to the
          dialogue corpus's training format (see build_dialogue.py)."""
    active = [t for t in turns if not t.get("excluded")]
    if fmt == "chat":
        return "".join(f"{t['role']}: {t['text']}\n" for t in active) + "model: "
    return "".join(t["text"] for t in active)


def parse_turn_request(body):
    """Validate and normalize a /api/turn request body. Returns (req, error)
    where exactly one is None; error is (http_status, message). Off-map
    characters in the text are deliberately NOT an error -- they are a
    teaching signal reported in the response."""
    if not isinstance(body, dict):
        return None, (400, "request body must be a JSON object")
    turns = body.get("turns")
    if not isinstance(turns, list) or not turns:
        return None, (400, "turns must be a non-empty list")
    if len(turns) > MAX_TURNS:
        return None, (400, f"too many turns (max {MAX_TURNS})")
    for t in turns:
        if not isinstance(t, dict) or t.get("role") not in ROLES \
                or not isinstance(t.get("text"), str):
            return None, (400, "each turn needs a role in ('user','model') "
                               "and string text")
    fmt = body.get("format", "raw")
    if fmt not in FORMATS:
        return None, (400, f"format must be one of {FORMATS}")
    s = body.get("sampling") or {}
    if not isinstance(s, dict):
        return None, (400, "sampling must be an object")
    try:
        sampling = {
            "temperature": float(s.get("temperature", 0.8)),
            "top_k": int(s.get("top_k", 0)),
            "seed": int(s.get("seed", 0)),
            "max_chars": min(int(s.get("max_chars", DEFAULT_CONTINUATION)),
                             MAX_CONTINUATION),
            "stop": s.get("stop") if isinstance(s.get("stop"), str) else None,
            "banned_chars": s.get("banned_chars") or [],
        }
    except (TypeError, ValueError):
        return None, (400, "sampling values must be numeric")
    if sampling["max_chars"] < 1:
        return None, (400, "max_chars must be >= 1")
    if not (0.0 <= sampling["temperature"] <= 4.0):
        return None, (400, "temperature must be in [0, 4]")
    if sampling["top_k"] < 0:
        return None, (400, "top_k must be >= 0")
    bc = sampling["banned_chars"]
    if not isinstance(bc, list) or any(not isinstance(c, str) or len(c) != 1
                                       for c in bc):
        return None, (400, "banned_chars must be a list of single characters")
    if len(bc) > 20:
        return None, (400, "banned_chars: at most 20")
    if sampling["stop"] is not None and len(sampling["stop"]) > 8:
        return None, (400, "stop must be at most 8 characters")
    tape = build_tape(turns, fmt)
    if len(tape) > MAX_TAPE_CHARS:
        return None, (413, f"tape is {len(tape)} chars; max {MAX_TAPE_CHARS}")
    return {"turns": turns, "format": fmt, "sampling": sampling,
            "tape": tape}, None


def run_turn(model, tape, sampling, files=()):
    """One honest generation step. Every number is computed by colophon.py's
    own functions against the loaded weights; this only assembles the
    payload. Banned chars the vocab doesn't contain are skipped (you cannot
    mask a logit that doesn't exist) and the response says which ones
    actually applied."""
    p, stoi, itos, K = model
    banned_applied = sorted({c for c in sampling["banned_chars"] if c in stoi})
    banned_ids = [stoi[c] for c in banned_applied]
    records = colophon.inspect_prompt(
        p, stoi, itos, K, tape,
        n_continuation=sampling["max_chars"] if tape else 0,
        seed=sampling["seed"], temp=sampling["temperature"],
        top_k=sampling["top_k"], banned_ids=banned_ids,
        stop=sampling["stop"])
    continuation = "".join(r["char"] for r in records if r["is_continuation"])
    entropy, unknown = colophon.prompt_confidence(p, stoi, K, tape)
    return {
        "tape": tape,
        "continuation": continuation,
        "records": records,
        "K": K,
        "entropy": entropy,
        "unknown_chars": unknown,
        "off_map": bool(unknown),
        "banned_applied": banned_applied,
        **marginalia.confidence_readout(entropy, unknown,
                                        has_prompt=bool(tape)),
        "source": marginalia.find_source_echo(files, tape),
    }
