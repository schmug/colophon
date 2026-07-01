#!/usr/bin/env python3
"""
colophon.py -- a fully-open, from-scratch character language model that ships its
               own colophon, trained on the European Open Source AI Index (the
               same data that *defines* openness).

A colophon is the note at the end of a book stating how it was made -- press,
typeface, paper, print run. This model is named for that and embodies it: it
emits a `colophon.json` documenting its own data, training, and openness grades.

The point is not capability. A ~45K-parameter char model is not a chatbot. The
point is that EVERY property people usually have to estimate about a model is,
here, ground truth you can verify:

  * Training data      -- you can read the entire corpus; nothing else exists.
  * Knowledge cutoff   -- absolute. The model's whole universe is this corpus.
  * Competence boundary-- exactly the edge of the corpus. Ask outside it and you
                          get confident nonsense, on demand.
  * Confidence signals -- fully white-box (we own every weight and logit).

No framework. The model, its gradients, and its optimizer are a few hundred lines
of NumPy you can audit end to end -- which is itself the argument.

Subcommands:
  prepare  Build the corpus + write colophon.json (data section) and print scope.
  train    Train from scratch; save colophon.npz weights + colophon.json.
  demo     Train (fast), then show generation, in- vs out-of-distribution
           confidence, and the OSAI 14-dimension openness scorecard.

Usage:
  python colophon.py demo                       # runs end-to-end on bundled data
  python colophon.py demo --src /path/to/osai   # use a real clone of the index
  python colophon.py train --steps 8000 --src /path/to/osai
  python colophon.py generate --prompt "availability_weights"

Data attribution: the bundled sample mimics the schema of the European Open
Source AI Index (CC-BY 4.0, doi:10.5281/zenodo.15386042). When using the real
index, cite it per its license.
"""

from __future__ import annotations
import argparse, glob, hashlib, json, os, time
import numpy as np

PAD = "\x00"  # boundary/padding token, never present in normal text
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SRC = os.path.join(HERE, "sample_data")

NAME = "Colophon"
TAGLINE = "A model that ships its own colophon."
WEIGHTS_FILE = "colophon.npz"    # open weights
COLOPHON_FILE = "colophon.json"  # the model's self-description (datasheet + card + scorecard)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load_corpus(src_dir: str):
    paths = sorted(glob.glob(os.path.join(src_dir, "*.yaml")) +
                   glob.glob(os.path.join(src_dir, "*.yml")))
    if not paths:
        raise SystemExit(f"No .yaml files found in {src_dir}")
    docs = []
    for p in paths:
        with open(p, encoding="utf-8", errors="replace") as f:
            docs.append(f.read())
    # Separate documents with an explicit boundary token so the model learns
    # where entries begin and end.
    text = ("\n" + PAD + "\n").join(docs)
    return text, paths


def build_vocab(text: str):
    chars = [PAD] + sorted(set(text) - {PAD})
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return chars, stoi, itos


def encode(text, stoi):
    return np.array([stoi[c] for c in text], dtype=np.int64)


def data_manifest(text, paths, chars):
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    counts = {"open": text.count("open"),
              "partial": text.count("partial"),
              "closed": text.count("closed")}
    return {
        "source_files": [os.path.basename(p) for p in paths],
        "num_files": len(paths),
        "num_characters": len(text),
        "vocab_size": len(chars),
        "sha256": sha,
        "openness_class_token_counts": counts,
        "knowledge_scope": (
            f"The model's entire universe is these {len(text)} characters across "
            f"{len(paths)} files. It has no knowledge cutoff in the usual fuzzy "
            f"sense -- its cutoff is absolute: nothing outside this corpus exists "
            f"to it. Any fluent output about anything else is, by construction, "
            f"confident invention."
        ),
    }


# --------------------------------------------------------------------------- #
# Model: char-level MLP language model (Bengio-2003 style), pure NumPy.
# A single hidden layer keeps the manual backprop small and auditable. The
# architecture is deliberately swappable -- a transformer block is a drop-in
# upgrade for a real run on a GPU/MLX; it changes capability, not the argument.
# --------------------------------------------------------------------------- #

# Apple's Accelerate/vecLib BLAS spuriously raises the divide/overflow/invalid
# floating-point flags from its `matmul` path even when the inputs and outputs
# are perfectly finite -- a well-known false positive that clutters the demo with
# RuntimeWarnings. We scope an errstate around ONLY the matmul-bearing lines, so
# these cosmetic flags stay quiet while a genuine NaN/Inf anywhere else (the
# softmax exp/log, a diverged loss) still surfaces -- the model stays auditable
# by eye. This is deliberately NOT a module-level np.seterr(all="ignore").
_MATMUL_FP = dict(divide="ignore", over="ignore", invalid="ignore")

def init_params(V, E, K, H, seed=0):
    rng = np.random.default_rng(seed)
    s = 0.02
    return {
        "C":  rng.normal(0, s, (V, E)),        # token embedding
        "W1": rng.normal(0, s, (K * E, H)),    # context -> hidden
        "b1": np.zeros(H),
        "W2": rng.normal(0, s, (H, V)),        # hidden -> next-char logits
        "b2": np.zeros(V),
    }


def forward(p, Xb):
    """Xb: (B, K) int context. Returns logits (B, V) and a cache for backprop
    (cache is None on the transformer path, which trains via torch autograd
    instead of the manual backprop below)."""
    if p.get("_arch") == "transformer":
        return forward_transformer(p, Xb)
    B, K = Xb.shape
    emb = p["C"][Xb]                 # (B, K, E)
    x = emb.reshape(B, -1)           # (B, K*E)
    with np.errstate(**_MATMUL_FP):  # Accelerate BLAS matmul false positive; see above
        h = np.tanh(x @ p["W1"] + p["b1"])   # (B, H)
        logits = h @ p["W2"] + p["b2"]       # (B, V)
    return logits, (Xb, emb, x, h)


def loss_and_grads(p, Xb, Yb):
    logits, (Xb, emb, x, h) = forward(p, Xb)
    B = Xb.shape[0]
    logits = logits - logits.max(1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(1, keepdims=True)
    loss = -np.log(probs[np.arange(B), Yb] + 1e-12).mean()

    dlogits = probs
    dlogits[np.arange(B), Yb] -= 1.0
    dlogits /= B
    with np.errstate(**_MATMUL_FP):      # Accelerate BLAS matmul false positive; see above
        dW2 = h.T @ dlogits
        db2 = dlogits.sum(0)
        dh = dlogits @ p["W2"].T
        dhpre = dh * (1.0 - h * h)        # tanh'
        dW1 = x.T @ dhpre
        db1 = dhpre.sum(0)
        dx = dhpre @ p["W1"].T            # (B, K*E)
    demb = dx.reshape(emb.shape)         # (B, K, E)
    dC = np.zeros_like(p["C"])
    np.add.at(dC, Xb, demb)              # scatter-add into embedding rows
    return loss, {"C": dC, "W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


class Adam:
    def __init__(self, params, lr=3e-3, betas=(0.9, 0.999), eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, betas[0], betas[1], eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


def n_params(p):
    if p.get("_arch") == "transformer":
        return int(sum(v.numel() for v in p["module"].parameters()))
    return int(sum(v.size for v in p.values()))


# --------------------------------------------------------------------------- #
# Optional arch: single-block causal transformer, torch, lazily imported.
# The NumPy MLP above stays the default and the auditable reference; this path
# changes capability, not the argument (CLAUDE.md). It mirrors the MLP's exact
# interface -- (B, K) int contexts in, (B, V) next-char logits out -- so
# generate() and prompt_confidence() work unmodified against either arch, and
# entropy / the off-map signal stay comparable across both.
# --------------------------------------------------------------------------- #

def _require_torch():
    try:
        import torch
    except ImportError as e:
        raise SystemExit(
            "--arch transformer requires PyTorch, which is not installed by "
            "default (the NumPy path stays dependency-free). Install with: "
            "pip install torch"
        ) from e
    return torch


def _build_transformer_module(V, K, E, n_head, seed):
    torch = _require_torch()
    import torch.nn as nn

    class _CausalBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(V, E)
            self.pos_emb = nn.Parameter(torch.zeros(1, K, E))
            self.attn = nn.MultiheadAttention(E, n_head, batch_first=True)
            self.ln1 = nn.LayerNorm(E)
            self.ff = nn.Sequential(nn.Linear(E, 4 * E), nn.GELU(), nn.Linear(4 * E, E))
            self.ln2 = nn.LayerNorm(E)
            self.head = nn.Linear(E, V)
            self.register_buffer("mask", torch.triu(torch.ones(K, K), diagonal=1).bool())

        def forward(self, idx):
            x = self.tok_emb(idx) + self.pos_emb
            a, _ = self.attn(x, x, x, attn_mask=self.mask, need_weights=False)
            x = self.ln1(x + a)
            x = self.ln2(x + self.ff(x))
            return self.head(x[:, -1, :])  # last context position -> next-char logits

    torch.manual_seed(seed)
    return _CausalBlock()


def init_params_transformer(V, K, E=32, n_head=4, seed=0):
    module = _build_transformer_module(V, K, E, n_head, seed)
    return {"_arch": "transformer", "module": module, "V": V, "K": K, "E": E, "n_head": n_head}


def forward_transformer(p, Xb):
    torch = _require_torch()
    idx = torch.from_numpy(np.asarray(Xb, dtype=np.int64))
    with torch.no_grad():
        logits = p["module"](idx)
    return logits.numpy(), None


def train_transformer(p, seq, N, K, steps, batch, lr, seed, log_every):
    torch = _require_torch()
    module = p["module"]
    opt = torch.optim.Adam(module.parameters(), lr=lr)
    rng = np.random.default_rng(seed + 1)
    t0 = time.time()
    hist = []
    for s in range(1, steps + 1):
        Xb, Yb = get_batch(seq, N, K, batch, rng)
        logits = module(torch.from_numpy(Xb))
        loss = torch.nn.functional.cross_entropy(logits, torch.from_numpy(Yb))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if s % log_every == 0 or s == 1:
            hist.append((s, float(loss.item())))
            print(f"  step {s:>6}/{steps}   loss {loss.item():.4f}")
    module.eval()
    return hist, round(time.time() - t0, 2)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def get_batch(seq, N, K, B, rng):
    """seq = [PAD]*K + encoded_text. Context seq[i:i+K] predicts seq[i+K]."""
    ix = rng.integers(0, N, size=B)
    Xb = np.stack([seq[i:i + K] for i in ix])
    Yb = seq[ix + K]
    return Xb, Yb


def train_model(text, stoi, chars, K=12, E=24, H=128, steps=6000, batch=64,
                lr=3e-3, seed=0, log_every=1000, arch="mlp"):
    V = len(chars)
    idx = encode(text, stoi)
    seq = np.concatenate([np.zeros(K, dtype=np.int64), idx])
    N = len(idx)

    if arch == "transformer":
        p = init_params_transformer(V, K, seed=seed)
        hist, wall = train_transformer(p, seq, N, K, steps, batch, lr, seed, log_every)
        manifest = {
            "arch": "transformer",
            "architecture": f"causal single-block transformer (torch), "
                             f"K={K} E={p['E']} heads={p['n_head']}",
            "backend": "PyTorch, CPU, autograd",
            "context_length_K": K, "embed_dim_E": p["E"], "hidden_H": 4 * p["E"],
            "attention_heads": p["n_head"],
            "vocab_size_V": V, "parameters": n_params(p),
            "steps": steps, "batch": batch, "lr": lr, "seed": seed,
            "final_loss": hist[-1][1] if hist else None,
            "wall_clock_seconds": wall,
        }
        return p, manifest

    p = init_params(V, E, K, H, seed=seed)
    opt = Adam(p, lr=lr)
    rng = np.random.default_rng(seed + 1)
    t0 = time.time()
    hist = []
    for s in range(1, steps + 1):
        Xb, Yb = get_batch(seq, N, K, batch, rng)
        loss, grads = loss_and_grads(p, Xb, Yb)
        opt.step(p, grads)
        if s % log_every == 0 or s == 1:
            hist.append((s, float(loss)))
            print(f"  step {s:>6}/{steps}   loss {loss:.4f}")
    manifest = {
        "arch": "mlp",
        "architecture": "char-level MLP LM (embedding -> tanh hidden -> logits)",
        "backend": "pure NumPy, CPU, manual backprop",
        "context_length_K": K, "embed_dim_E": E, "hidden_H": H,
        "vocab_size_V": V, "parameters": n_params(p),
        "steps": steps, "batch": batch, "lr": lr, "seed": seed,
        "final_loss": hist[-1][1] if hist else None,
        "wall_clock_seconds": round(time.time() - t0, 2),
    }
    return p, manifest


# --------------------------------------------------------------------------- #
# Generation + white-box confidence signals
# --------------------------------------------------------------------------- #

def generate(p, stoi, itos, K, prompt="", n=240, temp=0.8, seed=0):
    rng = np.random.default_rng(seed)
    ctx = [0] * K
    for ch in prompt:
        if ch in stoi:
            ctx = (ctx + [stoi[ch]])[-K:]
    out = []
    for _ in range(n):
        logits, _ = forward(p, np.array([ctx]))
        l = logits[0] / temp
        l -= l.max()
        pr = np.exp(l); pr /= pr.sum()
        j = int(rng.choice(len(pr), p=pr))
        out.append(itos[j])
        ctx = (ctx + [j])[-K:]
    return prompt + "".join(out)


# --------------------------------------------------------------------------- #
# White-box inspection helpers (used by marginalia.py; see inspect_prompt).
# --------------------------------------------------------------------------- #

_DISPLAY = {"\x00": "∅", " ": "␣", "\n": "⏎", "\t": "⇥"}


def _display_char(ch):
    """Human-readable form of a single character for the inspection UI.
    PAD/whitespace get glyphs; everything else is shown as itself."""
    return _DISPLAY.get(ch, ch)


def prompt_confidence(p, stoi, K, prompt):
    """Teacher-force through `prompt`, returning mean NORMALIZED next-char
    entropy (0 = certain, 1 = uniform) and any characters the model has never
    seen. Unknown characters are the cleanest 'off-map' signal there is: the
    model has no representation for them at all."""
    unknown = sorted({ch for ch in prompt if ch not in stoi})
    ctx = [0] * K
    ents = []
    for ch in prompt:
        logits, _ = forward(p, np.array([ctx]))
        l = logits[0]; l -= l.max()
        pr = np.exp(l); pr /= pr.sum()
        ent = -(pr * np.log(pr + 1e-12)).sum() / np.log(len(pr))
        ents.append(ent)
        ctx = (ctx + [stoi.get(ch, 0)])[-K:]
    return (float(np.mean(ents)) if ents else 0.0), unknown


# --------------------------------------------------------------------------- #
# OSAI 14-dimension openness scorecard
# --------------------------------------------------------------------------- #

# The index groups openness into Availability / Documentation / Access. Below we
# score THIS artifact honestly (we do not hand ourselves free greens -- there is
# no preprint or peer-reviewed paper, so those are red) and contrast with a
# typical closed API model. G = open, P = partial, R = closed.
SCORECARD = [
    # dimension,                     ours, typical_closed, note
    ("Availability: data (base)",    "G", "R", "entire corpus is readable + hashed"),
    ("Availability: data (tuning)",  "G", "R", "no separate opaque tuning stage"),
    ("Availability: weights (base)", "G", "R", "weights saved to an open .npz"),
    ("Availability: weights (end)",  "G", "R", "same file; ungated"),
    ("Availability: training code",  "G", "R", "the training loop is this file"),
    ("Documentation: code",          "G", "R", "every step commented"),
    ("Documentation: hardware",      "G", "P", "'NumPy on one CPU core' -- fully stated"),
    ("Documentation: preprint",      "R", "R", "none -- this is a demo, not a paper"),
    ("Documentation: paper",         "R", "R", "none"),
    ("Documentation: model card",    "G", "P", "the training section is the card"),
    ("Documentation: datasheet",     "G", "R", "the data section is the datasheet"),
    ("Access: licenses",             "G", "R", "code open; data CC-BY"),
]
# The index defines 14 dimensions; the two not scored here are additional Access
# methods beyond licensing, which do not apply to a local artifact.

_GLYPH = {"G": "[open]   ", "P": "[partial]", "R": "[closed] "}
_LABEL = {"G": "open", "P": "partial", "R": "closed"}


def scorecard_section():
    """The openness scorecard as structured data, for colophon.json. Same source
    of truth as print_scorecard() -- glyphs expanded to readable labels."""
    ours_open = sum(1 for r in SCORECARD if r[1] == "G")
    closed_open = sum(1 for r in SCORECARD if r[2] == "G")
    return {
        "framework": "European Open Source AI Index (14 dimensions)",
        "colophon_open": ours_open,
        "typical_closed_open": closed_open,
        "scored": len(SCORECARD),
        "not_applicable": 14 - len(SCORECARD),
        "dimensions": [
            {"dimension": dim, "colophon": _LABEL[ours],
             "typical_closed": _LABEL[closed], "note": note}
            for dim, ours, closed, note in SCORECARD
        ],
    }


def print_scorecard():
    print("\nOSAI openness scorecard (this artifact  vs  a typical closed API model)")
    print("-" * 78)
    ours_open = sum(1 for r in SCORECARD if r[1] == "G")
    closed_open = sum(1 for r in SCORECARD if r[2] == "G")
    for dim, ours, closed, note in SCORECARD:
        print(f"  {dim:<30} {_GLYPH[ours]}  vs {_GLYPH[closed]}   {note}")
    print("-" * 78)
    print(f"  fully-open dimensions:   this artifact {ours_open}/12   "
          f"typical closed model {closed_open}/12")
    print("  (The two remaining OSAI dimensions are extra access methods, N/A here.)")


# --------------------------------------------------------------------------- #
# The colophon itself: one self-describing file (datasheet + model card +
# scorecard). A book's colophon in JSON -- the model documents how it was made.
# --------------------------------------------------------------------------- #

def build_colophon(data, training=None):
    """Assemble colophon.json. `training` is None before a model is trained
    (the `prepare` path), in which case the training section is left null."""
    return {
        "name": NAME,
        "tagline": TAGLINE,
        "generated_by": os.path.basename(__file__),
        "data": data,
        "training": training,
        "scorecard": scorecard_section(),
    }


def write_colophon(data, training=None):
    col = build_colophon(data, training)
    with open(os.path.join(HERE, COLOPHON_FILE), "w") as f:
        json.dump(col, f, indent=2, ensure_ascii=False)
    return col


# --------------------------------------------------------------------------- #
# Weight persistence. MLP save/load is byte-identical to before (no new keys
# in the .npz); the transformer path stores its torch state_dict as plain
# numpy arrays plus a tiny "arch"/"cfg" tag so `generate` can rebuild it
# without the CLI having to pass --arch again.
# --------------------------------------------------------------------------- #

def save_weights(p, chars):
    path = os.path.join(HERE, WEIGHTS_FILE)
    if p.get("_arch") == "transformer":
        state = {f"t__{k}": v.detach().cpu().numpy()
                 for k, v in p["module"].state_dict().items()}
        cfg = json.dumps({"V": p["V"], "K": p["K"], "E": p["E"], "n_head": p["n_head"]})
        np.savez(path, chars=np.array(chars, dtype=object),
                 arch=np.array("transformer"), cfg=np.array(cfg), **state)
    else:
        np.savez(path, **p, chars=np.array(chars, dtype=object))


def load_weights():
    d = np.load(os.path.join(HERE, WEIGHTS_FILE), allow_pickle=True)
    chars = list(d["chars"])
    arch = str(d["arch"]) if "arch" in d.files else "mlp"
    if arch == "transformer":
        cfg = json.loads(str(d["cfg"]))
        p = init_params_transformer(cfg["V"], cfg["K"], E=cfg["E"], n_head=cfg["n_head"])
        torch = _require_torch()
        state = {k[3:]: torch.from_numpy(d[k]) for k in d.files if k.startswith("t__")}
        p["module"].load_state_dict(state)
        p["module"].eval()
        return p, chars
    p = {k: d[k] for k in ("C", "W1", "b1", "W2", "b2")}
    return p, chars


def _infer_K(p):
    if p.get("_arch") == "transformer":
        return p["K"]
    return p["W1"].shape[0] // p["C"].shape[1]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

# Prompts chosen to sit clearly inside vs outside the corpus distribution.
# These mirror the real OSAI-index schema (nested dimension blocks: a top-level
# key such as `weights_basemodel:` followed by an indented `class:`), so every
# character is one the model has seen thousands of times. The bundled sample
# files use a flattened key style; on the real index these are the in-corpus form.
IN_DIST = [
    "weights_basemodel:\n    class: ",
    "datasheet:\n    class: ",
    "licenses:\n    class: ",
]
OUT_DIST = [
    "The mitochondria is the powerhouse of the cell.",
    "In 2027 the election results showed",
    "\u65e5\u672c\u8a9e\u3067\u66f8\u3044\u3066\u304f\u3060\u3055\u3044",  # Japanese
]


def cmd_prepare(args):
    text, paths = load_corpus(args.src)
    chars, stoi, itos = build_vocab(text)
    man = data_manifest(text, paths, chars)
    write_colophon(man, training=None)
    print(json.dumps({k: man[k] for k in
          ("num_files", "num_characters", "vocab_size", "sha256",
           "openness_class_token_counts")}, indent=2))
    print("\nknowledge scope:\n  " + man["knowledge_scope"])
    print(f"\nwrote {COLOPHON_FILE} (data section; train to fill the rest)")


def _train_from_args(args):
    text, paths = load_corpus(args.src)
    chars, stoi, itos = build_vocab(text)
    print(f"corpus: {len(text)} chars, {len(paths)} files, vocab {len(chars)}")
    p, man = train_model(text, stoi, chars, steps=args.steps, seed=args.seed, arch=args.arch)
    return text, paths, chars, stoi, itos, p, man


def cmd_train(args):
    text, paths, chars, stoi, itos, p, man = _train_from_args(args)
    save_weights(p, chars)
    dman = data_manifest(text, paths, chars)
    write_colophon(dman, training=man)
    print("\ntraining section:\n" + json.dumps(man, indent=2))
    print(f"saved {WEIGHTS_FILE} + {COLOPHON_FILE}")


def cmd_demo(args):
    text, paths, chars, stoi, itos, p, man = _train_from_args(args)
    K = man["context_length_K"]
    print(f"\ntrained {man['parameters']} params in "
          f"{man['wall_clock_seconds']}s, final loss {man['final_loss']:.4f}")

    print("\n--- generation from an in-distribution prompt "
          "(the model's native world) ---")
    print(repr(generate(p, stoi, itos, K,
                        prompt="weights_basemodel:\n    class: ", n=180, seed=args.seed)))

    print("\n--- confidence: in-distribution vs out-of-distribution ---")
    print("  (normalized next-char entropy: 0 = certain, 1 = no idea)")
    for label, prompts in (("IN ", IN_DIST), ("OUT", OUT_DIST)):
        for pr in prompts:
            ent, unk = prompt_confidence(p, stoi, K, pr)
            flag = f"  <-- {len(unk)} chars never seen in training" if unk else ""
            show = pr if len(pr) <= 42 else pr[:39] + "..."
            print(f"  [{label}] entropy {ent:.3f}  {show!r}{flag}")

    print("\n  Note the failure mode: the OUT prompts about the cell, a 2027")
    print("  election, or Japanese are not things this model can know. It either")
    print("  has no characters for them, or it confidently continues in its own")
    print("  world regardless -- the 'fluent, certain, and wrong' case that no")
    print("  amount of confidence-reading catches. Here you can PROVE it's wrong,")
    print("  because you can read everything it was ever trained on.")

    print_scorecard()

    save_weights(p, chars)
    dman = data_manifest(text, paths, chars)
    write_colophon(dman, training=man)
    print(f"\nsaved {WEIGHTS_FILE} + {COLOPHON_FILE} "
          "(the model's own colophon: data + training + scorecard)")


def cmd_generate(args):
    p, chars = load_weights()
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    K = _infer_K(p)
    print(generate(p, stoi, itos, K, prompt=args.prompt, n=args.n, seed=args.seed))


def main():
    ap = argparse.ArgumentParser(
        description="Colophon -- a fully-open from-scratch char LM that ships its own colophon.")
    ap.add_argument("--src", default=DEFAULT_SRC,
                    help="directory of OSAI-index .yaml files (default: bundled sample)")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", choices=["mlp", "transformer"], default="mlp",
                    help="model architecture for prepare/train/demo: 'mlp' (default, "
                         "dependency-free NumPy) or 'transformer' (optional, requires "
                         "`pip install torch`). generate reads the arch from the saved "
                         "weights, so this flag doesn't need repeating there.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    sub.add_parser("train")
    sub.add_parser("demo")
    g = sub.add_parser("generate")
    g.add_argument("--prompt", default="weights_basemodel:")
    g.add_argument("--n", type=int, default=240)

    args = ap.parse_args()
    {"prepare": cmd_prepare, "train": cmd_train,
     "demo": cmd_demo, "generate": cmd_generate}[args.cmd](args)


if __name__ == "__main__":
    main()
