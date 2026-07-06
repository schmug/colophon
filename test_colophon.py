#!/usr/bin/env python3
"""Tests for colophon.py.

Fast, dependency-free (stdlib unittest + numpy only), and on-theme: an
auditable model deserves an auditable test suite. Covers the three things that
would silently rot if the code changed underneath the docs:

  * the manual backprop, via a finite-difference gradient check (open-work #4);
  * the off-map / unknown-character signal, the demo's headline result;
  * the unified colophon.json contract the README now promises.

Run: python -m unittest test_colophon    (or: python test_colophon.py)
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
import warnings
import numpy as np

import colophon as C

HERE = os.path.dirname(os.path.abspath(__file__))

try:
    import torch  # noqa: F401  -- only used to gate the optional transformer tests
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class GradientCheck(unittest.TestCase):
    """Finite-difference check that loss_and_grads matches numerical gradients.
    If the hand-derived backprop is wrong, this is where it shows."""

    def test_analytic_matches_numeric(self):
        rng = np.random.default_rng(0)
        V, E, K, H, B = 7, 5, 3, 8, 4
        p = C.init_params(V, E, K, H, seed=1)
        Xb = rng.integers(0, V, size=(B, K))
        Yb = rng.integers(0, V, size=B)

        _, grads = C.loss_and_grads(p, Xb, Yb)

        eps = 1e-5
        for name in p:
            flat = p[name].ravel()
            # Probe a few random coordinates per tensor -- enough to catch a
            # wrong derivation without checking every one of ~45K params.
            for idx in rng.choice(flat.size, size=min(5, flat.size), replace=False):
                orig = flat[idx]
                flat[idx] = orig + eps
                lp, _ = C.loss_and_grads(p, Xb, Yb)
                flat[idx] = orig - eps
                lm, _ = C.loss_and_grads(p, Xb, Yb)
                flat[idx] = orig
                numeric = (lp - lm) / (2 * eps)
                analytic = grads[name].ravel()[idx]
                self.assertAlmostEqual(
                    numeric, analytic, places=4,
                    msg=f"grad mismatch on {name}[{idx}]: "
                        f"numeric={numeric:.6g} analytic={analytic:.6g}")


class OffMapSignal(unittest.TestCase):
    """The categorical off-map flag: characters never seen in training must be
    reported, regardless of what entropy says."""

    def setUp(self):
        self.text, _ = C.load_corpus(C.DEFAULT_SRC)
        self.chars, self.stoi, _ = C.build_vocab(self.text)

    def test_unknown_chars_flagged(self):
        p = C.init_params(len(self.chars), 8, 4, 16, seed=0)
        # A prompt of characters guaranteed absent from an ASCII YAML corpus.
        _, unknown = C.prompt_confidence(p, self.stoi, 4, "日本語")
        self.assertEqual(unknown, sorted(unknown))
        self.assertTrue(unknown, "expected off-map characters to be flagged")

    def test_in_dist_has_no_unknowns(self):
        p = C.init_params(len(self.chars), 8, 4, 16, seed=0)
        native = self.text[:50]
        _, unknown = C.prompt_confidence(p, self.stoi, 4, native)
        self.assertEqual(unknown, [], "native text should have no off-map chars")


class DemoSchemaWarning(unittest.TestCase):
    """demo's IN_DIST/OUT_DIST prompts and sample generation are hardcoded to
    the OSAI-index schema. On a corpus that doesn't use it, demo must warn
    rather than silently mislabel in-corpus text as off-map (issue #28)."""

    def test_osai_shaped_corpus_passes(self):
        text, _ = C.load_corpus(C.DEFAULT_SRC)
        self.assertTrue(C._looks_like_osai_corpus(text))

    def test_non_osai_corpus_is_flagged(self):
        text = "number: 26\nsymbol: Fe\nname: Iron\nperiod: 4\n"
        self.assertFalse(C._looks_like_osai_corpus(text))

    def _run_demo(self, src):
        import argparse, contextlib, io, tempfile
        # out=None mirrors the CLI default added alongside --out; cmd_demo
        # resolves it to the standard colophon.npz path (redirected via C.HERE).
        args = argparse.Namespace(src=src, steps=5, seed=0, arch="mlp", out=None,
                                  K=12, E=24, H=128, lr=3e-3)
        buf = io.StringIO()
        with tempfile.TemporaryDirectory() as d:
            orig_here = C.HERE
            C.HERE = d
            try:
                with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                    with contextlib.redirect_stdout(buf):
                        C.cmd_demo(args)
            finally:
                C.HERE = orig_here
        return buf.getvalue()

    def test_demo_warns_on_non_osai_src(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "elements.yaml"), "w") as f:
                f.write("number: 26\nsymbol: Fe\nname: Iron\nperiod: 4\n" * 5)
            out = self._run_demo(d)
        self.assertIn("NOTE:", out)
        self.assertIn("marginalia.py", out)

    def test_demo_default_src_has_no_warning(self):
        out = self._run_demo(C.DEFAULT_SRC)
        self.assertNotIn("NOTE:", out)


class DisplayChar(unittest.TestCase):
    """Readable glyphs for the inspection UI; every other character is itself."""
    def test_control_and_whitespace_glyphs(self):
        self.assertEqual(C._display_char("\x00"), "∅")
        self.assertEqual(C._display_char(" "), "␣")
        self.assertEqual(C._display_char("\n"), "⏎")
        self.assertEqual(C._display_char("\t"), "⇥")

    def test_ordinary_char_passes_through(self):
        self.assertEqual(C._display_char("a"), "a")
        self.assertEqual(C._display_char(":"), ":")


class ColophonJson(unittest.TestCase):
    """The self-describing colophon.json contract the README promises: one file
    with data, training, and scorecard sections."""

    def test_build_colophon_has_all_sections(self):
        text, paths = C.load_corpus(C.DEFAULT_SRC)
        chars, stoi, _ = C.build_vocab(text)
        dman = C.data_manifest(text, paths, chars)
        # Train a tiny model so there's a real training section. The tiny matrix
        # sizes here bypass BLAS and hit NumPy's fallback matmul loop, which
        # emits spurious divide/overflow FP-flag warnings on the transposed
        # (non-contiguous) operands in the backward pass. The values are correct
        # (the gradient check proves it); scope the suppression to this call so
        # real numerical issues in production-sized training still surface.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            p, tman = C.train_model(text, stoi, chars, steps=5, log_every=5)

        col = C.build_colophon(dman, tman)
        self.assertEqual(col["name"], "Colophon")
        self.assertIn("tagline", col)
        self.assertEqual(col["data"], dman)
        self.assertEqual(col["training"], tman)

        sc = col["scorecard"]
        self.assertEqual(len(sc["dimensions"]), len(C.SCORECARD))
        self.assertEqual(sc["colophon_open"],
                         sum(1 for r in C.SCORECARD if r[1] == "G"))
        self.assertEqual(sc["typical_closed_open"],
                         sum(1 for r in C.SCORECARD if r[2] == "G"))
        # Each dimension row must be fully labelled, not raw G/P/R glyphs.
        for dim in sc["dimensions"]:
            self.assertIn(dim["colophon"], ("open", "partial", "closed"))
            self.assertIn(dim["typical_closed"], ("open", "partial", "closed"))

    def test_prepare_section_omits_training(self):
        text, paths = C.load_corpus(C.DEFAULT_SRC)
        chars, _, _ = C.build_vocab(text)
        dman = C.data_manifest(text, paths, chars)
        col = C.build_colophon(dman, training=None)
        self.assertIsNone(col["training"])
        self.assertEqual(col["data"], dman)


@unittest.skipUnless(_HAS_TORCH, "torch not installed -- --arch transformer is optional")
class TransformerArch(unittest.TestCase):
    """The optional --arch transformer path: torch is lazily imported, but once
    present it must mirror the MLP's (B, K) contexts -> (B, V) logits interface
    closely enough that generate()/prompt_confidence() work unmodified and the
    colophon.json contract still holds."""

    def setUp(self):
        self.text, self.paths = C.load_corpus(C.DEFAULT_SRC)
        self.chars, self.stoi, self.itos = C.build_vocab(self.text)

    def _tiny_transformer(self):
        return C.train_model(self.text, self.stoi, self.chars, K=4, E=8,
                              steps=3, batch=8, log_every=3, seed=0, arch="transformer")

    def test_manifest_records_arch(self):
        p, man = self._tiny_transformer()
        self.assertEqual(man["arch"], "transformer")
        self.assertEqual(p["_arch"], "transformer")
        self.assertGreater(man["parameters"], 0)

    def test_generate_and_confidence_use_shared_interface(self):
        p, man = self._tiny_transformer()
        K = man["context_length_K"]
        out = C.generate(p, self.stoi, self.itos, K, prompt="availability_", n=10, seed=0)
        self.assertTrue(out.startswith("availability_"))

        _, unknown = C.prompt_confidence(p, self.stoi, K, self.text[:20])
        self.assertEqual(unknown, [], "native text should have no off-map chars")
        _, unknown = C.prompt_confidence(p, self.stoi, K, "日本語")
        self.assertTrue(unknown, "expected off-map characters to be flagged")

    def test_colophon_json_contract_holds(self):
        p, man = self._tiny_transformer()
        dman = C.data_manifest(self.text, self.paths, self.chars)
        col = C.build_colophon(dman, man)
        self.assertEqual(col["training"]["arch"], "transformer")
        self.assertEqual(len(col["scorecard"]["dimensions"]), len(C.SCORECARD))

    def test_save_load_roundtrip(self):
        p, man = self._tiny_transformer()
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            orig_here, orig_file = C.HERE, C.WEIGHTS_FILE
            C.HERE, C.WEIGHTS_FILE = d, "colophon.npz"
            try:
                C.save_weights(p, self.chars)
                p2, chars2 = C.load_weights()
            finally:
                C.HERE, C.WEIGHTS_FILE = orig_here, orig_file
        self.assertEqual(chars2, self.chars)
        self.assertEqual(C._infer_K(p2), man["context_length_K"])
        out = C.generate(p2, self.stoi, self.itos, man["context_length_K"],
                          prompt="availability_", n=10, seed=0)
        self.assertTrue(out.startswith("availability_"))


class OutPathPlumbing(unittest.TestCase):
    """--out lets a second corpus (the periodic-table teaching model) train to
    its own weights + colophon json without clobbering the flagship
    colophon.npz. save_weights/load_weights must honor an explicit path, and the
    colophon-json path must derive from the weights path."""

    def setUp(self):
        self.text, self.paths = C.load_corpus(C.DEFAULT_SRC)
        self.chars, self.stoi, self.itos = C.build_vocab(self.text)

    def test_colophon_json_path_derivation(self):
        self.assertTrue(C.colophon_json_path("elements.npz").endswith("elements.json"))
        self.assertTrue(C.colophon_json_path("/a/b/model.npz").endswith("/a/b/model.json"))

    def test_save_and_load_explicit_path(self):
        import tempfile
        p = C.init_params(len(self.chars), 8, 4, 16, seed=0)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "elements.npz")
            C.save_weights(p, self.chars, path=path)
            self.assertTrue(os.path.exists(path))
            p2, chars2 = C.load_weights(path=path)
        self.assertEqual(chars2, self.chars)
        np.testing.assert_array_equal(p2["C"], p["C"])

    def test_two_models_coexist(self):
        import tempfile
        p_a = C.init_params(len(self.chars), 8, 4, 16, seed=0)
        p_b = C.init_params(len(self.chars), 8, 4, 16, seed=1)
        with tempfile.TemporaryDirectory() as d:
            a, b = os.path.join(d, "colophon.npz"), os.path.join(d, "elements.npz")
            C.save_weights(p_a, self.chars, path=a)
            C.save_weights(p_b, self.chars, path=b)
            self.assertTrue(os.path.exists(a) and os.path.exists(b))
            pa2, _ = C.load_weights(path=a)
            pb2, _ = C.load_weights(path=b)
        # The two files stayed independent -- writing b did not touch a.
        self.assertFalse(np.array_equal(pa2["C"], pb2["C"]))
        np.testing.assert_array_equal(pa2["C"], p_a["C"])


class InspectPrompt(unittest.TestCase):
    """Per-position white-box records: entropy must equal prompt_confidence's,
    top_k must be a real sorted distribution, truth-rank must be correct, and
    off-map chars must carry null truth fields."""

    def setUp(self):
        self.text, _ = C.load_corpus(C.DEFAULT_SRC)
        self.chars, self.stoi, self.itos = C.build_vocab(self.text)
        self.p = C.init_params(len(self.chars), 8, 4, 16, seed=0)
        self.K = 4

    def test_record_count_and_shape(self):
        recs = C.inspect_prompt(self.p, self.stoi, self.itos, self.K,
                                "weights", topk=5, n_continuation=6, seed=0)
        self.assertEqual(len(recs), len("weights") + 6)
        r = recs[0]
        for key in ("char", "display", "is_continuation", "entropy", "top_k",
                    "context_window", "context_types", "truth_rank", "truth_prob",
                    "off_map"):
            self.assertIn(key, r)
        self.assertEqual(len(r["context_window"]), self.K)
        self.assertEqual(len(r["context_types"]), self.K)
        self.assertTrue(all(recs[i]["is_continuation"] is False
                            for i in range(len("weights"))))
        self.assertTrue(recs[-1]["is_continuation"] is True)

    def test_prompt_entropy_mean_matches_prompt_confidence(self):
        text = self.text[:30]
        recs = C.inspect_prompt(self.p, self.stoi, self.itos, self.K, text,
                                n_continuation=0, seed=0)
        mean_ent = sum(r["entropy"] for r in recs) / len(recs)
        ent, _ = C.prompt_confidence(self.p, self.stoi, self.K, text)
        self.assertAlmostEqual(mean_ent, ent, places=9)

    def test_top_k_sorted_and_normalized(self):
        recs = C.inspect_prompt(self.p, self.stoi, self.itos, self.K, "class",
                                topk=5, n_continuation=0)
        top = recs[0]["top_k"]
        self.assertLessEqual(len(top), 5)
        probs = [pr for _, pr in top]
        self.assertEqual(probs, sorted(probs, reverse=True))
        self.assertTrue(all(0.0 < pr <= 1.0 for pr in probs))

    def test_truth_rank_matches_hand_computed(self):
        # Rank the actual next char in position 0's full distribution by hand.
        text = "class"
        recs = C.inspect_prompt(self.p, self.stoi, self.itos, self.K, text,
                                n_continuation=0)
        ctx = np.array([[0, 0, 0, 0]])
        logits, _ = C.forward(self.p, ctx)
        l = logits[0] - logits[0].max()
        pr = np.exp(l); pr /= pr.sum()
        cid = self.stoi[text[0]]
        expected_rank = int((pr > pr[cid]).sum()) + 1
        self.assertEqual(recs[0]["truth_rank"], expected_rank)
        self.assertAlmostEqual(recs[0]["truth_prob"], float(pr[cid]), places=9)

    def test_off_map_char_has_null_truth(self):
        recs = C.inspect_prompt(self.p, self.stoi, self.itos, self.K, "日x",
                                n_continuation=0)
        self.assertTrue(recs[0]["off_map"])
        self.assertIsNone(recs[0]["truth_rank"])
        self.assertIsNone(recs[0]["truth_prob"])
        self.assertFalse(recs[1]["off_map"])  # "x" is ASCII, in vocab

    def test_pad_slots_shown_as_glyph(self):
        recs = C.inspect_prompt(self.p, self.stoi, self.itos, self.K, "ab",
                                n_continuation=0)
        # Position 0 sees an all-pad context.
        self.assertEqual(recs[0]["context_window"], ["∅", "∅", "∅", "∅"])

    def test_context_types_distinguish_pad_offmap_prompt_continuation(self):
        # "日" is unseen (off-map), "x" is ASCII (in vocab); K=4 so the first
        # record's window is entirely synthetic pad, and later windows mix in
        # the off-map char, the rest of the prompt, and the continuation --
        # all of which collapse to the same '∅' display glyph as real pad, so
        # context_types is the only way to tell them apart.
        recs = C.inspect_prompt(self.p, self.stoi, self.itos, self.K, "日x",
                                n_continuation=2, seed=0)
        self.assertEqual(len(recs), 4)
        # Record 0 ('日'): window is [K synthetic pad slots].
        self.assertEqual(recs[0]["context_types"], ["pad"] * self.K)
        # Record 1 ('x'): window's last slot is the off-map '日', not pad,
        # even though both render as id 0 / glyph '∅'.
        self.assertEqual(recs[1]["context_types"], ["pad", "pad", "pad", "off_map"])
        # Record 2 (first continuation char): window now includes the
        # off-map char and the known prompt char 'x'.
        self.assertEqual(recs[2]["context_types"], ["pad", "pad", "off_map", "prompt"])
        # Record 3 (second continuation char): window's own predecessor is
        # itself a continuation char.
        self.assertEqual(recs[3]["context_types"],
                         ["pad", "off_map", "prompt", "continuation"])
        # No slot type outside the documented four values.
        for r in recs:
            for t in r["context_types"]:
                self.assertIn(t, ("pad", "off_map", "prompt", "continuation"))


class ContextSaliency(unittest.TestCase):
    """Occlusion attribution over the K-char window: real, model-derived
    'which remembered character mattered', not a simulated attention weight."""

    def setUp(self):
        self.text, _ = C.load_corpus(C.DEFAULT_SRC)
        self.chars, self.stoi, self.itos = C.build_vocab(self.text)
        self.p = C.init_params(len(self.chars), 8, 4, 16, seed=0)
        self.K = 4

    def test_window_shape_and_delta_range(self):
        out = C.context_saliency(self.p, self.stoi, self.itos, self.K,
                                 "weights", pos=6, n_continuation=0)
        self.assertEqual(out["pos"], 6)
        self.assertEqual(len(out["window"]), self.K)
        for cell in out["window"]:
            self.assertTrue(0.0 <= cell["delta"] <= 1.0)
            for key in ("char", "display", "delta", "is_pad"):
                self.assertIn(key, cell)

    def test_pad_slot_has_zero_delta(self):
        # Position 0 sees an all-pad context; occluding PAD with PAD changes nothing.
        out = C.context_saliency(self.p, self.stoi, self.itos, self.K,
                                 "weights", pos=0, n_continuation=0)
        for cell in out["window"]:
            self.assertTrue(cell["is_pad"])
            self.assertAlmostEqual(cell["delta"], 0.0, places=12)

    def test_delta_matches_hand_computed_tv_distance(self):
        text = "weights"
        pos = 6
        out = C.context_saliency(self.p, self.stoi, self.itos, self.K, text,
                                 pos=pos, n_continuation=0)
        ids = [0] * self.K + [self.stoi.get(c, 0) for c in text]
        ctx = ids[pos:pos + self.K]
        base = C._dist(self.p, ctx)
        occ = list(ctx); occ[0] = 0
        expected = float(0.5 * np.abs(base - C._dist(self.p, occ)).sum())
        self.assertAlmostEqual(out["window"][0]["delta"], expected, places=12)

    def test_out_of_range_pos_raises(self):
        with self.assertRaises(IndexError):
            C.context_saliency(self.p, self.stoi, self.itos, self.K, "hi",
                               pos=99, n_continuation=0)


class EmbeddingProjection(unittest.TestCase):
    """PCA-via-SVD projection of the embedding table `C`, plus cosine-similarity
    nearest neighbors over the full (un-projected) matrix."""

    def setUp(self):
        self.text, _ = C.load_corpus(C.DEFAULT_SRC)
        self.chars, self.stoi, self.itos = C.build_vocab(self.text)
        self.p = C.init_params(len(self.chars), 8, 4, 16, seed=0)

    def test_output_shape(self):
        out = C.embedding_projection(self.p, self.chars, n_components=2)
        self.assertEqual(len(out["points"]), len(self.chars))
        self.assertEqual(len(out["variance_explained"]), 2)
        self.assertEqual(out["embed_dim"], self.p["C"].shape[1])
        for pt in out["points"]:
            self.assertEqual(len(pt["coords"]), 2)
            self.assertIn("display", pt)

    def test_variance_explained_in_unit_range_and_ordered(self):
        out = C.embedding_projection(self.p, self.chars, n_components=2)
        ve = out["variance_explained"]
        self.assertTrue(all(0.0 <= v <= 1.0 for v in ve))
        self.assertGreaterEqual(ve[0], ve[1])

    def test_pca_on_known_matrix(self):
        # A hand-built embedding table where the first axis (0.1 per column)
        # carries all the variance and the second axis is a constant offset
        # (zero variance): PCA on 4 chars, E=3. The centered first column is
        # [-1.5, -0.5, 0.5, 1.5] * 0.1, columns 2-3 are constant -> after
        # centering, only column 1 has any spread, so 100% of the variance
        # must land on the first component.
        p = {"C": np.array([[0.0, 5.0, -1.0],
                            [0.1, 5.0, -1.0],
                            [0.2, 5.0, -1.0],
                            [0.3, 5.0, -1.0]])}
        chars = ["a", "b", "c", "d"]
        out = C.embedding_projection(p, chars, n_components=2)
        ve = out["variance_explained"]
        self.assertAlmostEqual(ve[0], 1.0, places=9)
        self.assertAlmostEqual(ve[1], 0.0, places=9)
        # Points must fall in the same relative order as the raw first column
        # (a < b < c < d) along the sole principal axis -- PCA's sign is
        # arbitrary, so check monotonicity rather than a specific direction.
        xs = [pt["coords"][0] for pt in out["points"]]
        diffs = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
        self.assertTrue(all(d > 0 for d in diffs) or all(d < 0 for d in diffs))

    def test_neighbors_sorted_desc_and_exclude_self(self):
        out = C.embedding_projection(self.p, self.chars, top_k=5)
        for pt in out["points"]:
            sims = [n["similarity"] for n in pt["neighbors"]]
            self.assertEqual(sims, sorted(sims, reverse=True))
            self.assertNotIn(pt["char"], [n["char"] for n in pt["neighbors"]])
            self.assertLessEqual(len(pt["neighbors"]), 5)

    def test_identical_embeddings_are_mutual_top_neighbors(self):
        # Two rows with identical (nonzero) embeddings must be perfect cosine
        # matches (similarity 1.0) and rank first for each other.
        p = {"C": np.array([[1.0, 2.0, 3.0],
                            [1.0, 2.0, 3.0],
                            [-1.0, 0.5, 2.0]])}
        chars = ["x", "y", "z"]
        out = C.embedding_projection(p, chars, top_k=2)
        by_char = {pt["char"]: pt for pt in out["points"]}
        self.assertAlmostEqual(by_char["x"]["neighbors"][0]["similarity"], 1.0, places=9)
        self.assertEqual(by_char["x"]["neighbors"][0]["char"], "y")


class AccelerateMatmulWarnings(unittest.TestCase):
    """Apple's Accelerate/vecLib BLAS spuriously raises the divide/overflow/invalid
    floating-point flags from its `matmul` path even when inputs and outputs are
    finite, cluttering the demo. colophon suppresses ONLY those matmul false
    positives -- a genuine NaN/Inf must still surface so the model stays auditable
    by eye. We stand in for the (hardware-dependent) spurious flag with a real
    matmul overflow, which reproduces deterministically on every platform."""

    def _overflowing_params(self):
        """init_params for the shapes; inflate the matmul weights so both the
        context->hidden and hidden->logits matmuls genuinely overflow to inf."""
        V, E, K, H = 7, 5, 3, 8
        p = C.init_params(V, E, K, H, seed=0)
        p["C"] = np.full_like(p["C"], 1e300)
        p["W1"] = np.full_like(p["W1"], 1e300)
        p["W2"] = np.full_like(p["W2"], 1e308)
        return p, K

    def test_matmul_warning_is_suppressed_in_forward(self):
        p, K = self._overflowing_params()
        Xb = np.zeros((4, K), dtype=np.int64)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            C.forward(p, Xb)
        leaked = [str(w.message) for w in caught if "matmul" in str(w.message)]
        self.assertEqual(leaked, [], f"matmul false positive leaked: {leaked}")

    def test_genuine_overflow_still_surfaces_as_nan_loss(self):
        # Silencing the cosmetic matmul flag must NOT hide a real blow-up: the
        # inf/nan must still propagate to an auditable NaN loss.
        p, K = self._overflowing_params()
        Xb = np.zeros((4, K), dtype=np.int64)
        Yb = np.zeros(4, dtype=np.int64)
        loss, _ = C.loss_and_grads(p, Xb, Yb)
        self.assertTrue(np.isnan(loss), "a real overflow must still surface")

    def test_training_is_matmul_warning_clean(self):
        # The end-to-end demo assertion: a real (short) training run must not
        # emit the Accelerate matmul false positives on any platform.
        text, _ = C.load_corpus(C.DEFAULT_SRC)
        chars, stoi, _ = C.build_vocab(text)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            C.train_model(text, stoi, chars, steps=40, log_every=40)
        leaked = sorted({str(w.message) for w in caught if "matmul" in str(w.message)})
        self.assertEqual(leaked, [], f"matmul warnings during training: {leaked}")


class TxtCorpusSupport(unittest.TestCase):
    """The dialogue teaching corpus is plain .txt (it is not YAML); both
    corpus loaders must pick it up alongside .yaml/.yml."""

    def test_load_corpus_reads_txt_files(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "a.txt"), "w", encoding="utf-8") as f:
                f.write("user: hi\nmodel: hello.\n")
            text, paths = C.load_corpus(d)
            self.assertEqual(len(paths), 1)
            self.assertIn("user: hi", text)

    def test_yaml_and_txt_sort_together(self):
        with tempfile.TemporaryDirectory() as d:
            for name, body in (("b.txt", "BBB"), ("a.yaml", "AAA")):
                with open(os.path.join(d, name), "w", encoding="utf-8") as f:
                    f.write(body)
            text, paths = C.load_corpus(d)
            self.assertEqual([os.path.basename(p) for p in paths],
                             ["a.yaml", "b.txt"])
            self.assertTrue(text.startswith("AAA"))


class SamplingControls(unittest.TestCase):
    """New sampling knobs on generate(): top_k, banned_ids, stop, temp<=0
    greedy. All default-off, so historical calls stay byte-identical."""

    def setUp(self):
        text, _ = C.load_corpus(C.DEFAULT_SRC)
        self.chars, self.stoi, self.itos = C.build_vocab(text)
        self.p = C.init_params(len(self.chars), 8, 4, 16, seed=0)
        self.K = 4

    def test_defaults_unchanged(self):
        a = C.generate(self.p, self.stoi, self.itos, self.K, "class", n=40, seed=3)
        b = C.generate(self.p, self.stoi, self.itos, self.K, "class", n=40, seed=3,
                       top_k=0, banned_ids=(), stop=None)
        self.assertEqual(a, b)

    def test_temp_zero_is_greedy_and_seed_independent(self):
        a = C.generate(self.p, self.stoi, self.itos, self.K, "cl", n=30, temp=0, seed=1)
        b = C.generate(self.p, self.stoi, self.itos, self.K, "cl", n=30, temp=0, seed=99)
        self.assertEqual(a, b)

    def test_top_k_1_equals_greedy(self):
        greedy = C.generate(self.p, self.stoi, self.itos, self.K, "cl", n=30, temp=0)
        topk1 = C.generate(self.p, self.stoi, self.itos, self.K, "cl", n=30,
                           temp=1.7, top_k=1, seed=5)
        self.assertEqual(greedy, topk1)

    def test_banned_char_never_sampled(self):
        out = C.generate(self.p, self.stoi, self.itos, self.K, "cl", n=120,
                         temp=1.2, seed=2, banned_ids=[self.stoi["e"]])
        self.assertNotIn("e", out[len("cl"):])

    def test_all_banned_raises(self):
        with self.assertRaises(ValueError):
            C.generate(self.p, self.stoi, self.itos, self.K, "cl", n=5,
                       banned_ids=list(range(len(self.chars))))

    def test_stop_string_halts_generation(self):
        out = C.generate(self.p, self.stoi, self.itos, self.K, "class", n=200,
                         temp=1.0, seed=0, stop="\n")
        cont = out[len("class"):]
        # The assertion must actually exercise the stop path: with seed=0 the
        # untrained model emits "\n" well before 200 chars. If a code change
        # ever makes this seed miss, pick a seed that hits -- do NOT make the
        # assertion conditional.
        self.assertIn("\n", cont,
                      "stop char never sampled -- choose a seed that hits it")
        self.assertEqual(cont.index("\n"), len(cont) - 1,
                         "generation must halt at the stop string")


class InspectPromptSampling(unittest.TestCase):
    """inspect_prompt must sample its continuation with the caller's knobs --
    the records then describe exactly what generate() would emit."""

    def setUp(self):
        text, _ = C.load_corpus(C.DEFAULT_SRC)
        self.chars, self.stoi, self.itos = C.build_vocab(text)
        self.p = C.init_params(len(self.chars), 8, 4, 16, seed=0)
        self.K = 4

    def test_records_continuation_matches_generate_with_same_knobs(self):
        kw = dict(temp=1.3, top_k=3, banned_ids=[self.stoi["e"]], seed=7)
        recs = C.inspect_prompt(self.p, self.stoi, self.itos, self.K, "class",
                                n_continuation=25, **kw)
        cont = "".join(r["char"] for r in recs if r["is_continuation"])
        want = C.generate(self.p, self.stoi, self.itos, self.K, "class", n=25, **kw)
        self.assertEqual(cont, want[len("class"):])

    def test_stop_shortens_records_not_crashes(self):
        recs = C.inspect_prompt(self.p, self.stoi, self.itos, self.K, "class",
                                n_continuation=100, seed=0, stop="e")
        cont = "".join(r["char"] for r in recs if r["is_continuation"])
        # Must exercise the stop path (same rule as SamplingControls' stop
        # test): if this seed ever stops hitting "e", pick one that does.
        self.assertIn("e", cont,
                      "stop char never sampled -- choose a seed that hits it")
        self.assertTrue(cont.endswith("e"))

    def test_default_records_unchanged(self):
        a = C.inspect_prompt(self.p, self.stoi, self.itos, self.K, "class",
                             n_continuation=10, seed=3)
        b = C.inspect_prompt(self.p, self.stoi, self.itos, self.K, "class",
                             n_continuation=10, seed=3, temp=0.8, top_k=0,
                             banned_ids=(), stop=None)
        self.assertEqual(a, b)


class GenerateUnknownCharPadMapping(unittest.TestCase):
    """generate() must map unknown prompt chars to PAD (id 0) -- the one
    convention prompt_confidence()/inspect_prompt()/_full_context_ids()
    already share. The old behavior silently SKIPPED unknowns, so the
    inspection records described a context the sampler never used."""

    def setUp(self):
        text, _ = C.load_corpus(C.DEFAULT_SRC)
        self.chars, self.stoi, self.itos = C.build_vocab(text)
        self.p = C.init_params(len(self.chars), 8, 6, 32, seed=0)
        self.K = 6
        self.unk = "日"  # not in the sample vocab
        self.assertNotIn(self.unk, self.stoi)

    def test_generate_unknown_char_equals_explicit_pad(self):
        # An unknown char and an explicit PAD ("\x00", always id 0) must
        # condition identically -- same ids in, same rng draws, same bytes out.
        pa = "class: " + self.unk + "open"
        pb = "class: " + "\x00" + "open"
        a = C.generate(self.p, self.stoi, self.itos, self.K, prompt=pa,
                       n=20, seed=3)
        b = C.generate(self.p, self.stoi, self.itos, self.K, prompt=pb,
                       n=20, seed=3)
        self.assertEqual(a[len(pa):], b[len(pb):])
        # Greedy with the unknown at the very end is the discriminating case:
        # under the old skip behavior "class: 日" conditioned exactly like
        # "class: " (measured: the two continuations diverge on this fixture),
        # so this pair fails on a regression to skipping. If a code change
        # ever makes the skip/pad continuations coincide here, pick a fixture
        # where they diverge -- do NOT weaken the pad-equality assertion.
        end_unk = "class: " + self.unk
        end_pad = "class: " + "\x00"
        end_skip = "class: "
        g_unk = C.generate(self.p, self.stoi, self.itos, self.K,
                           prompt=end_unk, n=20, temp=0)[len(end_unk):]
        g_pad = C.generate(self.p, self.stoi, self.itos, self.K,
                           prompt=end_pad, n=20, temp=0)[len(end_pad):]
        g_skip = C.generate(self.p, self.stoi, self.itos, self.K,
                            prompt=end_skip, n=20, temp=0)[len(end_skip):]
        self.assertEqual(g_unk, g_pad)
        self.assertNotEqual(g_unk, g_skip,
                            "skip/pad continuations coincide -- fixture no "
                            "longer discriminates; choose one that does")

    def test_generate_known_prompts_deterministic(self):
        # Prompts of entirely known chars are untouched by the PAD-mapping
        # change: same call, same bytes, across repeated calls.
        known = "class: open"
        a = C.generate(self.p, self.stoi, self.itos, self.K, prompt=known,
                       n=20, seed=5)
        b = C.generate(self.p, self.stoi, self.itos, self.K, prompt=known,
                       n=20, seed=5)
        self.assertEqual(a, b)
        c = C.generate(self.p, self.stoi, self.itos, self.K,
                       prompt=known + self.unk, n=20, seed=5)
        d = C.generate(self.p, self.stoi, self.itos, self.K,
                       prompt=known + "\x00", n=20, seed=5)
        self.assertEqual(c[len(known) + 1:], d[len(known) + 1:])

    def test_inspect_prompt_records_match_sampler_context_on_ood(self):
        # On an OOD prompt the continuation embedded in inspect_prompt's
        # records must be byte-identical to generate() with the same knobs --
        # the records and the sampler now share one ingestion convention.
        prompt = "class: " + self.unk + "open"
        recs = C.inspect_prompt(self.p, self.stoi, self.itos, self.K, prompt,
                                n_continuation=10, seed=7)
        cont = "".join(r["char"] for r in recs if r["is_continuation"])
        want = C.generate(self.p, self.stoi, self.itos, self.K, prompt=prompt,
                          n=10, seed=7)
        self.assertEqual(cont, want[len(prompt):])


class CliHyperparamFlags(unittest.TestCase):
    """--K/--E/--H must reach train_model and shape the saved weights. One
    training step on the bundled sample keeps this end-to-end test fast."""

    def test_train_with_K_E_H_flags_shapes_weights(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "tiny.npz")
            r = subprocess.run(
                [sys.executable, os.path.join(HERE, "colophon.py"),
                 "--steps", "1", "--K", "4", "--E", "8", "--H", "16",
                 "--out", out, "train"],
                capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 0, r.stderr)
            # allow_pickle=True is safe here: the archive was written moments
            # ago by this test's own training run (repo convention -- `chars`
            # is an object array), not untrusted input.
            saved = np.load(out, allow_pickle=True)
            self.assertEqual(saved["C"].shape[1], 8)          # E
            self.assertEqual(saved["W1"].shape, (4 * 8, 16))  # (K*E, H)
            self.assertEqual(saved["b1"].shape, (16,))

    def test_lr_flag_reaches_training(self):
        # --lr must reach train_model. A tiny 2-step run at a distinctive lr
        # trains without error and saves usable weights; the paired colophon.json
        # records the lr, proving the flag threaded through (train_model writes
        # lr into its manifest).
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "lr.npz")
            r = subprocess.run(
                [sys.executable, os.path.join(HERE, "colophon.py"),
                 "--steps", "2", "--K", "4", "--E", "8", "--H", "16",
                 "--lr", "0.0005", "--out", out, "train"],
                capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 0, r.stderr)
            json_path = C.colophon_json_path(out)
            with open(json_path) as f:
                man = json.load(f)
            self.assertEqual(man["training"]["lr"], 0.0005)


if __name__ == "__main__":
    unittest.main()
