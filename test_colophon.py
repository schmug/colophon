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

import unittest
import warnings
import numpy as np

import colophon as C

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
        args = argparse.Namespace(src=src, steps=5, seed=0, arch="mlp")
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
                    "context_window", "truth_rank", "truth_prob", "off_map"):
            self.assertIn(key, r)
        self.assertEqual(len(r["context_window"]), self.K)
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


if __name__ == "__main__":
    unittest.main()
