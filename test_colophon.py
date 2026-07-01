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


if __name__ == "__main__":
    unittest.main()
