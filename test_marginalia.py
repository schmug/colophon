#!/usr/bin/env python3
"""Tests for marginalia.py's pure analysis logic (no live server needed).

Marginalia's job is to expose colophon.py's white-box signals over HTTP without
re-deriving them. These tests check that analyze_prompt() is a faithful, thin
wrapper around prompt_confidence() and generate(), and that the off-map signal
it forwards behaves the same way the demo's headline result does.

Run: python -m unittest test_marginalia
"""

import unittest

import colophon as C
import marginalia as M


class AnalyzePrompt(unittest.TestCase):
    def setUp(self):
        text, _ = C.load_corpus(C.DEFAULT_SRC)
        chars, stoi, itos = C.build_vocab(text)
        self.stoi, self.itos = stoi, itos
        self.p = C.init_params(len(chars), 8, 4, 16, seed=0)
        self.K = 4
        self.native = text[:50]

    def test_matches_prompt_confidence(self):
        result = M.analyze_prompt(self.p, self.stoi, self.itos, self.K, self.native, n=10)
        ent, unknown = C.prompt_confidence(self.p, self.stoi, self.K, self.native)
        self.assertAlmostEqual(result["entropy"], ent, places=9)
        self.assertEqual(result["unknown_chars"], unknown)
        self.assertFalse(result["off_map"])

    def test_off_map_true_for_unseen_chars(self):
        result = M.analyze_prompt(self.p, self.stoi, self.itos, self.K, "日本語", n=10)
        self.assertTrue(result["off_map"])
        self.assertEqual(result["unknown_chars"], sorted(set("日本語")))

    def test_continuation_is_generated_suffix(self):
        prompt = self.native[:5]
        result = M.analyze_prompt(self.p, self.stoi, self.itos, self.K, prompt, n=15, seed=1)
        full = C.generate(self.p, self.stoi, self.itos, self.K, prompt=prompt, n=15, seed=1)
        self.assertEqual(result["continuation"], full[len(prompt):])
        self.assertEqual(len(result["continuation"]), 15)

    def test_empty_prompt_does_not_crash(self):
        result = M.analyze_prompt(self.p, self.stoi, self.itos, self.K, "", n=5)
        self.assertEqual(result["entropy"], 0.0)
        self.assertEqual(result["unknown_chars"], [])
        self.assertFalse(result["off_map"])


class ScorecardPassthrough(unittest.TestCase):
    def test_scorecard_matches_colophon(self):
        self.assertEqual(M.colophon.scorecard_section(), C.scorecard_section())


if __name__ == "__main__":
    unittest.main()
