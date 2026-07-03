#!/usr/bin/env python3
"""Tests for the hiragana kana teaching corpus (teaching_data/build_kana.py).

The kana corpus is the off-map teacher: a third corpus in a script the other
two never touch, so the suite finally contains a model for which the canonical
OOD prompt (日本語で書いてください) is *partly* home turf. These tests pin the
properties that make that demo work: undisputed chart facts (Hepburn spot
checks), a vocabulary disjoint from the Latin corpora (71 kana, no digits, no
uppercase), and byte-for-byte reproduction from the committed generator. The
teaching-signal class pins the payoff itself: on the kana model the unknown
characters of the canonical prompt are exactly the four kanji — off-map is a
fact about the model–data pairing, not about the text.
"""
import glob
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "teaching_data"))
import build_kana as bk  # noqa: E402

KANA_DIR = os.path.join(HERE, "teaching_data", "kana")
SAMPLE_DIR = os.path.join(HERE, "sample_data")
# The canonical OOD prompt used across the project (colophon.py, Marginalia).
CANONICAL = "日本語で書いてください"
KANJI = {"日", "本", "語", "書"}
HIRAGANA_IN_CANONICAL = {"で", "い", "て", "く", "だ", "さ"}
# Two fields, lowercase Hepburn values -> shared tokens with the Latin corpora
# shrink to lowercase letters, space, colon, newline. No digits, no uppercase.
ALLOWED_LATIN = set("abcdefghijklmnopqrstuvwxyz :\n")


class TestKanaData(unittest.TestCase):
    def test_71_kana_numbered_in_chart_order(self):
        self.assertEqual(len(bk.KANA), 71)
        self.assertEqual([k["number"] for k in bk.KANA], list(range(1, 72)))
        # Chart landmarks: gojūon starts あ, ends ん; voiced rows follow が..ぽ.
        self.assertEqual((bk.KANA[0]["kana"], bk.KANA[0]["romaji"]), ("あ", "a"))
        self.assertEqual((bk.KANA[45]["kana"], bk.KANA[45]["romaji"]), ("ん", "n"))
        self.assertEqual((bk.KANA[46]["kana"], bk.KANA[46]["romaji"]), ("が", "ga"))
        self.assertEqual((bk.KANA[70]["kana"], bk.KANA[70]["romaji"]), ("ぽ", "po"))

    def test_hepburn_spot_checks(self):
        by_kana = {k["kana"]: k["romaji"] for k in bk.KANA}
        # The Hepburn-distinctive readings a learner's chart shows.
        self.assertEqual(by_kana["し"], "shi")
        self.assertEqual(by_kana["ち"], "chi")
        self.assertEqual(by_kana["つ"], "tsu")
        self.assertEqual(by_kana["ふ"], "fu")
        self.assertEqual(by_kana["を"], "wo")
        # Hepburn collapses ぢ/づ onto ji/zu -- same romaji as じ/ず is the fact.
        self.assertEqual(by_kana["じ"], "ji")
        self.assertEqual(by_kana["ぢ"], "ji")
        self.assertEqual(by_kana["ず"], "zu")
        self.assertEqual(by_kana["づ"], "zu")
        self.assertEqual(by_kana["で"], "de")

    def test_every_canonical_hiragana_is_in_the_corpus(self):
        kana = {k["kana"] for k in bk.KANA}
        self.assertLessEqual(HIRAGANA_IN_CANONICAL, kana,
                             "the voiced kana で/だ must be included or the "
                             "flagship kanji-vs-kana split demo breaks")

    def test_no_archaic_or_small_kana(self):
        kana = {k["kana"] for k in bk.KANA}
        for excluded in "ゐゑゃゅょっ":
            self.assertNotIn(excluded, kana)

    def test_render_is_stable_two_line_schema(self):
        shi = next(k for k in bk.KANA if k["kana"] == "し")
        self.assertEqual(bk.render(shi), "kana: し\nromaji: shi\n")


class TestGeneratedCorpus(unittest.TestCase):
    def setUp(self):
        self.paths = sorted(glob.glob(os.path.join(KANA_DIR, "*.yaml")))

    def test_one_file_per_kana(self):
        self.assertEqual(len(self.paths), 71)

    def test_vocab_is_disjoint_from_the_latin_corpora(self):
        chars = set()
        for p in self.paths:
            with open(p, encoding="utf-8") as f:
                chars |= set(f.read())
        kana = {k["kana"] for k in bk.KANA}
        self.assertLessEqual(kana, chars, "every kana must appear in the corpus")
        extra = chars - kana
        self.assertLessEqual(extra, ALLOWED_LATIN,
                             f"unexpected chars: {sorted(extra - ALLOWED_LATIN)}")
        self.assertFalse(chars & set("0123456789"),
                         "no digits: keeps the vocab disjoint from elements'")

    def test_checked_in_files_reproduce_from_generator(self):
        for k in bk.KANA:
            path = os.path.join(KANA_DIR, bk.filename(k))
            self.assertTrue(os.path.exists(path), f"missing {path}")
            with open(path, encoding="utf-8") as f:
                self.assertEqual(f.read(), bk.render(k))


class TestTeachingSignals(unittest.TestCase):
    """The payoff: the same canonical prompt that is fully off-map for the
    Latin-corpus models splits cleanly on the kana model -- the four kanji
    stay unknown while the six hiragana are in-vocab. Off-map is a property
    of the model-data pairing, not of the text. Trained once for the class.
    """

    @classmethod
    def setUpClass(cls):
        import numpy as np
        import colophon as C
        cls.C = C
        text, _ = C.load_corpus(KANA_DIR)
        cls.chars, cls.stoi, cls.itos = C.build_vocab(text)
        # 3000 steps reliably reproduces the chart facts (measured); train once.
        with np.errstate(all="ignore"):
            p, man = C.train_model(text, cls.stoi, cls.chars, steps=3000, seed=0)
        cls.p, cls.K = p, man["context_length_K"]

    def test_canonical_prompt_unknowns_are_exactly_the_kanji(self):
        _, unknown = self.C.prompt_confidence(self.p, self.stoi, self.K, CANONICAL)
        self.assertEqual(set(unknown), KANJI)
        for ch in HIRAGANA_IN_CANONICAL:
            self.assertIn(ch, self.stoi, f"{ch} must be in-vocab on the kana model")

    def test_same_prompt_is_fully_off_map_for_the_osai_sample(self):
        text, _ = self.C.load_corpus(SAMPLE_DIR)
        _, stoi, _ = self.C.build_vocab(text)
        unseen = {ch for ch in CANONICAL if ch not in stoi}
        self.assertEqual(unseen, set(CANONICAL),
                         "all 10 distinct characters are unseen by the sample")
        self.assertEqual(len(set(CANONICAL)), 10)

    def test_in_distribution_prompt_reproduces_chart_facts(self):
        cont = self.C.generate(self.p, self.stoi, self.itos, self.K,
                               prompt="kana: し\n", n=30, temp=0.15, seed=0)
        cont = cont[len("kana: し\n"):]
        self.assertIn("romaji: shi", cont,
                      "し romanizes as shi -- checkable on any chart")


if __name__ == "__main__":
    unittest.main()
