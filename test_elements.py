#!/usr/bin/env python3
"""Tests for the periodic-table teaching corpus (teaching_data/build_elements.py).

The teaching corpus is the layperson on-ramp: a dataset whose ground truth
already lives in the reader's head, so the confidence / off-map signals can be
validated against facts they already know. These tests pin the two properties
that make it useful for that: undisputed facts (a few spot-checks) and a tiny,
clean character vocabulary (so the model stays auditable and the off-map demo
stays crisp). They also assert the checked-in files reproduce from the
generator, so the data can never silently drift from its cited source.
"""
import glob
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "teaching_data"))
import build_elements as be  # noqa: E402

ELEMENTS_DIR = os.path.join(HERE, "teaching_data", "elements")
# Undisputed facts only -> a tiny, clean vocab: letters, digits, space, colon,
# newline. No punctuation, URLs, or provenance comments leak into training text.
ALLOWED = set(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 :\n"
)


class TestElementData(unittest.TestCase):
    def test_118_elements_numbered_1_to_118(self):
        self.assertEqual(len(be.ELEMENTS), 118)
        self.assertEqual([e["number"] for e in be.ELEMENTS], list(range(1, 119)))

    def test_periods_follow_the_table(self):
        by_num = {e["number"]: e for e in be.ELEMENTS}
        self.assertEqual(by_num[1]["period"], 1)   # H
        self.assertEqual(by_num[2]["period"], 1)   # He
        self.assertEqual(by_num[3]["period"], 2)   # Li
        self.assertEqual(by_num[26]["period"], 4)  # Fe
        self.assertEqual(by_num[54]["period"], 5)  # Xe
        self.assertEqual(by_num[118]["period"], 7)  # Og

    def test_known_facts(self):
        by_num = {e["number"]: e for e in be.ELEMENTS}
        self.assertEqual((by_num[1]["symbol"], by_num[1]["name"]), ("H", "Hydrogen"))
        self.assertEqual((by_num[26]["symbol"], by_num[26]["name"]), ("Fe", "Iron"))
        self.assertEqual((by_num[79]["symbol"], by_num[79]["name"]), ("Au", "Gold"))
        self.assertEqual((by_num[118]["symbol"], by_num[118]["name"]), ("Og", "Oganesson"))

    def test_render_is_stable_four_line_schema(self):
        iron = next(e for e in be.ELEMENTS if e["number"] == 26)
        self.assertEqual(
            be.render(iron),
            "number: 26\nsymbol: Fe\nname: Iron\nperiod: 4\n",
        )


class TestGeneratedCorpus(unittest.TestCase):
    def setUp(self):
        self.paths = sorted(glob.glob(os.path.join(ELEMENTS_DIR, "*.yaml")))

    def test_one_file_per_element(self):
        self.assertEqual(len(self.paths), 118)

    def test_vocab_is_small_and_clean(self):
        chars = set()
        for p in self.paths:
            with open(p, encoding="utf-8") as f:
                chars |= set(f.read())
        self.assertLessEqual(chars, ALLOWED, f"unexpected chars: {sorted(chars - ALLOWED)}")

    def test_checked_in_files_reproduce_from_generator(self):
        for el in be.ELEMENTS:
            path = os.path.join(ELEMENTS_DIR, be.filename(el))
            self.assertTrue(os.path.exists(path), f"missing {path}")
            with open(path, encoding="utf-8") as f:
                self.assertEqual(f.read(), be.render(el))


class TestTeachingSignals(unittest.TestCase):
    """The payoff: on facts the reader already knows, the model's own signals
    behave the way Colophon teaches. Trained once for the whole class.

    Two things must hold, and they are the whole reason this corpus exists:
      * The categorical off-map flag is the trustworthy tell -- it fires on
        never-seen characters and stays silent on an all-in-vocab fake.
      * Entropy UNDER-reads out of distribution: a made-up `number: 250` is not
        meaningfully less certain than a real element, so the friendly signal
        alone would call confident invention "sure". This is the load-bearing
        lesson (CLAUDE.md), made legible on facts anyone can check.
    """

    @classmethod
    def setUpClass(cls):
        import numpy as np
        import colophon as C
        cls.C = C
        text, _ = C.load_corpus(ELEMENTS_DIR)
        cls.chars, cls.stoi, cls.itos = C.build_vocab(text)
        # 3000 steps reliably reproduces the Fe/Iron fact (measured); train once.
        with np.errstate(all="ignore"):
            p, man = C.train_model(text, cls.stoi, cls.chars, steps=3000, seed=0)
        cls.p, cls.K = p, man["context_length_K"]

    def test_off_map_flag_fires_on_foreign_characters(self):
        _, unknown = self.C.prompt_confidence(self.p, self.stoi, self.K, "日本語")
        self.assertTrue(unknown, "foreign characters must trip the off-map flag")

    def test_fake_element_is_not_caught_by_the_friendly_signals(self):
        # number: 250 is confident invention -- every character is in vocab, so
        # the off-map flag stays silent AND entropy does not spike vs a real
        # element. Only reading the (auditable) corpus catches it.
        real_ent, real_unk = self.C.prompt_confidence(self.p, self.stoi, self.K, "number: 26\n")
        fake_ent, fake_unk = self.C.prompt_confidence(self.p, self.stoi, self.K, "number: 250\n")
        self.assertEqual(fake_unk, [], "an all-in-vocab fake has no off-map chars")
        self.assertEqual(real_unk, [])
        # entropy under-reads OOD: the fake is NOT meaningfully more uncertain.
        self.assertLess(fake_ent, real_ent + 0.05,
                        "entropy should not reliably flag the fake -- that's the lesson")

    def test_in_distribution_prompt_reproduces_a_known_fact(self):
        cont = self.C.generate(self.p, self.stoi, self.itos, self.K,
                               prompt="number: 26\n", n=30, temp=0.15, seed=0)
        cont = cont[len("number: 26\n"):]
        self.assertIn("Fe", cont, "element 26 is Iron (Fe) -- ground truth in corpus")
        self.assertIn("Iron", cont)


if __name__ == "__main__":
    unittest.main()
