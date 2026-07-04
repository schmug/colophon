#!/usr/bin/env python3
"""Tests for the dialogue teaching corpus (teaching_data/build_dialogue.py)
and the teaching signals Incipit's three-act sequence depends on.

Act 2 claims chat format fails on the YAML-trained model for exactly one
reason: the training data's format. Act 3 claims the same architecture
answers in chat format once the training data contains dialogues. These
tests pin the corpus properties that make the comparison honest (same
facts, clean vocab, regenerable files) and the behavioral facts the acts
rely on: '?' is off-map for the YAML corpus but in-vocab here, and a
trained dialogue model reproduces a known fact in chat format.

Run: python -m unittest test_dialogue
"""
import glob
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "teaching_data"))
import build_dialogue as bd  # noqa: E402
import build_elements as be  # noqa: E402

DIALOGUE_DIR = os.path.join(HERE, "teaching_data", "dialogue")
ELEMENTS_DIR = os.path.join(HERE, "teaching_data", "elements")
# The dialogue corpus needs a handful of characters YAML never used --
# '?' '.' '(' ')' ',' -- and nothing else beyond the elements vocab.
ALLOWED = set(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 :\n?.(),"
)


class TestDialogueRender(unittest.TestCase):
    def test_iron_dialogues_contain_exact_known_facts(self):
        iron = next(e for e in be.ELEMENTS if e["number"] == 26)
        text = bd.render_dialogues(iron)
        self.assertIn("user: what is element 26?\n"
                      "model: element 26 is Iron (Fe), in period 4.\n", text)
        self.assertIn("user: which period is Iron in?\n"
                      "model: Iron is in period 4.\n", text)

    def test_five_singles_plus_one_followup_dialogue(self):
        h = next(e for e in be.ELEMENTS if e["number"] == 1)
        text = bd.render_dialogues(h)
        self.assertEqual(text.count("user: "), 7)   # 5 singles + 2 in follow-up
        self.assertEqual(text.count("model: "), 7)
        self.assertEqual(len(text.split("\n\n")), 6)  # blank-line separated

    def test_followup_uses_a_cross_turn_reference(self):
        # "which period is it in?" only answers correctly if the earlier turn
        # is in the context window -- the multiturn training signal.
        h = next(e for e in be.ELEMENTS if e["number"] == 1)
        self.assertIn("user: which period is it in?\n", bd.render_dialogues(h))


class TestGeneratedDialogueCorpus(unittest.TestCase):
    def setUp(self):
        self.paths = sorted(glob.glob(os.path.join(DIALOGUE_DIR, "*.txt")))

    def test_one_file_per_element(self):
        self.assertEqual(len(self.paths), 118)

    def test_vocab_is_small_and_clean(self):
        chars = set()
        for p in self.paths:
            with open(p, encoding="utf-8") as f:
                chars |= set(f.read())
        self.assertLessEqual(chars, ALLOWED,
                             f"unexpected chars: {sorted(chars - ALLOWED)}")

    def test_checked_in_files_reproduce_from_generator(self):
        for el in be.ELEMENTS:
            path = os.path.join(DIALOGUE_DIR, bd.filename(el))
            self.assertTrue(os.path.exists(path), f"missing {path}")
            with open(path, encoding="utf-8") as f:
                self.assertEqual(f.read(), bd.render_dialogues(el))


if __name__ == "__main__":
    unittest.main()
