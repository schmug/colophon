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


CHAT_PROMPT = "user: what is element 26?\nmodel: "


class TestFormatIsTheOnlyVariable(unittest.TestCase):
    """Act 2's honesty rests on vocabulary ground truth that needs no
    training: the chat format's '?' does not exist in the YAML corpus's
    vocab (so the off-map flag fires on the question itself), and the
    dialogue corpus covers the chat format completely."""

    def test_question_mark_is_off_map_for_yaml_corpus(self):
        import colophon as C
        text, _ = C.load_corpus(ELEMENTS_DIR)
        chars, stoi, _ = C.build_vocab(text)
        self.assertNotIn("?", stoi)
        p = C.init_params(len(chars), 8, 4, 16, seed=0)  # untrained: vocab-only signal
        _, unknown = C.prompt_confidence(p, stoi, 4, CHAT_PROMPT)
        self.assertIn("?", unknown)

    def test_chat_format_is_fully_in_vocab_for_dialogue_corpus(self):
        import colophon as C
        text, _ = C.load_corpus(DIALOGUE_DIR)
        chars, stoi, _ = C.build_vocab(text)
        p = C.init_params(len(chars), 8, 4, 16, seed=0)
        _, unknown = C.prompt_confidence(p, stoi, 4, CHAT_PROMPT)
        self.assertEqual(unknown, [])


class TestDialogueModelAnswersInChatFormat(unittest.TestCase):
    """Act 3, demonstrated: a model trained on the dialogue corpus completes
    chat-format prompts with the right facts. K must span the question
    (~33 chars), so this trains a K=48 config once for the class.
    14000 steps reliably reproduces the Iron fact (measured; see Step 3)."""

    @classmethod
    def setUpClass(cls):
        import numpy as np
        import colophon as C
        cls.C = C
        text, _ = C.load_corpus(DIALOGUE_DIR)
        cls.chars, cls.stoi, cls.itos = C.build_vocab(text)
        with np.errstate(all="ignore"):
            p, man = C.train_model(text, cls.stoi, cls.chars,
                                   K=48, E=32, H=256, steps=14000, seed=0)
        cls.p, cls.K = p, man["context_length_K"]

    def test_element_26_answered_with_iron(self):
        out = self.C.generate(self.p, self.stoi, self.itos, self.K,
                              prompt=CHAT_PROMPT, n=60, temp=0.15, seed=0,
                              stop="\n")
        self.assertIn("Iron", out[len(CHAT_PROMPT):])

    def test_followup_question_answers_from_context(self):
        tape = ("user: what is element 26?\n"
                "model: element 26 is Iron (Fe), in period 4.\n"
                "user: which period is it in?\nmodel: ")
        out = self.C.generate(self.p, self.stoi, self.itos, self.K,
                              prompt=tape, n=60, temp=0.15, seed=0, stop="\n")
        self.assertIn("period 4", out[len(tape):])


if __name__ == "__main__":
    unittest.main()
