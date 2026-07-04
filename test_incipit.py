#!/usr/bin/env python3
"""Tests for incipit.py -- the multiturn glass-box chat server.

Incipit's contract is honesty plus statelessness: the tape it returns is the
exact string the model was conditioned on, every per-char record comes from
colophon.inspect_prompt() with the caller's own sampling knobs, and two
identical requests produce identical responses because the server remembers
nothing. These tests pin that contract at the pure-function level
(build_tape / parse_turn_request / run_turn) and, in Task 8's classes, over
a live ephemeral server (routing, statuses, static serving).

Run: python -m unittest test_incipit
"""
import http.client
import json
import os
import tempfile
import threading
import unittest
from http.server import HTTPServer

import colophon as C
import incipit as I


def _make_model():
    """A small but real model tuple in the (p, stoi, itos, K) shape
    marginalia.load_model() returns. Untrained: contract, not capability."""
    text, _ = C.load_corpus(C.DEFAULT_SRC)
    chars, stoi, itos = C.build_vocab(text)
    p = C.init_params(len(chars), 8, 4, 16, seed=0)
    return p, stoi, itos, 4


def _sampling(**over):
    s = {"temperature": 0.8, "top_k": 0, "seed": 0, "max_chars": 20,
         "stop": None, "banned_chars": []}
    s.update(over)
    return s


class BuildTape(unittest.TestCase):
    def test_raw_concatenates_active_turn_texts(self):
        turns = [{"role": "user", "text": "number: 26\n"},
                 {"role": "model", "text": "symbol: Fe\n"}]
        self.assertEqual(I.build_tape(turns, "raw"), "number: 26\nsymbol: Fe\n")

    def test_chat_adds_role_markers_and_generation_prefix(self):
        turns = [{"role": "user", "text": "what is element 26?"}]
        self.assertEqual(I.build_tape(turns, "chat"),
                         "user: what is element 26?\nmodel: ")

    def test_excluded_turns_vanish(self):
        turns = [{"role": "user", "text": "AAA", "excluded": True},
                 {"role": "user", "text": "BBB"}]
        self.assertEqual(I.build_tape(turns, "raw"), "BBB")

    def test_chat_multiturn_matches_training_format(self):
        turns = [{"role": "user", "text": "what is element 26?"},
                 {"role": "model", "text": "element 26 is Iron (Fe), in period 4."},
                 {"role": "user", "text": "which period is it in?"}]
        self.assertEqual(I.build_tape(turns, "chat"),
                         "user: what is element 26?\n"
                         "model: element 26 is Iron (Fe), in period 4.\n"
                         "user: which period is it in?\nmodel: ")


class ParseTurnRequest(unittest.TestCase):
    def _ok(self, **over):
        body = {"turns": [{"role": "user", "text": "hi"}]}
        body.update(over)
        return body

    def test_minimal_request_fills_defaults(self):
        req, err = I.parse_turn_request(self._ok())
        self.assertIsNone(err)
        self.assertEqual(req["format"], "raw")
        self.assertEqual(req["tape"], "hi")
        self.assertEqual(req["sampling"]["temperature"], 0.8)
        self.assertEqual(req["sampling"]["top_k"], 0)
        self.assertEqual(req["sampling"]["max_chars"], I.DEFAULT_CONTINUATION)
        self.assertIsNone(req["sampling"]["stop"])
        self.assertEqual(req["sampling"]["banned_chars"], [])

    def test_max_chars_clamped_to_cap(self):
        req, err = I.parse_turn_request(self._ok(sampling={"max_chars": 99999}))
        self.assertIsNone(err)
        self.assertEqual(req["sampling"]["max_chars"], I.MAX_CONTINUATION)

    def assert400(self, body, needle):
        req, err = I.parse_turn_request(body)
        self.assertIsNone(req)
        self.assertEqual(err[0], 400)
        self.assertIn(needle, err[1])

    def test_error_paths(self):
        self.assert400("nope", "JSON object")
        self.assert400({}, "turns")
        self.assert400({"turns": []}, "turns")
        self.assert400({"turns": [{"role": "narrator", "text": "x"}]}, "role")
        self.assert400({"turns": [{"role": "user"}]}, "role")
        self.assert400(self._ok(format="xml"), "format")
        self.assert400(self._ok(sampling={"temperature": 9}), "temperature")
        self.assert400(self._ok(sampling={"temperature": "hot"}), "numeric")
        self.assert400(self._ok(sampling={"banned_chars": ["ab"]}), "banned_chars")
        self.assert400(self._ok(sampling={"banned_chars": list("abcdefghijklmnopqrstu")}),
                       "banned_chars")
        self.assert400(self._ok(sampling={"stop": "long stop str"}), "stop")

    def test_oversized_tape_is_413(self):
        body = {"turns": [{"role": "user", "text": "x" * (I.MAX_TAPE_CHARS + 1)}]}
        req, err = I.parse_turn_request(body)
        self.assertIsNone(req)
        self.assertEqual(err[0], 413)


class RunTurn(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = _make_model()
        cls.p, cls.stoi, cls.itos, cls.K = cls.model

    def test_contract_keys(self):
        r = I.run_turn(self.model, "class", _sampling())
        for key in ("tape", "continuation", "records", "K", "entropy",
                    "unknown_chars", "off_map", "banned_applied",
                    "confidence_pct", "verdict_level", "verdict", "source"):
            self.assertIn(key, r)
        self.assertEqual(r["K"], self.K)
        self.assertEqual(r["tape"], "class")

    def test_continuation_matches_generate_with_same_knobs(self):
        s = _sampling(temperature=1.1, top_k=3, seed=5, banned_chars=["e"])
        r = I.run_turn(self.model, "class", s)
        want = C.generate(self.p, self.stoi, self.itos, self.K, "class",
                          n=s["max_chars"], temp=1.1, top_k=3, seed=5,
                          banned_ids=[self.stoi["e"]])
        self.assertEqual(r["continuation"], want[len("class"):])
        cont_from_records = "".join(
            rec["char"] for rec in r["records"] if rec["is_continuation"])
        self.assertEqual(r["continuation"], cont_from_records)

    def test_banned_applied_skips_chars_outside_vocab(self):
        s = _sampling(banned_chars=["e", "日"])
        r = I.run_turn(self.model, "class", s)
        self.assertEqual(r["banned_applied"], ["e"])

    def test_stateless_determinism(self):
        a = I.run_turn(self.model, "class", _sampling(seed=9))
        b = I.run_turn(self.model, "class", _sampling(seed=9))
        self.assertEqual(a, b)

    def test_empty_tape_yields_no_records_and_no_confidence(self):
        r = I.run_turn(self.model, "", _sampling())
        self.assertEqual(r["records"], [])
        self.assertEqual(r["continuation"], "")
        self.assertIsNone(r["confidence_pct"])

    def test_source_echo_uses_corpus_files(self):
        files = [("f.yaml", "class contents here")]
        r = I.run_turn(self.model, "class", _sampling(), files=files)
        self.assertTrue(r["source"]["matched"])


if __name__ == "__main__":
    unittest.main()
