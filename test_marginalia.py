#!/usr/bin/env python3
"""Tests for marginalia.py's pure analysis logic (no live server needed).

Marginalia's job is to expose colophon.py's white-box signals over HTTP without
re-deriving them. These tests check that analyze_prompt() is a faithful, thin
wrapper around prompt_confidence() and generate(), and that the off-map signal
it forwards behaves the same way the demo's headline result does.

Run: python -m unittest test_marginalia
"""

import http.client
import json
import os
import tempfile
import threading
import unittest
import unittest.mock
from http.server import HTTPServer

import numpy as np

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


def _make_model():
    """A small but real model tuple, as make_handler() expects it."""
    text, _ = C.load_corpus(C.DEFAULT_SRC)
    chars, stoi, itos = C.build_vocab(text)
    p = C.init_params(len(chars), 8, 4, 16, seed=0)
    return p, stoi, itos, 4


class _ServerFixture:
    """Boots a Handler on an ephemeral port in a background thread. The
    do_GET() routing (4 paths, statuses 200/503/404/500) is only reachable
    through a live server, so these tests exercise the class end to end."""

    def __init__(self, model):
        self.httpd = HTTPServer(("127.0.0.1", 0), M.make_handler(model))
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def port(self):
        return self.httpd.server_address[1]

    def get(self, path):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            conn.request("GET", path)
            resp = conn.getresponse()
            return resp.status, dict(resp.getheaders()), resp.read()
        finally:
            conn.close()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


class HandlerRouting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = _ServerFixture(_make_model())

    @classmethod
    def tearDownClass(cls):
        cls.server.close()

    def test_index_route(self):
        status, headers, body = self.server.get("/")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
        self.assertIn(b"Marginalia", body)

    def test_index_html_alias(self):
        status, _, body = self.server.get("/index.html")
        self.assertEqual(status, 200)
        self.assertIn(b"Marginalia", body)

    def test_scorecard_route(self):
        status, headers, body = self.server.get("/api/scorecard")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
        self.assertEqual(json.loads(body), C.scorecard_section())

    def test_analyze_route(self):
        status, headers, body = self.server.get("/api/analyze?prompt=weights")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
        data = json.loads(body)
        self.assertIn("entropy", data)
        self.assertIn("off_map", data)
        self.assertEqual(data["prompt"], "weights")

    def test_unknown_route_404(self):
        status, _, _ = self.server.get("/nope")
        self.assertEqual(status, 404)

    def test_send_helpers_set_content_length(self):
        # _send()/_send_json() must set an accurate Content-Length header.
        status, headers, body = self.server.get("/api/scorecard")
        self.assertEqual(status, 200)
        self.assertEqual(int(headers["Content-Length"]), len(body))


class HandlerDegraded(unittest.TestCase):
    def test_analyze_503_when_no_model(self):
        server = _ServerFixture(None)
        try:
            status, headers, body = server.get("/api/analyze?prompt=hi")
            self.assertEqual(status, 503)
            self.assertEqual(headers["Content-Type"],
                             "application/json; charset=utf-8")
            self.assertIn("error", json.loads(body))
        finally:
            server.close()

    def test_scorecard_still_serves_without_model(self):
        server = _ServerFixture(None)
        try:
            status, _, body = server.get("/api/scorecard")
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body), C.scorecard_section())
        finally:
            server.close()

    def test_analyze_500_on_analysis_failure(self):
        # A corrupted model can make analyze_prompt() raise mid-request; the
        # handler must return a 500 JSON body rather than dropping the socket.
        server = _ServerFixture(_make_model())
        try:
            with unittest.mock.patch.object(
                    M, "analyze_prompt", side_effect=ValueError("boom")):
                status, headers, body = server.get("/api/analyze?prompt=x")
            self.assertEqual(status, 500)
            self.assertEqual(headers["Content-Type"],
                             "application/json; charset=utf-8")
            self.assertIn("boom", json.loads(body)["error"])
        finally:
            server.close()


class LoadModel(unittest.TestCase):
    def _write_npz(self, tmpdir, **arrays):
        path = os.path.join(tmpdir, "model.npz")
        np.savez(path, **arrays)
        return path

    def test_round_trips_a_real_npz(self):
        text, _ = C.load_corpus(C.DEFAULT_SRC)
        chars, stoi, _ = C.build_vocab(text)
        p = C.init_params(len(chars), 8, 4, 16, seed=0)
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_npz(tmp, chars=np.array(chars, dtype=object), **p)
            lp, lstoi, litos, K = M.load_model(path)
            self.assertEqual(K, 4)
            self.assertEqual(lstoi, stoi)
            self.assertEqual(litos[0], chars[0])
            np.testing.assert_array_equal(lp["W1"], p["W1"])

    def test_missing_key_raises_keyerror(self):
        p = C.init_params(10, 8, 4, 16, seed=0)
        del p["b2"]  # drop a required array
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_npz(tmp, chars=np.array(list("abc"), dtype=object), **p)
            with self.assertRaises(KeyError):
                M.load_model(path)

    def test_zero_width_embedding_raises_valueerror(self):
        p = C.init_params(10, 8, 4, 16, seed=0)
        p["C"] = np.zeros((10, 0))  # zero-width embedding -> would divide by zero
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_npz(tmp, chars=np.array(list("abc"), dtype=object), **p)
            with self.assertRaises(ValueError):
                M.load_model(path)

    def test_missing_file_raises_filenotfound(self):
        with self.assertRaises(FileNotFoundError):
            M.load_model("/no/such/colophon.npz")


if __name__ == "__main__":
    unittest.main()
