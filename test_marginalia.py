#!/usr/bin/env python3
"""Tests for marginalia.py's pure analysis logic (no live server needed).

Marginalia's job is to expose colophon.py's white-box signals over HTTP without
re-deriving them. These tests check that analyze_prompt() is a faithful, thin
wrapper around inspect_prompt(), and that the off-map signal it forwards
behaves the same way the demo's headline result does. The saliency wrapper is
checked the same way against colophon.context_saliency().

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

    def test_records_match_inspect_prompt(self):
        result = M.analyze_prompt(self.p, self.stoi, self.itos, self.K,
                                  self.native, n=10)
        recs = C.inspect_prompt(self.p, self.stoi, self.itos, self.K,
                                self.native, n_continuation=10)
        self.assertEqual(result["records"], recs)
        self.assertEqual(result["prompt"], self.native)
        self.assertFalse(result["off_map"])
        self.assertEqual(result["unknown_chars"], [])

    def test_prompt_entropy_still_matches_prompt_confidence(self):
        result = M.analyze_prompt(self.p, self.stoi, self.itos, self.K,
                                  self.native, n=0)
        mean_ent = sum(r["entropy"] for r in result["records"]) / len(result["records"])
        ent, _ = C.prompt_confidence(self.p, self.stoi, self.K, self.native)
        self.assertAlmostEqual(mean_ent, ent, places=9)

    def test_off_map_true_for_unseen_chars(self):
        result = M.analyze_prompt(self.p, self.stoi, self.itos, self.K, "日本語", n=5)
        self.assertTrue(result["off_map"])
        self.assertEqual(result["unknown_chars"], sorted(set("日本語")))

    def test_empty_prompt_yields_no_records(self):
        result = M.analyze_prompt(self.p, self.stoi, self.itos, self.K, "", n=5)
        self.assertEqual(result["records"], [])
        self.assertEqual(result["unknown_chars"], [])
        self.assertFalse(result["off_map"])

    def test_source_absent_when_no_files_given(self):
        result = M.analyze_prompt(self.p, self.stoi, self.itos, self.K, self.native, n=5)
        self.assertEqual(result["source"], {"matched": False})

    def test_source_uses_find_source_echo(self):
        files = [("f.yaml", self.native)]
        result = M.analyze_prompt(self.p, self.stoi, self.itos, self.K, self.native, files=files, n=5)
        self.assertEqual(result["source"], M.find_source_echo(files, self.native))
        self.assertTrue(result["source"]["matched"])


class FindSourceEcho(unittest.TestCase):
    def setUp(self):
        self.files = [("a.yaml", "0123456789foobarbaz9876543210"),
                      ("b.yaml", "another entry\nsecond line here\n")]

    def test_exact_match_reports_correct_file_and_line(self):
        result = M.find_source_echo(self.files, "second line here")
        self.assertTrue(result["matched"])
        self.assertEqual(result["file"], "b.yaml")
        self.assertEqual(result["line"], 2)
        self.assertEqual(result["match"], "second line here")

    def test_longest_suffix_backoff(self):
        # The leading "xyz " isn't in the corpus, but the trailing
        # "foobarbaz" is -- the search must back off to find it.
        result = M.find_source_echo(self.files, "xyz foobarbaz")
        self.assertTrue(result["matched"])
        self.assertEqual(result["match"], "foobarbaz")
        self.assertEqual(result["file"], "a.yaml")

    def test_floor_excludes_short_suffixes(self):
        result = M.find_source_echo(self.files, "baz", floor=4)
        self.assertFalse(result["matched"])

    def test_no_match_reports_absent(self):
        result = M.find_source_echo(self.files, "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz")
        self.assertEqual(result, {"matched": False})

    def test_highlight_offsets_bracket_the_match(self):
        result = M.find_source_echo(self.files, "foobarbaz", context=5)
        self.assertEqual(result["pre"], "56789")
        self.assertEqual(result["match"], "foobarbaz")
        self.assertEqual(result["post"], "98765")
        self.assertEqual(result["line"], 1)

    def test_match_never_spans_files(self):
        # A suffix straddling the join point of two files must not match --
        # each file is searched independently.
        files = [("x.yaml", "endsInFOO"), ("y.yaml", "BARstartsHere")]
        result = M.find_source_echo(files, "FOOBAR", floor=4)
        self.assertFalse(result["matched"])


class CorpusHelpers(unittest.TestCase):
    def test_load_corpus_files_reads_sample_data(self):
        files = M.load_corpus_files(C.DEFAULT_SRC)
        self.assertTrue(files)
        names = {name for name, _ in files}
        self.assertTrue(all(name.endswith((".yaml", ".yml")) for name in names))

    def test_corpus_sha256_matches_colophon_data_manifest(self):
        files = M.load_corpus_files(C.DEFAULT_SRC)
        text, paths = C.load_corpus(C.DEFAULT_SRC)
        chars, _, _ = C.build_vocab(text)
        manifest = C.data_manifest(text, paths, chars)
        self.assertEqual(M.corpus_sha256(files), manifest["sha256"])


class ConfidenceReadout(unittest.TestCase):
    """The layperson-facing translation of the raw entropy signal. It must be
    the INVERSE of entropy (high entropy -> low confidence), must not oversell
    an off-map prompt, and must not fabricate confidence for an empty prompt."""

    def test_confidence_is_inverse_of_entropy(self):
        # entropy 0.20 -> 80% sure; entropy 0.85 -> 15% sure.
        self.assertEqual(M.confidence_readout(0.20, [])["confidence_pct"], 80)
        self.assertEqual(M.confidence_readout(0.85, [])["confidence_pct"], 15)

    def test_low_entropy_reads_confident(self):
        r = M.confidence_readout(0.15, [])
        self.assertEqual(r["confidence_pct"], 85)
        self.assertEqual(r["verdict_level"], "confident")

    def test_off_map_overrides_and_warns_not_to_trust_number(self):
        # A fully off-map prompt can still read as moderately "sure" (~41%);
        # the verdict must flag it and tell the reader to ignore the number.
        r = M.confidence_readout(0.59, ["日", "本", "語"])
        self.assertEqual(r["confidence_pct"], 41)
        self.assertEqual(r["verdict_level"], "off-map")
        self.assertIn("never seen", r["verdict"].lower())

    def test_empty_prompt_has_no_confidence_number(self):
        r = M.confidence_readout(0.0, [], has_prompt=False)
        self.assertIsNone(r["confidence_pct"])
        self.assertEqual(r["verdict_level"], "none")

    def test_analyze_prompt_includes_readout(self):
        text, _ = C.load_corpus(C.DEFAULT_SRC)
        chars, stoi, itos = C.build_vocab(text)
        p = C.init_params(len(chars), 8, 4, 16, seed=0)
        result = M.analyze_prompt(p, stoi, itos, 4, text[:40], n=5)
        self.assertEqual(result["confidence_pct"],
                         round((1.0 - result["entropy"]) * 100))
        self.assertIn("verdict", result)
        self.assertIn(result["verdict_level"],
                      {"confident", "unsure", "struggling", "off-map"})

    def test_analyze_empty_prompt_readout_is_neutral(self):
        text, _ = C.load_corpus(C.DEFAULT_SRC)
        chars, stoi, itos = C.build_vocab(text)
        p = C.init_params(len(chars), 8, 4, 16, seed=0)
        result = M.analyze_prompt(p, stoi, itos, 4, "", n=5)
        self.assertIsNone(result["confidence_pct"])
        self.assertEqual(result["verdict_level"], "none")


class SaliencyWrapper(unittest.TestCase):
    def setUp(self):
        text, _ = C.load_corpus(C.DEFAULT_SRC)
        chars, stoi, itos = C.build_vocab(text)
        self.stoi, self.itos = stoi, itos
        self.p = C.init_params(len(chars), 8, 4, 16, seed=0)
        self.K = 4

    def test_wrapper_matches_colophon(self):
        got = M.context_saliency(self.p, self.stoi, self.itos, self.K,
                                 "weights", pos=6, n=0)
        want = C.context_saliency(self.p, self.stoi, self.itos, self.K,
                                  "weights", pos=6, n_continuation=0)
        self.assertEqual(got, want)


class ScorecardPassthrough(unittest.TestCase):
    def test_scorecard_matches_colophon(self):
        self.assertEqual(M.colophon.scorecard_section(), C.scorecard_section())


class EmbeddingsWrapper(unittest.TestCase):
    def setUp(self):
        text, _ = C.load_corpus(C.DEFAULT_SRC)
        chars, stoi, itos = C.build_vocab(text)
        self.stoi, self.itos, self.chars = stoi, itos, chars
        self.p = C.init_params(len(chars), 8, 4, 16, seed=0)

    def test_wrapper_matches_colophon(self):
        got = M.embeddings_payload(self.p, self.itos)
        want = C.embedding_projection(self.p, self.chars)
        self.assertEqual(got, want)


def _make_model():
    """A small but real model tuple, as a mode config expects it."""
    text, _ = C.load_corpus(C.DEFAULT_SRC)
    chars, stoi, itos = C.build_vocab(text)
    p = C.init_params(len(chars), 8, 4, 16, seed=0)
    return p, stoi, itos, 4


def _make_modes(model, files=()):
    """One-mode `modes` dict in the shape make_handler() now expects."""
    return {"osai": {"model": model, "files": files, "label": "Openness index",
                     "blurb": "flagship", "examples": [("ex", "weights")]}}


class _ServerFixture:
    """Boots a Handler on an ephemeral port in a background thread. The
    do_GET() routing (paths, statuses 200/400/503/404/500) is only reachable
    through a live server, so these tests exercise the class end to end."""

    def __init__(self, modes, default_mode="osai"):
        self.httpd = HTTPServer(("127.0.0.1", 0),
                                M.make_handler(modes, default_mode=default_mode))
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
        cls.server = _ServerFixture(_make_modes(_make_model()))

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
        self.assertEqual(data["prompt"], "weights")
        self.assertIn("records", data)
        self.assertIn("off_map", data)
        self.assertGreaterEqual(len(data["records"]), len("weights"))
        self.assertIn("entropy", data["records"][0])

    def test_unknown_route_404(self):
        status, _, _ = self.server.get("/nope")
        self.assertEqual(status, 404)

    def test_send_helpers_set_content_length(self):
        # _send()/_send_json() must set an accurate Content-Length header.
        status, headers, body = self.server.get("/api/scorecard")
        self.assertEqual(status, 200)
        self.assertEqual(int(headers["Content-Length"]), len(body))

    def test_saliency_route_ok(self):
        status, headers, body = self.server.get("/api/saliency?prompt=weights&pos=3")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
        data = json.loads(body)
        self.assertEqual(data["pos"], 3)
        self.assertEqual(len(data["window"]), 4)  # K == 4 in the fixture

    def test_saliency_bad_pos_400(self):
        status, _, body = self.server.get("/api/saliency?prompt=hi&pos=nope")
        self.assertEqual(status, 400)
        self.assertIn("error", json.loads(body))

    def test_saliency_out_of_range_pos_400(self):
        status, _, body = self.server.get("/api/saliency?prompt=hi&pos=999")
        self.assertEqual(status, 400)
        self.assertIn("error", json.loads(body))

    def test_embeddings_route_ok(self):
        status, headers, body = self.server.get("/api/embeddings")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
        data = json.loads(body)
        self.assertIn("points", data)
        self.assertIn("variance_explained", data)
        self.assertIn("embed_dim", data)
        self.assertEqual(len(data["variance_explained"]), 2)
        point = data["points"][0]
        for key in ("char", "display", "coords", "neighbors"):
            self.assertIn(key, point)
        self.assertEqual(len(point["coords"]), 2)


class HandlerDegraded(unittest.TestCase):
    def test_analyze_503_when_no_model(self):
        server = _ServerFixture(_make_modes(None))
        try:
            status, headers, body = server.get("/api/analyze?prompt=hi")
            self.assertEqual(status, 503)
            self.assertEqual(headers["Content-Type"],
                             "application/json; charset=utf-8")
            self.assertIn("error", json.loads(body))
        finally:
            server.close()

    def test_embeddings_503_when_no_model(self):
        server = _ServerFixture(_make_modes(None))
        try:
            status, headers, body = server.get("/api/embeddings")
            self.assertEqual(status, 503)
            self.assertEqual(headers["Content-Type"],
                             "application/json; charset=utf-8")
            self.assertIn("error", json.loads(body))
        finally:
            server.close()

    def test_scorecard_still_serves_without_model(self):
        server = _ServerFixture(_make_modes(None))
        try:
            status, _, body = server.get("/api/scorecard")
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body), C.scorecard_section())
        finally:
            server.close()

    def test_analyze_500_on_analysis_failure(self):
        # A corrupted model can make analyze_prompt() raise mid-request; the
        # handler must return a 500 JSON body rather than dropping the socket.
        server = _ServerFixture(_make_modes(_make_model()))
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


class IndexHtmlContract(unittest.TestCase):
    """The single-page inspector must ship all six regions (incl. the full
    context-window sidebar) + the black-box framing banner, and must not
    smuggle in an external dependency."""

    def test_regions_present(self):
        html = M.INDEX_HTML
        for marker in ('id="heatmap"', 'id="rail"', 'id="saliency"',
                       'id="inspector"', 'id="aggregates"', 'id="scorecard"',
                       'id="bb-banner"', 'id="sidebar"', 'id="tape"',
                       'id="embed-plot"'):
            self.assertIn(marker, html)

    def test_calls_both_apis(self):
        html = M.INDEX_HTML
        self.assertIn("/api/analyze", html)
        self.assertIn("/api/saliency", html)
        self.assertIn("/api/modes", html)
        self.assertIn("/api/embeddings", html)

    def test_no_external_dependencies(self):
        # The SVG XML namespace URI is a DOM API constant required by
        # createElementNS, not a fetched resource -- carve it out same as the
        # local server's own http://127.0.0.1.
        html = M.INDEX_HTML
        stripped = (html.replace("http://127.0.0.1", "")
                        .replace("http://www.w3.org/2000/svg", ""))
        self.assertNotIn("http://", stripped)
        self.assertNotIn("https://", html)
        self.assertNotIn("cdn", html.lower())


class ModeRouting(unittest.TestCase):
    """The teaching-mode toggle: two models served from one page, /api/analyze
    and /api/saliency route by ?mode=, and /api/modes tells the frontend what's
    available."""

    @classmethod
    def setUpClass(cls):
        model = _make_model()
        cls.modes = {
            "osai": {"model": model, "files": (), "label": "Openness index",
                     "blurb": "flagship", "examples": [("a", "weights")]},
            "elements": {"model": model, "files": (), "label": "Periodic table",
                         "blurb": "teaching", "examples": [("b", "number: 26\n")]},
        }
        cls.server = _ServerFixture(cls.modes, default_mode="osai")

    @classmethod
    def tearDownClass(cls):
        cls.server.close()

    def test_modes_endpoint_lists_both_available(self):
        status, _, body = self.server.get("/api/modes")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data["default"], "osai")
        by_id = {m["id"]: m for m in data["modes"]}
        self.assertEqual(set(by_id), {"osai", "elements"})
        self.assertTrue(by_id["osai"]["available"] and by_id["elements"]["available"])
        self.assertEqual(by_id["elements"]["examples"][0]["prompt"], "number: 26\n")

    def test_analyze_routes_to_requested_mode(self):
        status, _, body = self.server.get("/api/analyze?mode=elements&prompt=number")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["prompt"], "number")

    def test_saliency_routes_to_requested_mode(self):
        status, _, body = self.server.get("/api/saliency?mode=elements&prompt=number&pos=3")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["pos"], 3)

    def test_embeddings_routes_to_requested_mode(self):
        status, _, body = self.server.get("/api/embeddings?mode=elements")
        self.assertEqual(status, 200)
        self.assertIn("points", json.loads(body))

    def test_no_mode_uses_default(self):
        status, _, body = self.server.get("/api/analyze?prompt=weights")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["prompt"], "weights")

    def test_unknown_mode_400(self):
        status, _, body = self.server.get("/api/analyze?mode=bogus&prompt=x")
        self.assertEqual(status, 400)
        self.assertIn("error", json.loads(body))

    def test_embeddings_unknown_mode_400(self):
        status, _, body = self.server.get("/api/embeddings?mode=bogus")
        self.assertEqual(status, 400)
        self.assertIn("error", json.loads(body))


class ModeDegraded(unittest.TestCase):
    def test_absent_mode_model_503_and_marked_unavailable(self):
        modes = {
            "osai": {"model": _make_model(), "files": (), "label": "o",
                     "blurb": "", "examples": []},
            "elements": {"model": None, "files": (), "label": "e",
                         "blurb": "", "examples": []},
        }
        server = _ServerFixture(modes, default_mode="osai")
        try:
            status, _, body = server.get("/api/analyze?mode=elements&prompt=x")
            self.assertEqual(status, 503)
            self.assertIn("error", json.loads(body))
            status, _, body = server.get("/api/embeddings?mode=elements")
            self.assertEqual(status, 503)
            self.assertIn("error", json.loads(body))
            _, _, modes_body = server.get("/api/modes")
            by_id = {m["id"]: m for m in json.loads(modes_body)["modes"]}
            self.assertFalse(by_id["elements"]["available"])
            self.assertTrue(by_id["osai"]["available"])
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
