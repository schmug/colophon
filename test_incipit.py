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

    def test_non_finite_sampling_is_400_not_crash(self):
        for field in ("temperature", "top_k", "seed", "max_chars"):
            for bad in (float("inf"), float("nan")):
                req, err = I.parse_turn_request(self._ok(sampling={field: bad}))
                self.assertIsNone(req, f"{field}={bad} should be rejected")
                self.assertEqual(err[0], 400, f"{field}={bad} should 400")

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


def _make_modes(model, files=()):
    """Two-mode dict in the shape make_handler() expects: one available
    (elements64), one absent (dialogue) -- exercises 200 and 503 paths."""
    return {
        "elements64": {"model": model, "files": files,
                       "params": C.n_params(model[0]) if model else None,
                       **I.MODE_META["elements64"]},
        "dialogue": {"model": None, "files": (), "params": None,
                     **I.MODE_META["dialogue"]},
    }


class _ServerFixture:
    """Boots the Handler on an ephemeral port in a daemon thread (the
    test_marginalia.py pattern, plus POST support)."""

    def __init__(self, modes, default_mode="elements64", dist_dir=None):
        handler = I.make_handler(
            modes, default_mode=default_mode,
            dist_dir=dist_dir or os.path.join(I.HERE, "no-such-dist"))
        self.httpd = HTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       daemon=True)
        self.thread.start()

    @property
    def port(self):
        return self.httpd.server_address[1]

    def _request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            conn.request(method, path, body, headers or {})
            resp = conn.getresponse()
            return resp.status, dict(resp.getheaders()), resp.read()
        finally:
            conn.close()

    def get(self, path):
        return self._request("GET", path)

    def post(self, path, obj=None, raw=None):
        body = raw if raw is not None else json.dumps(obj).encode("utf-8")
        return self._request("POST", path, body,
                             {"Content-Type": "application/json",
                              "Content-Length": str(len(body))})

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


def _turn_body(**over):
    body = {"mode": "elements64",
            "turns": [{"role": "user", "text": "class"}],
            "format": "raw",
            "sampling": {"seed": 4, "max_chars": 15}}
    body.update(over)
    return body


INCIPIT_SOURCE_FILES = (("q.txt", "user: hi\nmodel: hello\n"),
                        ("evil.txt", '<script>bad()</script>\n'))


class ModeProvenance(unittest.TestCase):
    def test_every_mode_documents_its_source(self):
        for mid, meta in I.MODE_META.items():
            self.assertTrue(meta.get("source_note"),
                            f"{mid} is missing a source_note")


class IncipitSourceRoute(unittest.TestCase):
    """GET /source serves one training file from a mode's in-memory corpus --
    Incipit's own route (it links to no other server). Covers the dialogue-
    shaped corpus that has no home in Marginalia."""

    @classmethod
    def setUpClass(cls):
        cls.server = _ServerFixture(
            _make_modes(_make_model(), files=INCIPIT_SOURCE_FILES))

    @classmethod
    def tearDownClass(cls):
        cls.server.close()

    def test_serves_dialogue_shaped_file_with_highlight(self):
        status, headers, body = self.server.get(
            "/source?mode=elements64&file=q.txt&line=2")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
        page = body.decode("utf-8")
        self.assertIn("model: hello", page)
        self.assertIn('<tr id="L2" class="hit">', page)

    def test_traversal_name_404_html(self):
        status, headers, _ = self.server.get(
            "/source?mode=elements64&file=../incipit.py")
        self.assertEqual(status, 404)
        self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")

    def test_bad_line_400_html(self):
        status, headers, _ = self.server.get(
            "/source?mode=elements64&file=q.txt&line=nope")
        self.assertEqual(status, 400)
        self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")

    def test_absent_model_503_html(self):
        status, headers, _ = self.server.get("/source?mode=dialogue&file=q.txt")
        self.assertEqual(status, 503)
        self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")

    def test_unknown_mode_400_html(self):
        status, headers, _ = self.server.get("/source?mode=nope&file=q.txt")
        self.assertEqual(status, 400)
        self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")

    def test_contents_escaped(self):
        _, _, body = self.server.get("/source?mode=elements64&file=evil.txt")
        self.assertNotIn(b"<script", body)
        self.assertIn(b"&lt;script&gt;", body)

    def test_not_swallowed_by_static(self):
        # /source is matched before _serve_static; the fixture's dist_dir does
        # not exist, so a mis-ordered route would return the build-help page.
        _, _, body = self.server.get("/source?mode=elements64&file=q.txt")
        self.assertNotIn(b"npm run build", body)


class IncipitCorpusRoute(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = _ServerFixture(
            _make_modes(_make_model(), files=INCIPIT_SOURCE_FILES))

    @classmethod
    def tearDownClass(cls):
        cls.server.close()

    def test_lists_files_as_html(self):
        status, headers, body = self.server.get("/corpus?mode=elements64")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
        page = body.decode("utf-8")
        self.assertIn("file=q.txt", page)
        self.assertIn("file=evil.txt", page)

    def test_footer_has_provenance_note(self):
        _, _, body = self.server.get("/corpus?mode=elements64")
        self.assertIn(b"build_elements.py", body)

    def test_unknown_mode_400_html(self):
        status, headers, _ = self.server.get("/corpus?mode=nope")
        self.assertEqual(status, 400)
        self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")

    def test_absent_model_503_html(self):
        status, headers, _ = self.server.get("/corpus?mode=dialogue")
        self.assertEqual(status, 503)
        self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")

    def test_not_swallowed_by_static(self):
        _, _, body = self.server.get("/corpus?mode=elements64")
        self.assertNotIn(b"npm run build", body)


class ServerRouting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = _ServerFixture(_make_modes(_make_model()))

    @classmethod
    def tearDownClass(cls):
        cls.server.close()

    def test_modes_payload(self):
        status, headers, body = self.server.get("/api/modes")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data["default"], "elements64")
        by_id = {m["id"]: m for m in data["modes"]}
        self.assertTrue(by_id["elements64"]["available"])
        self.assertFalse(by_id["dialogue"]["available"])
        self.assertEqual(by_id["elements64"]["K"], 4)
        self.assertEqual(by_id["elements64"]["acts"], [1, 2])
        self.assertIn("train_hint", by_id["dialogue"])

    def test_turn_contract(self):
        status, _, body = self.server.post("/api/turn", _turn_body())
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data["tape"], "class")
        self.assertEqual(data["format"], "raw")
        self.assertEqual(len(data["continuation"]), 15)
        self.assertIn("records", data)
        self.assertIn("confidence_pct", data)

    def test_statelessness_identical_requests_identical_bytes(self):
        _, _, a = self.server.post("/api/turn", _turn_body())
        _, _, b = self.server.post("/api/turn", _turn_body())
        self.assertEqual(a, b)

    def test_unknown_mode_400(self):
        status, _, body = self.server.post("/api/turn",
                                           _turn_body(mode="gpt4"))
        self.assertEqual(status, 400)
        self.assertIn("error", json.loads(body))

    def test_untrained_mode_503_with_train_hint(self):
        status, _, body = self.server.post("/api/turn",
                                           _turn_body(mode="dialogue"))
        self.assertEqual(status, 503)
        self.assertIn("dialogue_k64", json.loads(body)["error"])

    def test_malformed_json_400(self):
        status, _, body = self.server.post("/api/turn", raw=b"{nope")
        self.assertEqual(status, 400)

    def test_oversized_tape_413(self):
        body = _turn_body(turns=[{"role": "user",
                                  "text": "x" * (I.MAX_TAPE_CHARS + 1)}])
        status, _, _ = self.server.post("/api/turn", body)
        self.assertEqual(status, 413)

    def test_saliency_ok(self):
        body = {"mode": "elements64", "text": "class contents", "pos": 6}
        status, _, resp = self.server.post("/api/saliency", body)
        self.assertEqual(status, 200)
        data = json.loads(resp)
        self.assertEqual(data["pos"], 6)
        self.assertEqual(len(data["window"]), 4)  # K == 4 in the fixture

    def test_saliency_bad_pos_400(self):
        for pos in (999, "nope", None):
            body = {"mode": "elements64", "text": "class", "pos": pos}
            status, _, _ = self.server.post("/api/saliency", body)
            self.assertEqual(status, 400, f"pos={pos!r}")

    def test_scorecard_route(self):
        status, _, body = self.server.get("/api/scorecard")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), C.scorecard_section())

    def test_unknown_api_404(self):
        status, _, _ = self.server.get("/api/nope")
        self.assertEqual(status, 404)


class StaticServing(unittest.TestCase):
    def test_missing_dist_serves_build_help_page(self):
        server = _ServerFixture(_make_modes(_make_model()))
        try:
            status, headers, body = server.get("/")
            self.assertEqual(status, 200)
            self.assertIn(b"npm run build", body)
        finally:
            server.close()

    def test_dist_files_served_with_mime_types(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "index.html"), "w") as f:
                f.write("<title>INCIPIT-TEST-PAGE</title>")
            with open(os.path.join(d, "app.js"), "w") as f:
                f.write("console.log(1)")
            server = _ServerFixture(_make_modes(_make_model()), dist_dir=d)
            try:
                status, headers, body = server.get("/")
                self.assertEqual(status, 200)
                self.assertIn(b"INCIPIT-TEST-PAGE", body)
                status, headers, _ = server.get("/app.js")
                self.assertEqual(status, 200)
                self.assertIn("javascript", headers["Content-Type"])
            finally:
                server.close()

    def test_path_traversal_is_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "index.html"), "w") as f:
                f.write("ok")
            server = _ServerFixture(_make_modes(_make_model()), dist_dir=d)
            try:
                status, _, body = server.get("/../colophon.py")
                self.assertNotEqual(status, 200)
            finally:
                server.close()


if __name__ == "__main__":
    unittest.main()
