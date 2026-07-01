"""
competence_gate.py
==================

A cheap, pre-generation gate that estimates whether a prompt lands in a zone the
acting model should be trusted to answer *unaided*. It evaluates two orthogonal
axes and combines them into a single action for an agent harness:

    1. DOMAIN competence  -- embedding distance to your own eval clusters.
                             Answers: "is this area one I've tested the model on,
                             and tested it *well*?"  (GREEN / AMBER / GREY)

    2. TEMPORAL validity  -- the query's temporal referent vs. the model's
                             knowledge cutoff. Answers: "is this fact even
                             potentially knowable by this checkpoint?"
                             (OK / STALE / UNKNOWN)

Why two axes: domain competence handles *what* the model knows; temporal validity
handles *when* its knowledge stops. They fail independently. A model can be
domain-GREEN on K-12 ed-law yet TEMPORAL-STALE on a statute amended after its
cutoff -- and the second failure is the dangerous one, because it shows up as
*confident, low-entropy, and wrong*. No output-confidence signal (entropy,
verbalized confidence, even semantic entropy) catches that reliably; the temporal
axis does, deterministically, before a single token is generated.

Design properties:
  * Pre-generation. One embedding + a regex parse + a date compare. No sampling.
  * Model-agnostic. The embedding model is separate from the *acting* model, so
    this gates closed APIs (Claude Code, no logprobs) and open weights alike.
  * Honest about thresholds. Distance cutoffs are CALIBRATED from your own data,
    not hardcoded magic numbers. See DomainMap.calibrate().
  * Retrieval-biased, not refusal-biased. A tripped gate means "ground it before
    answering," never "refuse" -- because RAG/post-training legitimately extend
    effective knowledge past the nominal cutoff.

Dependencies: numpy. (Embeddings are injected via a callable, so you choose the
backend -- sentence-transformers locally, MLX, a Workers AI endpoint, etc.)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Callable, Protocol, Sequence

import numpy as np

# --------------------------------------------------------------------------- #
# Types and enums
# --------------------------------------------------------------------------- #


class DomainState(str, Enum):
    GREEN = "green"   # close to a tested-STRONG cluster
    AMBER = "amber"   # close to a tested-WEAK cluster
    GREY = "grey"     # far from ANY tested cluster -> off-map / untested


class TemporalState(str, Enum):
    OK = "ok"            # no temporal referent, or referent <= cutoff
    STALE = "stale"      # explicit referent (year/date) is after cutoff
    UNKNOWN = "unknown"  # implicit recency ("latest", "current") -> now > cutoff


class Action(str, Enum):
    PROCEED = "proceed"  # answer unaided
    GROUND = "ground"    # must retrieve evidence before answering
    FLAG = "flag"        # proceed but surface to operator / consider a verifier


class EmbeddingFn(Protocol):
    """Any callable that maps texts -> a 2-D array of L2-comparable vectors."""
    def __call__(self, texts: Sequence[str]) -> np.ndarray: ...


# --------------------------------------------------------------------------- #
# Axis 1: domain competence via eval clusters
# --------------------------------------------------------------------------- #


@dataclass
class DomainCluster:
    """One tested area. `examples` are representative prompts you actually
    evaluated the model on; `competence` is the verdict from those evals."""
    name: str
    competence: str                 # "strong" or "weak"
    examples: list[str]
    centroid: np.ndarray | None = None
    # Max distance still considered "inside" this cluster. Set by calibrate().
    radius: float | None = None


@dataclass
class DomainMap:
    """A precomputed map of where the model is tested-strong / tested-weak.

    The map is built OFFLINE from your eval runs. At runtime it costs one
    embedding + a nearest-centroid lookup. Everything expensive (running the
    evals, fitting the radii) happens here, once, ahead of time.
    """
    embed: EmbeddingFn
    clusters: list[DomainCluster] = field(default_factory=list)

    # ---- offline construction -------------------------------------------- #

    def build(self) -> "DomainMap":
        """Compute centroids for every cluster from its example embeddings."""
        for c in self.clusters:
            vecs = _normalize(self.embed(c.examples))
            c.centroid = vecs.mean(axis=0)
        return self

    def calibrate(self, percentile: float = 90.0) -> "DomainMap":
        """Set each cluster's radius from the spread of its OWN examples.

        We use the Nth-percentile intra-cluster distance as the boundary: a new
        prompt closer than that is "inside" the tested region, farther is
        off-map. This is deliberately data-driven -- there is no universal
        distance threshold, and a hardcoded one is just an uncalibrated guess.

        Tune `percentile` against a held-out set: too low over-flags familiar
        prompts as grey; too high lets genuinely off-map prompts read as green.
        90 is a starting point, not a recommendation.
        """
        for c in self.clusters:
            if c.centroid is None:
                raise RuntimeError("call build() before calibrate()")
            vecs = _normalize(self.embed(c.examples))
            dists = _cosine_distance(vecs, c.centroid)
            c.radius = float(np.percentile(dists, percentile))
        return self

    # ---- runtime lookup -------------------------------------------------- #

    def assess(self, query: str) -> tuple[DomainState, str, float]:
        """Return (state, nearest_cluster_name, distance) for a single prompt."""
        if not self.clusters:
            return DomainState.GREY, "<empty-map>", float("inf")

        qv = _normalize(self.embed([query]))[0]
        centroids = np.vstack([c.centroid for c in self.clusters])
        dists = _cosine_distance(centroids, qv)
        i = int(np.argmin(dists))
        nearest, dist = self.clusters[i], float(dists[i])

        # Off-map: outside the calibrated radius of even the closest cluster.
        if nearest.radius is not None and dist > nearest.radius:
            return DomainState.GREY, nearest.name, dist

        state = (
            DomainState.GREEN if nearest.competence == "strong" else DomainState.AMBER
        )
        return state, nearest.name, dist


# --------------------------------------------------------------------------- #
# Axis 2: temporal referent vs. knowledge cutoff
# --------------------------------------------------------------------------- #

# Implicit recency markers. Their referent is "now" (the query-time clock), which
# in production is always after the cutoff -- so they always trip the gate. That
# is correct: "current / latest X" is never reliably answerable from parametric
# memory, no matter how confident the model sounds.
_RECENCY_MARKERS = re.compile(
    r"\b(latest|current|currently|now|today|recent(?:ly)?|"
    r"as of (?:now|today)|so far|still|these days|up to date|up-to-date|"
    r"this (?:year|month|week)|right now|nowadays|present[- ]day)\b",
    re.IGNORECASE,
)

_MONTHS = (
    "january|february|march|april|may|june|july|august|september|"
    "october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
_MONTH_YEAR = re.compile(rf"\b(?:{_MONTHS})\.?\s+(\d{{4}})\b", re.IGNORECASE)
_QUARTER_YEAR = re.compile(r"\bQ[1-4]\s*[' ]?\s*(\d{4})\b", re.IGNORECASE)
_BARE_YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")


@dataclass
class TemporalReferent:
    state: TemporalState
    detail: str            # human-readable reason
    referent: date | None  # the latest date the query points at, if any


def parse_temporal(
    query: str,
    cutoff: date,
    now: date | None = None,
) -> TemporalReferent:
    """Resolve the query's latest temporal referent and compare to `cutoff`.

    Conservative by construction: an explicit year resolves to Dec 31 of that
    year (the latest it could mean), and we take the MAX over all referents found.
    Over-flagging here is cheap (it just means "retrieve"); under-flagging risks a
    confident post-cutoff hallucination, which is the failure we most want to stop.
    """
    now = now or date.today()

    # 1) Implicit recency -> referent is "now".
    if _RECENCY_MARKERS.search(query):
        if now > cutoff:
            return TemporalReferent(
                TemporalState.UNKNOWN,
                "recency marker ('latest'/'current'/...) resolves to now, "
                f"which is after cutoff {cutoff.isoformat()}",
                now,
            )
        # Rare: model cutoff is in the future relative to the clock. No issue.
        return TemporalReferent(TemporalState.OK, "recency marker within cutoff", now)

    # 2) Explicit dates: month-year, quarter-year, bare year. Take the latest.
    years: list[int] = []
    for rx in (_MONTH_YEAR, _QUARTER_YEAR, _BARE_YEAR):
        years += [int(m) for m in rx.findall(query)]

    if years:
        latest = max(years)
        referent = date(latest, 12, 31)
        if latest > cutoff.year:
            return TemporalReferent(
                TemporalState.STALE,
                f"query references {latest}, after cutoff year {cutoff.year}",
                referent,
            )
        return TemporalReferent(
            TemporalState.OK, f"latest referenced year {latest} within cutoff", referent
        )

    # 3) No temporal referent -> timeless query, nothing to gate on this axis.
    return TemporalReferent(TemporalState.OK, "no temporal referent", None)


# --------------------------------------------------------------------------- #
# The combined gate
# --------------------------------------------------------------------------- #


@dataclass
class ModelProfile:
    name: str
    knowledge_cutoff: date  # soft floor; effective cutoff is fuzzy in practice


@dataclass
class GateDecision:
    action: Action
    domain_state: DomainState
    temporal_state: TemporalState
    nearest_cluster: str
    distance: float
    reasons: list[str]

    @property
    def badge(self) -> str:
        """One-glance operator indicator. Grey on either axis -> grey overall."""
        if self.domain_state is DomainState.GREY or self.temporal_state in (
            TemporalState.STALE,
            TemporalState.UNKNOWN,
        ):
            return "grey"
        if self.domain_state is DomainState.AMBER:
            return "amber"
        return "green"


@dataclass
class CompetenceGate:
    domain_map: DomainMap
    model: ModelProfile

    def evaluate(self, query: str, now: date | None = None) -> GateDecision:
        d_state, cluster, dist = self.domain_map.assess(query)
        temporal = parse_temporal(query, self.model.knowledge_cutoff, now=now)
        reasons: list[str] = []

        # --- action resolution ------------------------------------------- #
        # Temporal failure dominates: even a domain-strong model cannot supply a
        # fact that postdates its cutoff. Send it to retrieval regardless.
        if temporal.state in (TemporalState.STALE, TemporalState.UNKNOWN):
            reasons.append(f"temporal: {temporal.detail}")
            action = Action.GROUND
        # Off-map domain: no tested basis to trust an unaided answer. Ground if you
        # can, else at least flag for a verifier/operator.
        elif d_state is DomainState.GREY:
            reasons.append(
                f"domain: off-map (nearest '{cluster}' at distance {dist:.3f} "
                "exceeds its calibrated radius)"
            )
            action = Action.GROUND
        # Tested-weak: answer, but mark it and consider escalating to sampling or
        # a second-model verifier on consequential turns.
        elif d_state is DomainState.AMBER:
            reasons.append(f"domain: tested-weak area ('{cluster}')")
            action = Action.FLAG
        else:
            reasons.append(
                f"domain: tested-strong ('{cluster}'); temporal: {temporal.detail}"
            )
            action = Action.PROCEED

        return GateDecision(
            action=action,
            domain_state=d_state,
            temporal_state=temporal.state,
            nearest_cluster=cluster,
            distance=dist,
            reasons=reasons,
        )


# --------------------------------------------------------------------------- #
# Small numeric helpers
# --------------------------------------------------------------------------- #


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.atleast_2d(np.asarray(v, dtype=np.float32))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


def _cosine_distance(matrix: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Cosine distance (1 - cos sim) from each row of `matrix` to `vec`.
    Inputs are assumed L2-normalized."""
    sims = matrix @ vec
    return 1.0 - sims


# --------------------------------------------------------------------------- #
# Worked example
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # A toy embedding fn so the file runs with no model download. Replace with a
    # real sentence embedder (sentence-transformers, MLX, Workers AI, ...).
    # This stub just hashes tokens into a fixed space -- enough to demonstrate
    # the control flow, NOT representative of real semantic distances.
    def toy_embed(texts: Sequence[str], dim: int = 64) -> np.ndarray:
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in re.findall(r"\w+", t.lower()):
                out[i, hash(tok) % dim] += 1.0
        return out

    # Build a tiny domain map. In reality each cluster's `examples` are prompts
    # you actually ran evals on, and `competence` is the measured verdict.
    dmap = DomainMap(
        embed=toy_embed,
        clusters=[
            DomainCluster(
                "cis-control-15",
                "strong",
                [
                    "explain CIS Control 15 service provider management",
                    "how to inventory third-party vendors under CIS v8",
                    "vendor risk tiering for CIS Control 15",
                ],
            ),
            DomainCluster(
                "nc-ed-law",
                "strong",
                [
                    "difference between an LEA and a PSU under G.S. 115C",
                    "NC statutory definition of an innovative school",
                    "charter school governance under North Carolina law",
                ],
            ),
            DomainCluster(
                "obscure-crypto-trivia",
                "weak",
                [
                    "history of the Solitaire cipher in fiction",
                    "trivia about pre-DES block cipher proposals",
                ],
            ),
        ],
    ).build().calibrate(percentile=90.0)

    gate = CompetenceGate(
        domain_map=dmap,
        model=ModelProfile("local-checkpoint", knowledge_cutoff=date(2026, 1, 31)),
    )

    probes = [
        "Explain CIS Control 15 vendor inventory requirements.",   # green / proceed
        "What's the latest Qwen model and how does it score?",     # temporal -> ground
        "What changed in NC charter school law in 2026?",          # temporal -> ground
        "Tell me trivia about the Solitaire cipher.",              # amber / flag
        "Summarize the plot of an obscure 1990s anime OVA.",       # off-map -> ground
    ]

    for p in probes:
        d = gate.evaluate(p, now=date(2026, 6, 30))
        print(f"[{d.badge:>5}] {d.action.value:<8} :: {p}")
        for r in d.reasons:
            print(f"          - {r}")
