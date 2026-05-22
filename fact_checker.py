"""
fact_checker.py
---------------
Fact-checking engine using DBpedia SPARQL lookups.

Scoring logic (returns a float between 0.0 and 1.0):

  1.0  - Triple found directly in DBpedia  → almost certainly true
  0.8  - Subject found, object found, but triple not present
           → entities are real but relationship is wrong
  0.4  - Only subject found in DBpedia
           → object may be wrong/unrecognised
  0.2  - Neither subject nor object found
           → likely fabricated fact
  0.0  - SPARQL error fallback → treated as unknown (returns 0.5)

A predicate-level prior (learned from training data) is blended in
to further improve AUC beyond raw triple lookup.
"""

import json
import time
import logging
from pathlib import Path
from typing import Optional

from SPARQLWrapper import SPARQLWrapper, JSON, SPARQLExceptions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DBPEDIA_ENDPOINT = "https://dbpedia.org/sparql"
CACHE_FILE       = Path("cache/sparql_cache.json")
REQUEST_DELAY    = 0.15   # seconds between SPARQL calls (be polite)
TIMEOUT          = 15     # seconds per SPARQL request

# Score constants (before predicate prior blending)
SCORE_TRIPLE_EXISTS      = 1.0
# SCORE_BOTH_ENTITIES      = 0.8
SCORE_BOTH_ENTITIES      = 0.35
# SCORE_SUBJECT_ONLY       = 0.4
SCORE_SUBJECT_ONLY       = 0.25
# SCORE_NOTHING_FOUND      = 0.2
SCORE_NOTHING_FOUND      = 0.1
SCORE_ERROR              = 0.5   # fallback when SPARQL fails

# How much weight to give the predicate prior vs the SPARQL score
# 0.0 = ignore prior, 1.0 = ignore SPARQL score
# PRIOR_BLEND_WEIGHT = 0.15
PRIOR_BLEND_WEIGHT = 0.25


# --------------------------------------------------------------------------- #
# Cache helpers
# --------------------------------------------------------------------------- #

def load_cache() -> dict:
    """Load the on-disk SPARQL result cache."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict) -> None:
    """Persist the cache to disk."""
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


# --------------------------------------------------------------------------- #
# SPARQL helpers
# --------------------------------------------------------------------------- #

def _make_sparql() -> SPARQLWrapper:
    sparql = SPARQLWrapper(DBPEDIA_ENDPOINT)
    sparql.setReturnFormat(JSON)
    sparql.setTimeout(TIMEOUT)
    sparql.addCustomHttpHeader(
        "User-Agent",
        "KG-FactChecker/1.0 (university project; contact via uni-paderborn.de)"
    )
    return sparql


def ask_triple(sparql: SPARQLWrapper, subject: str, predicate: str, obj: str) -> Optional[bool]:
    """
    ASK whether (subject, predicate, object) exists in DBpedia.
    Returns True/False, or None on error.
    """
    query = f"""
    ASK {{
        <{subject}> <{predicate}> <{obj}> .
    }}
    """
    try:
        sparql.setQuery(query)
        result = sparql.query().convert()
        return result.get("boolean", None)
    except Exception as e:
        log.warning(f"ASK triple error: {e}")
        return None


def ask_entity_exists(sparql: SPARQLWrapper, uri: str) -> Optional[bool]:
    """
    ASK whether a URI exists as a subject in DBpedia.
    Returns True/False, or None on error.
    """
    query = f"""
    ASK {{
        <{uri}> ?p ?o .
    }}
    """
    try:
        sparql.setQuery(query)
        result = sparql.query().convert()
        return result.get("boolean", None)
    except Exception as e:
        log.warning(f"ASK entity error ({uri}): {e}")
        return None


# --------------------------------------------------------------------------- #
# Main FactChecker class
# --------------------------------------------------------------------------- #

class FactChecker:
    """
    Checks facts against DBpedia and returns a veracity score [0, 1].

    Usage:
        checker = FactChecker(predicate_priors)
        checker.load_cache()
        score = checker.score(subject_uri, predicate_uri, object_uri)
        checker.save_cache()
    """

    def __init__(self, predicate_priors: Optional[dict] = None):
        """
        predicate_priors: dict mapping short predicate name -> ratio of true facts
                          (from get_predicate_stats() on training data).
                          If None, no prior blending is applied.
        """
        self.predicate_priors = predicate_priors or {}
        self.cache: dict = {}
        self.sparql = _make_sparql()
        self._calls_made = 0

    def load_cache(self) -> None:
        self.cache = load_cache()
        log.info(f"Loaded {len(self.cache)} cached SPARQL results.")

    def save_cache(self) -> None:
        save_cache(self.cache)
        log.info(f"Saved {len(self.cache)} entries to cache.")

    # ------------------------------------------------------------------ #
    # Internal query with caching
    # ------------------------------------------------------------------ #

    def _cached_ask_triple(self, subject: str, predicate: str, obj: str) -> Optional[bool]:
        key = f"triple|{subject}|{predicate}|{obj}"
        if key in self.cache:
            return self.cache[key]

        time.sleep(REQUEST_DELAY)
        self._calls_made += 1
        result = ask_triple(self.sparql, subject, predicate, obj)
        if result is not None:
            self.cache[key] = result
        return result

    def _cached_ask_entity(self, uri: str) -> Optional[bool]:
        key = f"entity|{uri}"
        if key in self.cache:
            return self.cache[key]

        time.sleep(REQUEST_DELAY)
        self._calls_made += 1
        result = ask_entity_exists(self.sparql, uri)
        if result is not None:
            self.cache[key] = result
        return result

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #

    def _sparql_score(self, subject: str, predicate: str, obj: str) -> float:
        """
        Multi-level SPARQL score:
          - Check if the full triple exists
          - If not, check subject and object individually
        """
        triple_exists = self._cached_ask_triple(subject, predicate, obj)

        if triple_exists is None:
            # SPARQL error — return neutral score
            return SCORE_ERROR

        if triple_exists:
            return SCORE_TRIPLE_EXISTS

        # Triple not found — check entities individually
        subj_exists = self._cached_ask_entity(subject)
        obj_exists  = self._cached_ask_entity(obj)

        if subj_exists and obj_exists:
            return SCORE_BOTH_ENTITIES
        elif subj_exists:
            return SCORE_SUBJECT_ONLY
        else:
            return SCORE_NOTHING_FOUND

    def _blend_with_prior(self, sparql_score: float, predicate: str) -> float:
        """
        Blend the SPARQL score with the per-predicate prior from training data.
        prior = fraction of facts with this predicate that are true in training set.
        """
        short = predicate.split("/")[-1]
        if short not in self.predicate_priors:
            return sparql_score

        prior = self.predicate_priors[short]["ratio"]
        blended = (1 - PRIOR_BLEND_WEIGHT) * sparql_score + PRIOR_BLEND_WEIGHT * prior
        return round(blended, 6)

    def score(self, subject: str, predicate: str, obj: str) -> float:
        """
        Return a veracity score in [0, 1] for the triple (subject, predicate, object).
        This is the main method to call.
        """
        sparql_score = self._sparql_score(subject, predicate, obj)
        final_score  = self._blend_with_prior(sparql_score, predicate)
        return final_score

    def score_fact(self, fact: dict) -> float:
        """
        Convenience wrapper: pass a fact dict from parse_data.parse_nt().
        """
        return self.score(fact["subject"], fact["predicate"], fact["object"])

    @property
    def calls_made(self) -> int:
        return self._calls_made


if __name__ == "__main__":
    # Quick test on a few known facts
    from parse_data import parse_nt, get_predicate_stats

    train = parse_nt("data/KG-2022-train.nt")
    priors = get_predicate_stats(train)

    checker = FactChecker(predicate_priors=priors)
    checker.load_cache()

    test_cases = [
        # Known true: Venus Williams born in Lynwood, California
        ("http://dbpedia.org/resource/Venus_Williams",
         "http://dbpedia.org/ontology/birthPlace",
         "http://dbpedia.org/resource/Lynwood,_California"),
        # Known false: David Lee played for Houston Rockets
        ("http://dbpedia.org/resource/David_Lee_(basketball)",
         "http://dbpedia.org/ontology/team",
         "http://dbpedia.org/resource/Houston_Rockets"),
    ]

    for subj, pred, obj in test_cases:
        s = checker.score(subj, pred, obj)
        print(f"Score: {s:.4f}  |  {subj.split('/')[-1]} -- {pred.split('/')[-1]} --> {obj.split('/')[-1]}")

    checker.save_cache()
    print(f"\nTotal SPARQL calls made: {checker.calls_made}")