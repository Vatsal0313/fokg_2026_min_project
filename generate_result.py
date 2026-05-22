"""
generate_result.py
------------------
Reads the test .nt file, runs every fact through the FactChecker,
and writes a result.ttl file in the required GERBIL submission format.

Output format (one line per fact):
    <Fact-URI> <http://swc2017.aksw.org/hasTruthValue> "value"^^<xsd:double> .

Usage:
    python generate_result.py
    python generate_result.py --test data/KG-2022-test.nt --out output/result.ttl
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from parse_data import parse_nt, get_predicate_stats
from fact_checker import FactChecker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

TRUTH_PROP = "http://swc2017.aksw.org/hasTruthValue"
XSD_DOUBLE = "http://www.w3.org/2001/XMLSchema#double"


def write_result(facts: dict, scores: dict, output_path: Path) -> None:
    """
    Write the result.ttl file.
    facts  : dict from parse_nt (keyed by URI string)
    scores : dict mapping URI string -> float score
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines_written = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for uri, fact in facts.items():
            score = scores.get(uri, 0.5)   # default 0.5 if missing
            line = (
                f"<{uri}> "
                f"<{TRUTH_PROP}> "
                f'"{score:.6f}"^^<{XSD_DOUBLE}> .\n'
            )
            f.write(line)
            lines_written += 1

    log.info(f"Written {lines_written} lines to {output_path}")


def run(test_path: str, train_path: str, output_path: str) -> None:
    # ------------------------------------------------------------------ #
    # 1. Load training data and compute predicate priors
    # ------------------------------------------------------------------ #
    log.info(f"Loading training data from {train_path} ...")
    train_facts = parse_nt(train_path)
    predicate_priors = get_predicate_stats(train_facts)
    log.info(f"Loaded {len(train_facts)} training facts.")
    log.info("Predicate priors (true ratio):")
    for pred, stats in sorted(predicate_priors.items()):
        log.info(f"  {pred:20s}  ratio={stats['ratio']:.2f}  "
                 f"(true={stats['true']}, false={stats['false']})")

    # ------------------------------------------------------------------ #
    # 2. Load test data
    # ------------------------------------------------------------------ #
    log.info(f"\nLoading test data from {test_path} ...")
    test_facts = parse_nt(test_path)
    log.info(f"Loaded {len(test_facts)} test facts to score.")

    # ------------------------------------------------------------------ #
    # 3. Set up the fact checker
    # ------------------------------------------------------------------ #
    checker = FactChecker(predicate_priors=predicate_priors)
    checker.load_cache()

    # ------------------------------------------------------------------ #
    # 4. Score every test fact
    # ------------------------------------------------------------------ #
    scores = {}
    total  = len(test_facts)
    start  = time.time()

    for i, (uri, fact) in enumerate(test_facts.items(), 1):
        score = checker.score_fact(fact)
        scores[uri] = score

        # Progress log every 50 facts
        if i % 50 == 0 or i == total:
            elapsed   = time.time() - start
            remaining = (elapsed / i) * (total - i)
            log.info(
                f"  [{i}/{total}] "
                f"score={score:.4f}  "
                f"elapsed={elapsed:.0f}s  "
                f"ETA={remaining:.0f}s  "
                f"SPARQL_calls={checker.calls_made}"
            )

        # Save cache every 100 facts so progress isn't lost on interruption
        if i % 100 == 0:
            checker.save_cache()

    # Final cache save
    checker.save_cache()

    elapsed = time.time() - start
    log.info(f"\nScoring complete. Total time: {elapsed:.1f}s, "
             f"SPARQL calls: {checker.calls_made}")

    # ------------------------------------------------------------------ #
    # 5. Write result file
    # ------------------------------------------------------------------ #
    output = Path(output_path)
    write_result(test_facts, scores, output)
    log.info(f"\nResult file ready: {output.resolve()}")

    # ------------------------------------------------------------------ #
    # 6. Quick sanity check
    # ------------------------------------------------------------------ #
    score_vals = list(scores.values())
    ones  = sum(1 for v in score_vals if v >= 0.9)
    zeros = sum(1 for v in score_vals if v <= 0.1)
    mids  = total - ones - zeros
    log.info(f"\nScore distribution:")
    log.info(f"  High (>=0.9): {ones}  ({100*ones/total:.1f}%)")
    log.info(f"  Mid  (0.1-0.9): {mids}  ({100*mids/total:.1f}%)")
    log.info(f"  Low  (<=0.1): {zeros}  ({100*zeros/total:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate result.ttl for GERBIL submission."
    )
    parser.add_argument(
        "--test",
        default="data/KG-2022-test.nt",
        help="Path to the test .nt file (default: data/KG-2022-test.nt)",
    )
    parser.add_argument(
        "--train",
        default="data/KG-2022-train.nt",
        help="Path to the train .nt file for predicate priors (default: data/KG-2022-train.nt)",
    )
    parser.add_argument(
        "--out",
        default="output/result.ttl",
        help="Output path for result.ttl (default: output/result.ttl)",
    )
    args = parser.parse_args()
    run(args.test, args.train, args.out)


if __name__ == "__main__":
    main()