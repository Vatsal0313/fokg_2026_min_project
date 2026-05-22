"""
parse_data.py
-------------
Parses RDF N-Triples (.nt) files into structured fact dictionaries.

Each fact in the dataset is an rdf:Statement reification with:
  - a URI identifying the statement
  - rdf:subject   → the entity the fact is about
  - rdf:predicate → the relationship type
  - rdf:object    → the target entity
  - hasTruthValue → (train only) 0.0 = false, 1.0 = true
"""

import re
from typing import Optional

# Namespace shortcuts
NS_TYPE      = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"
NS_SUBJECT   = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#subject>"
NS_PREDICATE = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate>"
NS_OBJECT    = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#object>"
NS_TRUTH     = "<http://swc2017.aksw.org/hasTruthValue>"


def parse_nt(filepath: str) -> dict:
    """
    Parse an N-Triples file and return a dict keyed by statement URI.

    Each value is a dict with keys:
        uri        - the full statement URI string (no angle brackets)
        subject    - DBpedia subject URI (no angle brackets)
        predicate  - DBpedia predicate URI (no angle brackets)
        object     - DBpedia object URI (no angle brackets)
        truth      - float (1.0 or 0.0) if present, else None (test set)

    Example:
        facts = parse_nt("data/KG-2022-train.nt")
        for uri, fact in facts.items():
            print(fact["subject"], fact["predicate"], fact["object"], fact["truth"])
    """
    raw = {}  # uri_string -> {predicate_string -> object_string}
    STATEMENT_TYPE = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement>"
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Split into 3 parts: subject, predicate, object
            parts = line.split(' ', 2)
            if len(parts) < 3:
                continue
            s = parts[0]
            p = parts[1]
            obj_part = parts[2]
            # Remove the trailing ' .' correctly
            if obj_part.endswith(' .'):
                o = obj_part[:-2]
            else:
                o = obj_part.rstrip('\n')
            raw.setdefault(s, {})[p] = o

    facts = {}
    for uri_bracketed, props in raw.items():
        if props.get(NS_TYPE) != STATEMENT_TYPE:
            continue
        uri = uri_bracketed.strip('<>')
        truth = None
        if NS_TRUTH in props:
            val = props[NS_TRUTH]
            # Extract number from "0.0"^^<...>
            if val.startswith('"'):
                num_str = val.split('"')[1]
                try:
                    truth = float(num_str)
                except ValueError:
                    pass
        facts[uri] = {
            "uri": uri,
            "subject": props.get(NS_SUBJECT, '').strip('<>'),
            "predicate": props.get(NS_PREDICATE, '').strip('<>'),
            "object": props.get(NS_OBJECT, '').strip('<>'),
            "truth": truth,
        }
    return facts


def get_predicate_stats(facts: dict) -> dict:
    """
    Compute per-predicate true/false counts from labelled facts.
    Returns dict: predicate_short_name -> {"true": int, "false": int, "ratio": float}

    Useful for understanding the dataset and calibrating scores.
    """
    from collections import defaultdict
    stats = defaultdict(lambda: {"true": 0, "false": 0})

    for fact in facts.values():
        if fact["truth"] is None:
            continue
        short = fact["predicate"].split("/")[-1]
        if fact["truth"] == 1.0:
            stats[short]["true"] += 1
        else:
            stats[short]["false"] += 1

    for short, counts in stats.items():
        total = counts["true"] + counts["false"]
        counts["ratio"] = counts["true"] / total if total > 0 else 0.5

    return dict(stats)


if __name__ == "__main__":
    # Quick smoke test
    print("=== TRAIN ===")
    train = parse_nt("data/KG-2022-train.nt")
    print(f"Total facts: {len(train)}")
    true_c  = sum(1 for f in train.values() if f["truth"] == 1.0)
    false_c = sum(1 for f in train.values() if f["truth"] == 0.0)
    print(f"True:  {true_c}")
    print(f"False: {false_c}")

    print("\nPer-predicate stats:")
    stats = get_predicate_stats(train)
    for pred, s in sorted(stats.items()):
        print(f"  {pred:20s}  true={s['true']:3d}  false={s['false']:3d}  ratio={s['ratio']:.2f}")

    print("\n=== TEST ===")
    test = parse_nt("data/KG-2022-test.nt")
    print(f"Total facts: {len(test)}")
    print("Sample:")
    for i, (uri, f) in enumerate(test.items()):
        if i >= 3:
            break
        print(f"  {f['subject'].split('/')[-1]} --[{f['predicate'].split('/')[-1]}]--> {f['object'].split('/')[-1]}")