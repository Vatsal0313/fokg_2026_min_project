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

facts = parse_nt("/home/vatsal1310/subjects/fokg/mini_project/data/KG-2022-train.nt")
print(f"Total facts: {len(facts)}")

true_count = sum(1 for f in facts.values() if f["truth"] == 1.0)
false_count = sum(1 for f in facts.values() if f["truth"] == 0.0)
print(f"True:  {true_count}")
print(f"False: {false_count}")

# Per-predicate stats
from collections import defaultdict
pred_stats = defaultdict(lambda: {"true":0, "false":0})
for f in facts.values():
    if f["truth"] is None:
        continue
    pred = f["predicate"].split('/')[-1]   # short name
    if f["truth"] == 1.0:
        pred_stats[pred]["true"] += 1
    else:
        pred_stats[pred]["false"] += 1

for pred, stats in sorted(pred_stats.items()):
    total = stats["true"] + stats["false"]
    ratio = stats["true"] / total if total > 0 else 0
    print(f"{pred:30s}  true={stats['true']:3d}  false={stats['false']:3d}  ratio={ratio:.2f}")