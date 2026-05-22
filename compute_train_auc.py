"""
compute_train_auc.py
--------------------
Computes ROC AUC on the training set using your trained FactChecker.
"""

from sklearn.metrics import roc_auc_score
from parse_data import parse_nt, get_predicate_stats
from fact_checker import FactChecker

# Paths – adjust if needed
TRAIN_FILE = "data/KG-2022-train.nt"

def main():
    print("Loading training data...")
    train_facts = parse_nt(TRAIN_FILE)
    print(f"Loaded {len(train_facts)} facts.")

    # Compute predicate priors (required by FactChecker)
    priors = get_predicate_stats(train_facts)

    # Initialise fact checker (will load cache if exists)
    checker = FactChecker(predicate_priors=priors)
    checker.load_cache()

    y_true = []
    y_scores = []

    print("Scoring training facts...")
    for i, (uri, fact) in enumerate(train_facts.items(), 1):
        if fact["truth"] is None:
            continue
        score = checker.score_fact(fact)
        y_true.append(fact["truth"])
        y_scores.append(score)
        if i % 100 == 0:
            print(f"  Processed {i} facts...")

    auc = roc_auc_score(y_true, y_scores)
    print(f"\nROC AUC on training set: {auc:.4f}")

    # Optional: print distribution of scores for true/false
    true_scores = [s for t, s in zip(y_true, y_scores) if t == 1.0]
    false_scores = [s for t, s in zip(y_true, y_scores) if t == 0.0]
    print(f"\nTrue facts:  mean score = {sum(true_scores)/len(true_scores):.4f}")
    print(f"False facts: mean score = {sum(false_scores)/len(false_scores):.4f}")

    checker.save_cache()   # optional

if __name__ == "__main__":
    main()