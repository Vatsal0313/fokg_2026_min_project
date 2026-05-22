"""
plot_test_distribution.py
-------------------------
Plots the distribution of fact-checking scores on the test set.
(No ROC – labels are unknown for test set.)
"""

import matplotlib.pyplot as plt
from parse_data import parse_nt, get_predicate_stats
from fact_checker import FactChecker

TEST_FILE = "data/KG-2022-test.nt"
TRAIN_FILE = "data/KG-2022-train.nt"

def main():
    print("Loading training data for priors...")
    train_facts = parse_nt(TRAIN_FILE)
    priors = get_predicate_stats(train_facts)

    print("Loading test data...")
    test_facts = parse_nt(TEST_FILE)
    print(f"Loaded {len(test_facts)} test facts.")

    checker = FactChecker(predicate_priors=priors)
    checker.load_cache()

    scores = []
    print("Scoring test facts...")
    for i, (uri, fact) in enumerate(test_facts.items(), 1):
        score = checker.score_fact(fact)
        scores.append(score)
        if i % 100 == 0:
            print(f"  Processed {i} facts...")

    # Plot histogram
    plt.figure(figsize=(8, 5))
    plt.hist(scores, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Truth Score (0 = false, 1 = true)')
    plt.ylabel('Number of facts')
    plt.title('Distribution of Fact‑Checking Scores on Test Set')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"\nScore statistics on test set:")
    print(f"  Min: {min(scores):.4f}")
    print(f"  Max: {max(scores):.4f}")
    print(f"  Mean: {sum(scores)/len(scores):.4f}")

    checker.save_cache()

if __name__ == "__main__":
    main()