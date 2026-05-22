"""
compute_train_auc.py
--------------------
Computes ROC AUC on the training set, plots the ROC curve,
and saves results to the output folder.
"""

import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from parse_data import parse_nt, get_predicate_stats
from fact_checker import FactChecker

TRAIN_FILE = "data/KG-2022-train.nt"
OUTPUT_DIR = "output"

def main():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading training data...")
    train_facts = parse_nt(TRAIN_FILE)
    print(f"Loaded {len(train_facts)} facts.")

    priors = get_predicate_stats(train_facts)
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
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    # Plot and save ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve – Training Set')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, 'roc_curve_train.png')
    plt.savefig(plot_path, dpi=150)
    print(f"ROC curve saved to {plot_path}")
    plt.close()  # Close to avoid display if running non-interactively

    # Compute statistics
    true_scores = [s for t, s in zip(y_true, y_scores) if t == 1.0]
    false_scores = [s for t, s in zip(y_true, y_scores) if t == 0.0]
    true_mean = sum(true_scores) / len(true_scores) if true_scores else 0
    false_mean = sum(false_scores) / len(false_scores) if false_scores else 0

    # Save AUC and stats to text file
    results_path = os.path.join(OUTPUT_DIR, 'train_auc_results.txt')
    with open(results_path, 'w') as f:
        f.write("Training Set Evaluation Results\n")
        f.write("===============================\n\n")
        f.write(f"ROC AUC score: {auc:.6f}\n\n")
        f.write(f"Number of true facts: {len(true_scores)}\n")
        f.write(f"Number of false facts: {len(false_scores)}\n\n")
        f.write(f"Mean score for true facts: {true_mean:.6f}\n")
        f.write(f"Mean score for false facts: {false_mean:.6f}\n")
    print(f"Results saved to {results_path}")

    # Print to console
    print(f"\nROC AUC on training set: {auc:.4f}")
    print(f"True facts:  mean score = {true_mean:.4f}")
    print(f"False facts: mean score = {false_mean:.4f}")

    checker.save_cache()

if __name__ == "__main__":
    main()