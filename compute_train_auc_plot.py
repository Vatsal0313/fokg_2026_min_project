# """
# compute_train_auc.py
# --------------------
# Computes ROC AUC on the training set and plots the ROC curve.
# """

# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_auc_score, roc_curve
# from parse_data import parse_nt, get_predicate_stats
# from fact_checker import FactChecker

# TRAIN_FILE = "data/KG-2022-train.nt"

# def main():
#     print("Loading training data...")
#     train_facts = parse_nt(TRAIN_FILE)
#     print(f"Loaded {len(train_facts)} facts.")

#     priors = get_predicate_stats(train_facts)
#     checker = FactChecker(predicate_priors=priors)
#     checker.load_cache()

#     y_true = []
#     y_scores = []

#     print("Scoring training facts...")
#     for i, (uri, fact) in enumerate(train_facts.items(), 1):
#         if fact["truth"] is None:
#             continue
#         score = checker.score_fact(fact)
#         y_true.append(fact["truth"])
#         y_scores.append(score)
#         if i % 100 == 0:
#             print(f"  Processed {i} facts...")

#     auc = roc_auc_score(y_true, y_scores)
#     fpr, tpr, _ = roc_curve(y_true, y_scores)

#     # Plot
#     plt.figure(figsize=(8, 6))
#     plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.4f})', linewidth=2)
#     plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('ROC Curve – Training Set')
#     plt.legend(loc='lower right')
#     plt.grid(alpha=0.3)
#     plt.tight_layout()
#     plt.show()

#     # Optional: save plot
#     plt.savefig('roc_curve_train.png', dpi=150)
#     print("ROC curve saved as roc_curve_train.png")

#     # Summary stats
#     true_scores = [s for t, s in zip(y_true, y_scores) if t == 1.0]
#     false_scores = [s for t, s in zip(y_true, y_scores) if t == 0.0]
#     print(f"\nROC AUC on training set: {auc:.4f}")
#     print(f"True facts:  mean score = {sum(true_scores)/len(true_scores):.4f}")
#     print(f"False facts: mean score = {sum(false_scores)/len(false_scores):.4f}")

#     checker.save_cache()

# if __name__ == "__main__":
#     main()
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from parse_data import parse_nt, get_predicate_stats
from fact_checker import FactChecker

TRAIN_FILE = "data/KG-2022-train.nt"
OUTPUT_DIR = "output"
CACHE_FILE = os.path.join(OUTPUT_DIR, "train_scores.npz")

def main(force_recompute=False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Try to load cached scores
    if not force_recompute and os.path.exists(CACHE_FILE):
        print("Loading cached training scores...")
        data = np.load(CACHE_FILE)
        y_true = data['y_true']
        y_scores = data['y_scores']
        print(f"Loaded {len(y_true)} scored facts from cache.")
    else:
        print("Loading training data and computing scores...")
        train_facts = parse_nt(TRAIN_FILE)
        priors = get_predicate_stats(train_facts)
        checker = FactChecker(predicate_priors=priors)
        checker.load_cache()

        y_true = []
        y_scores = []
        for i, (uri, fact) in enumerate(train_facts.items(), 1):
            if fact["truth"] is None:
                continue
            score = checker.score_fact(fact)
            y_true.append(fact["truth"])
            y_scores.append(score)
            if i % 100 == 0:
                print(f"  Processed {i} facts...")

        # Save cache for next time
        np.savez(CACHE_FILE, y_true=np.array(y_true), y_scores=np.array(y_scores))
        print(f"Saved training scores to {CACHE_FILE}")

        checker.save_cache()

    # Compute AUC and plot
    auc = roc_auc_score(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    # Plot and save ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve – Training Set')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'roc_curve_train.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"ROC curve saved to {plot_path}")

    # Save results text
    true_mean = np.mean(y_scores[y_true == 1])
    false_mean = np.mean(y_scores[y_true == 0])
    with open(os.path.join(OUTPUT_DIR, 'train_auc_results.txt'), 'w') as f:
        f.write(f"ROC AUC: {auc:.6f}\n")
        f.write(f"True facts mean score: {true_mean:.6f}\n")
        f.write(f"False facts mean score: {false_mean:.6f}\n")
    print(f"Results saved to {OUTPUT_DIR}/train_auc_results.txt")
    print(f"ROC AUC = {auc:.4f}")

if __name__ == "__main__":
    import sys
    force = '--force' in sys.argv
    main(force_recompute=force)