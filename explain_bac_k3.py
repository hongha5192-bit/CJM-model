"""
Explain BAC calculation for K=3 regime detection
"""

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from itertools import permutations
import matplotlib.pyplot as plt
import seaborn as sns

def best_balanced_accuracy_K3(s_pred, s_true):
    """
    Compute best balanced accuracy over all 6 permutations for K=3
    This handles the label switching problem in unsupervised learning
    """
    K = 3
    best_bac = 0
    best_perm = None
    all_bacs = []

    for perm in permutations(range(K)):
        # Map predictions using this permutation
        mapping = {old: new for old, new in enumerate(perm)}
        s_pred_mapped = pd.Series(s_pred).map(mapping).values

        # Compute BAC
        bac = balanced_accuracy_score(s_true, s_pred_mapped)
        all_bacs.append((perm, bac))

        if bac > best_bac:
            best_bac = bac
            best_perm = perm

    return best_bac, best_perm, all_bacs

# Example: Simulate a case where labels are switched
np.random.seed(42)
n = 300

# True states: 0=Bullish, 1=Neutral, 2=Bearish
s_true = np.array([0]*100 + [1]*100 + [2]*100)

# Predicted states with label switching
# Model learned: 0->2, 1->0, 2->1 (switched labels)
s_pred_switched = np.array([2]*90 + [0]*5 + [1]*5 +    # True 0 -> Pred 2 (90% correct)
                           [0]*85 + [2]*10 + [1]*5 +    # True 1 -> Pred 0 (85% correct)
                           [1]*80 + [0]*10 + [2]*10)    # True 2 -> Pred 1 (80% correct)

print("=" * 80)
print("BALANCED ACCURACY CALCULATION FOR K=3 REGIME DETECTION")
print("=" * 80)

# 1. Calculate naive BAC (without permutation)
naive_bac = balanced_accuracy_score(s_true, s_pred_switched)
print("\n1. NAIVE APPROACH (without handling label switching):")
print("-" * 50)

# Show confusion matrix
cm_naive = confusion_matrix(s_true, s_pred_switched)
print("Confusion Matrix (rows=true, cols=predicted):")
print(cm_naive)
print()

# Calculate recall for each class
print("Class-wise Recall:")
for i in range(3):
    tp = cm_naive[i, i]
    fn = cm_naive[i, :].sum() - tp
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"  State {i}: {tp}/{tp+fn} = {recall:.3f}")

print(f"\nNaive BAC = {naive_bac:.3f}")
print("(This is LOW because labels are switched!)")

# 2. Calculate permutation-invariant BAC
print("\n2. PERMUTATION-INVARIANT APPROACH:")
print("-" * 50)

best_bac, best_perm, all_bacs = best_balanced_accuracy_K3(s_pred_switched, s_true)

print("Testing all 6 permutations:")
for perm, bac in sorted(all_bacs, key=lambda x: -x[1]):
    mapping_str = f"0->{perm[0]}, 1->{perm[1]}, 2->{perm[2]}"
    print(f"  Permutation ({mapping_str}): BAC = {bac:.3f}")

print(f"\nBest permutation: 0->{best_perm[0]}, 1->{best_perm[1]}, 2->{best_perm[2]}")
print(f"Best BAC = {best_bac:.3f}")

# Apply best permutation and show confusion matrix
mapping = {old: new for old, new in enumerate(best_perm)}
s_pred_corrected = pd.Series(s_pred_switched).map(mapping).values

cm_corrected = confusion_matrix(s_true, s_pred_corrected)
print("\nConfusion Matrix after best permutation:")
print(cm_corrected)

print("\nClass-wise Recall after correction:")
for i in range(3):
    tp = cm_corrected[i, i]
    fn = cm_corrected[i, :].sum() - tp
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"  State {i}: {tp}/{tp+fn} = {recall:.3f}")

# 3. Explanation of BAC formula
print("\n3. BAC FORMULA EXPLANATION:")
print("-" * 50)
print("Balanced Accuracy (BAC) for K classes:")
print("  BAC = (1/K) × Σ(Recall_i)")
print()
print("For K=3 regime detection:")
print("  BAC = (Recall_Bullish + Recall_Neutral + Recall_Bearish) / 3")
print()
print("Why use BAC instead of regular accuracy?")
print("  - Handles imbalanced classes well")
print("  - Each regime gets equal weight regardless of frequency")
print("  - More robust for evaluating regime detection performance")

# 4. Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Naive confusion matrix
sns.heatmap(cm_naive, annot=True, fmt='d', cmap='YlOrRd', ax=axes[0],
            xticklabels=['Pred 0', 'Pred 1', 'Pred 2'],
            yticklabels=['True 0', 'True 1', 'True 2'])
axes[0].set_title(f'Naive Approach\nBAC = {naive_bac:.3f}')

# Plot 2: Corrected confusion matrix
sns.heatmap(cm_corrected, annot=True, fmt='d', cmap='YlGnBu', ax=axes[1],
            xticklabels=['Pred 0', 'Pred 1', 'Pred 2'],
            yticklabels=['True 0', 'True 1', 'True 2'])
axes[1].set_title(f'After Best Permutation\nBAC = {best_bac:.3f}')

# Plot 3: BAC values for all permutations
perms_str = [f"{p[0]}{p[1]}{p[2]}" for p, _ in all_bacs]
bac_values = [bac for _, bac in all_bacs]
axes[2].bar(perms_str, bac_values, color=['green' if b == best_bac else 'gray' for b in bac_values])
axes[2].set_xlabel('Permutation')
axes[2].set_ylabel('BAC')
axes[2].set_title('BAC for All Permutations')
axes[2].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('outputs/bac_explanation_k3.png', dpi=100, bbox_inches='tight')
plt.show()

print("\nVisualization saved to: outputs/bac_explanation_k3.png")
print("\n" + "=" * 80)
print("KEY TAKEAWAY:")
print("The permutation-invariant BAC finds the best label alignment")
print("between predicted and true states, solving the label switching")
print("problem inherent in unsupervised regime detection.")
print("=" * 80)