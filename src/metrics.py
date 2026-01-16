"""
Metrics module for balanced accuracy calculation
"""
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from itertools import permutations


def balanced_accuracy(y_true, y_pred):
    """
    Calculate balanced accuracy (average recall across classes)

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels

    Returns:
    --------
    float
        Balanced accuracy score
    """
    return balanced_accuracy_score(y_true, y_pred)


def best_balanced_accuracy_K2(y_true, y_pred):
    """
    Compute best balanced accuracy for K=2 with permutation fix.
    Tries both identity mapping and flip mapping (0↔1), returns the max.

    Parameters:
    -----------
    y_true : array-like
        True labels (0 or 1)
    y_pred : array-like
        Predicted labels (0 or 1)

    Returns:
    --------
    float
        Best balanced accuracy (max of identity and flipped mapping)
    """
    # Identity mapping
    bac_identity = balanced_accuracy(y_true, y_pred)

    # Flip mapping (0↔1)
    y_pred_flipped = 1 - np.array(y_pred)
    bac_flipped = balanced_accuracy(y_true, y_pred_flipped)

    return max(bac_identity, bac_flipped)


def best_balanced_accuracy_K3(y_true, y_pred):
    """
    Compute best balanced accuracy for K=3 with permutation fix.
    Tries all 6 permutations of {0, 1, 2}, returns the max BAC.

    Parameters:
    -----------
    y_true : array-like
        True labels (0, 1, or 2)
    y_pred : array-like
        Predicted labels (0, 1, or 2)

    Returns:
    --------
    float
        Best balanced accuracy across all permutations
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    best_bac = 0.0

    # Try all 6 permutations of {0, 1, 2}
    for perm in permutations([0, 1, 2]):
        # Create mapping
        mapping = {0: perm[0], 1: perm[1], 2: perm[2]}

        # Apply mapping to predictions
        y_pred_mapped = np.array([mapping[label] for label in y_pred])

        # Calculate BAC for this permutation
        bac = balanced_accuracy(y_true, y_pred_mapped)
        best_bac = max(best_bac, bac)

    return best_bac


def best_balanced_accuracy(y_true, y_pred, K=None):
    """
    General function to compute best BAC for any K

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    K : int
        Number of classes (if None, inferred from data)

    Returns:
    --------
    float
        Best balanced accuracy
    """
    if K is None:
        K = max(max(y_true), max(y_pred)) + 1

    if K == 2:
        return best_balanced_accuracy_K2(y_true, y_pred)
    elif K == 3:
        return best_balanced_accuracy_K3(y_true, y_pred)
    else:
        raise ValueError(f"K={K} not supported. Only K=2 and K=3 are implemented.")