"""
Verify BAC calculation and investigate why lambda values give similar results
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from itertools import permutations
import warnings
warnings.filterwarnings('ignore')

# Import JumpModel
from jumpmodels.jump import JumpModel

def best_balanced_accuracy_K3(s_pred, s_true):
    """Compute best balanced accuracy over all 6 permutations for K=3"""
    K = 3
    best_bac = 0
    best_perm = None
    best_cm = None

    for perm in permutations(range(K)):
        mapping = {old: new for old, new in enumerate(perm)}
        s_pred_mapped = pd.Series(s_pred).map(mapping).values
        bac = balanced_accuracy_score(s_true, s_pred_mapped)

        if bac > best_bac:
            best_bac = bac
            best_perm = perm
            best_cm = confusion_matrix(s_true, s_pred_mapped)

    return best_bac, best_perm, best_cm

def analyze_predictions(predictions_dict, s_true):
    """Analyze how different the predictions are across lambda values"""
    lambdas = sorted(predictions_dict.keys())
    n_samples = len(s_true)

    print("\n" + "="*80)
    print("PREDICTION ANALYSIS ACROSS LAMBDA VALUES")
    print("="*80)

    # 1. Compare prediction distributions
    print("\n1. STATE DISTRIBUTION FOR EACH LAMBDA:")
    print("-"*50)
    for lam in lambdas:
        s_pred = predictions_dict[lam]
        counts = np.bincount(s_pred, minlength=3)
        pcts = counts / n_samples * 100
        print(f"λ={lam:3.0f}: State 0={pcts[0]:5.1f}%, State 1={pcts[1]:5.1f}%, State 2={pcts[2]:5.1f}%")

    # 2. Agreement between different lambda predictions
    print("\n2. AGREEMENT MATRIX (% of samples with same prediction):")
    print("-"*50)
    print("      ", end="")
    for lam in lambdas:
        print(f"λ={lam:3.0f} ", end="")
    print()

    for i, lam1 in enumerate(lambdas):
        print(f"λ={lam1:3.0f}: ", end="")
        for j, lam2 in enumerate(lambdas):
            if j < i:
                print("     ", end="")
            else:
                agreement = np.mean(predictions_dict[lam1] == predictions_dict[lam2]) * 100
                print(f"{agreement:4.0f}% ", end="")
        print()

    # 3. Number of state transitions
    print("\n3. NUMBER OF STATE TRANSITIONS:")
    print("-"*50)
    for lam in lambdas:
        s_pred = predictions_dict[lam]
        n_transitions = np.sum(s_pred[1:] != s_pred[:-1])
        print(f"λ={lam:3.0f}: {n_transitions:3d} transitions ({n_transitions/n_samples*100:.1f}% of time points)")

    # 4. Samples that change prediction across lambda values
    print("\n4. PREDICTION STABILITY:")
    print("-"*50)
    all_preds = np.array([predictions_dict[lam] for lam in lambdas])

    # For each sample, count how many different predictions it gets
    n_unique_preds = [len(np.unique(all_preds[:, i])) for i in range(n_samples)]

    always_same = np.sum(np.array(n_unique_preds) == 1)
    sometimes_diff = np.sum(np.array(n_unique_preds) == 2)
    often_diff = np.sum(np.array(n_unique_preds) == 3)

    print(f"Samples with same prediction for ALL lambdas: {always_same} ({always_same/n_samples*100:.1f}%)")
    print(f"Samples with 2 different predictions: {sometimes_diff} ({sometimes_diff/n_samples*100:.1f}%)")
    print(f"Samples with 3 different predictions: {often_diff} ({often_diff/n_samples*100:.1f}%)")

# Load one simulation for testing
print("Loading simulation data...")
sim_path = Path('outputs/step2_sim/K3_price_based/sim_0000.parquet')
sim_data = pd.read_parquet(sim_path)

# Extract features and true states
feature_names = ['ADX', 'DMI_Plus', 'DMI_Minus', 'URSI', 'BB_PCTB', 'BBWP']
X_raw = sim_data[feature_names].values
ret_ser = sim_data['returns'].values
s_true = sim_data['s_true'].values

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

print(f"Data shape: {X.shape}")
print(f"True state distribution: {np.bincount(s_true)}")

# Test different lambda values
lambda_values = [5, 10, 15, 20, 50, 100]
predictions = {}
bac_scores = {}

print("\n" + "="*80)
print("TESTING CJM WITH DIFFERENT LAMBDA VALUES")
print("="*80)

for lambda_val in lambda_values:
    print(f"\n--- Lambda = {lambda_val} ---")

    # Fit model
    model = JumpModel(
        n_components=3,
        jump_penalty=lambda_val,
        cont=True,
        mode_loss=True,
        grid_size=0.05,
        n_init=3,
        max_iter=200,
        tol=1e-5,
        random_state=123,
        verbose=0
    )

    # Fit with returns for sorting
    model.fit(X, ret_ser=ret_ser, sort_by='cumret')

    # Get predictions
    s_pred = model.predict(X)
    predictions[lambda_val] = s_pred

    # Calculate BAC
    bac, best_perm, best_cm = best_balanced_accuracy_K3(s_pred, s_true)
    bac_scores[lambda_val] = bac

    # Show results
    print(f"Predicted state distribution: {np.bincount(s_pred, minlength=3)}")
    print(f"Best permutation: {best_perm}")
    print(f"BAC = {bac:.4f}")

    # Show confusion matrix for best permutation
    print("Confusion matrix (after best permutation):")
    print(best_cm)

# Analyze the differences in predictions
analyze_predictions(predictions, s_true)

# Plot BAC scores
print("\n" + "="*80)
print("SUMMARY OF BAC SCORES")
print("="*80)
for lam, bac in bac_scores.items():
    print(f"λ={lam:3.0f}: BAC = {bac:.4f}")

print(f"\nRange of BAC scores: {min(bac_scores.values()):.4f} to {max(bac_scores.values()):.4f}")
print(f"Difference: {max(bac_scores.values()) - min(bac_scores.values()):.4f}")

# Check if lambda is actually affecting the jump penalty
print("\n" + "="*80)
print("VERIFYING LAMBDA EFFECT ON JUMP PENALTY")
print("="*80)

# Create a simple test case with obvious regime changes
np.random.seed(42)
T = 300
# Create 3 clear regimes with sudden jumps
regime_true = np.array([0]*100 + [1]*100 + [2]*100)

# Generate features that clearly distinguish regimes
X_test = np.zeros((T, 6))
for i in range(T):
    if regime_true[i] == 0:
        X_test[i] = np.random.normal([1, 1, 1, 0, 0, 0], 0.1)
    elif regime_true[i] == 1:
        X_test[i] = np.random.normal([0, 0, 0, 1, 1, 1], 0.1)
    else:
        X_test[i] = np.random.normal([-1, -1, -1, -1, -1, -1], 0.1)

# Generate returns based on regimes
ret_test = np.zeros(T)
ret_test[regime_true == 0] = np.random.normal(0.002, 0.01, sum(regime_true == 0))
ret_test[regime_true == 1] = np.random.normal(0, 0.01, sum(regime_true == 1))
ret_test[regime_true == 2] = np.random.normal(-0.002, 0.01, sum(regime_true == 2))

print("\nTesting with synthetic data having clear regime boundaries...")
print(f"True regimes: {100} samples each of states 0, 1, 2")

test_predictions = {}
for lambda_val in [1, 10, 100, 1000]:
    model = JumpModel(
        n_components=3,
        jump_penalty=lambda_val,
        cont=True,
        mode_loss=True,
        grid_size=0.05,
        n_init=1,
        max_iter=200,
        tol=1e-5,
        random_state=123,
        verbose=0
    )

    # Standardize test data
    X_test_scaled = StandardScaler().fit_transform(X_test)
    model.fit(X_test_scaled, ret_ser=ret_test, sort_by='cumret')

    s_pred_test = model.predict(X_test_scaled)
    n_transitions = np.sum(s_pred_test[1:] != s_pred_test[:-1])

    test_predictions[lambda_val] = s_pred_test
    print(f"λ={lambda_val:4.0f}: {n_transitions:3d} transitions")

print("\nConclusion: Higher lambda should produce fewer transitions (more stable regimes)")