"""
Analyze BAC for each regime at lambda=5
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from itertools import permutations
from jumpmodels.jump import JumpModel
import warnings
warnings.filterwarnings('ignore')

def get_best_permutation(s_pred, s_true, K=3):
    """Find best permutation and return mapped predictions"""
    best_bac = 0
    best_perm = None
    best_pred_mapped = None

    for perm in permutations(range(K)):
        mapping = {old: new for old, new in enumerate(perm)}
        s_pred_mapped = pd.Series(s_pred).map(mapping).values
        bac = balanced_accuracy_score(s_true, s_pred_mapped)

        if bac > best_bac:
            best_bac = bac
            best_perm = perm
            best_pred_mapped = s_pred_mapped

    return best_pred_mapped, best_perm, best_bac

def analyze_single_simulation(sim_path, features_to_use, lambda_val=5):
    """Analyze a single simulation"""
    # Load simulation
    df = pd.read_parquet(sim_path)

    # Extract features
    X_raw = df[features_to_use].values
    ret_ser = df['returns'].values
    s_true = df['s_true'].values

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # Fit CJM with lambda=5
    model = JumpModel(
        n_components=3,
        jump_penalty=lambda_val,
        cont=True,
        mode_loss=True,
        grid_size=0.05,
        n_init=3,
        max_iter=500,
        tol=1e-5,
        random_state=123,
        verbose=0
    )

    model.fit(X, ret_ser=ret_ser, sort_by='cumret')
    s_pred = model.predict(X)

    # Get best permutation
    s_pred_mapped, best_perm, overall_bac = get_best_permutation(s_pred, s_true)

    # Calculate confusion matrix
    cm = confusion_matrix(s_true, s_pred_mapped)

    # Calculate per-regime recall (which is per-regime BAC)
    recalls = []
    for regime in range(3):
        if cm[regime].sum() > 0:
            recall = cm[regime, regime] / cm[regime].sum()
            recalls.append(recall)
        else:
            recalls.append(0)

    return {
        'overall_bac': overall_bac,
        'recalls': recalls,
        'confusion_matrix': cm,
        'best_perm': best_perm
    }

# Settings
lambda_val = 5
n_simulations = 10
regime_names = ['Bullish', 'Neutral', 'Bearish']

print("="*80)
print(f"REGIME-SPECIFIC BAC ANALYSIS (Lambda = {lambda_val})")
print("="*80)

# Features
features_without_ursi = ['ADX', 'DMI_Plus', 'DMI_Minus', 'URSI', 'BB_PCTB', 'BBWP']
features_with_ursi = ['ADX', 'DMI_Plus', 'DMI_Minus', 'URSI', 'BB_PCTB', 'BBWP', 'URSI_0_50', 'URSI_0_20']

# Process WITHOUT URSI
print("\n1. WITHOUT URSI breadth indicators (6 features):")
print("-"*50)

sim_dir_without = Path('outputs/step2_sim/K3_price_based')
sim_files_without = sorted(sim_dir_without.glob('sim_*.parquet'))[:n_simulations]

results_without = []
all_recalls_without = [[], [], []]  # For each regime

for i, sim_file in enumerate(sim_files_without):
    result = analyze_single_simulation(sim_file, features_without_ursi, lambda_val)
    results_without.append(result)

    for regime in range(3):
        all_recalls_without[regime].append(result['recalls'][regime])

    print(f"Sim {i:2d}: BAC={result['overall_bac']:.3f}, "
          f"Recalls=[{result['recalls'][0]:.3f}, {result['recalls'][1]:.3f}, {result['recalls'][2]:.3f}]")

# Calculate statistics
print(f"\nSummary:")
print(f"  Overall BAC: {np.mean([r['overall_bac'] for r in results_without]):.3f} ± "
      f"{np.std([r['overall_bac'] for r in results_without]):.3f}")
for regime in range(3):
    mean_recall = np.mean(all_recalls_without[regime])
    std_recall = np.std(all_recalls_without[regime])
    print(f"  {regime_names[regime]:8s}: {mean_recall:.3f} ± {std_recall:.3f}")

# Process WITH URSI
print("\n2. WITH URSI breadth indicators (8 features):")
print("-"*50)

sim_dir_with = Path('outputs/step2_sim/K3_with_ursi_breadth')
sim_files_with = sorted(sim_dir_with.glob('sim_*.parquet'))[:n_simulations]

results_with = []
all_recalls_with = [[], [], []]  # For each regime

for i, sim_file in enumerate(sim_files_with):
    result = analyze_single_simulation(sim_file, features_with_ursi, lambda_val)
    results_with.append(result)

    for regime in range(3):
        all_recalls_with[regime].append(result['recalls'][regime])

    print(f"Sim {i:2d}: BAC={result['overall_bac']:.3f}, "
          f"Recalls=[{result['recalls'][0]:.3f}, {result['recalls'][1]:.3f}, {result['recalls'][2]:.3f}]")

# Calculate statistics
print(f"\nSummary:")
print(f"  Overall BAC: {np.mean([r['overall_bac'] for r in results_with]):.3f} ± "
      f"{np.std([r['overall_bac'] for r in results_with]):.3f}")
for regime in range(3):
    mean_recall = np.mean(all_recalls_with[regime])
    std_recall = np.std(all_recalls_with[regime])
    print(f"  {regime_names[regime]:8s}: {mean_recall:.3f} ± {std_recall:.3f}")

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. Bar chart comparing recalls
ax = axes[0]
x = np.arange(len(regime_names))
width = 0.35

means_without = [np.mean(all_recalls_without[r]) for r in range(3)]
means_with = [np.mean(all_recalls_with[r]) for r in range(3)]
stds_without = [np.std(all_recalls_without[r]) for r in range(3)]
stds_with = [np.std(all_recalls_with[r]) for r in range(3)]

bars1 = ax.bar(x - width/2, means_without, width, yerr=stds_without,
               label='Without URSI', color='#FF6B6B', alpha=0.8, capsize=5)
bars2 = ax.bar(x + width/2, means_with, width, yerr=stds_with,
               label='With URSI', color='#4ECDC4', alpha=0.8, capsize=5)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

ax.set_ylabel('Recall (Per-regime BAC)', fontsize=12, fontweight='bold')
ax.set_xlabel('Regime', fontsize=12, fontweight='bold')
ax.set_title(f'Regime-Specific Performance (λ={lambda_val})', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(regime_names)
ax.set_ylim(0, 1.0)
ax.axhline(y=0.333, linestyle='--', color='gray', alpha=0.3, label='Random')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.2, axis='y')

# 2. Average confusion matrix WITHOUT URSI
ax = axes[1]
avg_cm_without = np.mean([r['confusion_matrix'] for r in results_without], axis=0)
# Normalize by row (true labels)
avg_cm_without_norm = avg_cm_without / avg_cm_without.sum(axis=1, keepdims=True)

sns.heatmap(avg_cm_without_norm, annot=True, fmt='.2f', cmap='YlOrRd',
            xticklabels=regime_names, yticklabels=regime_names,
            vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Proportion'})
ax.set_title(f'Avg Confusion Matrix - Without URSI (λ={lambda_val})', fontsize=12, fontweight='bold')
ax.set_xlabel('Predicted', fontsize=11)
ax.set_ylabel('True', fontsize=11)

# 3. Average confusion matrix WITH URSI
ax = axes[2]
avg_cm_with = np.mean([r['confusion_matrix'] for r in results_with], axis=0)
# Normalize by row
avg_cm_with_norm = avg_cm_with / avg_cm_with.sum(axis=1, keepdims=True)

sns.heatmap(avg_cm_with_norm, annot=True, fmt='.2f', cmap='YlGnBu',
            xticklabels=regime_names, yticklabels=regime_names,
            vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Proportion'})
ax.set_title(f'Avg Confusion Matrix - With URSI (λ={lambda_val})', fontsize=12, fontweight='bold')
ax.set_xlabel('Predicted', fontsize=11)
ax.set_ylabel('True', fontsize=11)

plt.suptitle(f'Per-Regime BAC Analysis at Lambda={lambda_val}', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()

# Save
output_path = 'outputs/regime_bac_lambda5.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nVisualization saved to: {output_path}")

# Improvement analysis
print("\n" + "="*80)
print("IMPROVEMENT FROM ADDING URSI BREADTH INDICATORS")
print("="*80)

print(f"\nAt Lambda = {lambda_val}:")
overall_improvement = (np.mean([r['overall_bac'] for r in results_with]) -
                      np.mean([r['overall_bac'] for r in results_without]))
print(f"  Overall BAC improvement: {overall_improvement:+.3f} "
      f"({overall_improvement/np.mean([r['overall_bac'] for r in results_without])*100:+.1f}%)")

print("\nPer-regime improvements:")
for regime in range(3):
    without_mean = np.mean(all_recalls_without[regime])
    with_mean = np.mean(all_recalls_with[regime])
    improvement = with_mean - without_mean
    pct_improvement = improvement / without_mean * 100 if without_mean > 0 else 0
    print(f"  {regime_names[regime]:8s}: {improvement:+.3f} ({pct_improvement:+.1f}%)")

# Show which regime benefits most
improvements = [np.mean(all_recalls_with[r]) - np.mean(all_recalls_without[r]) for r in range(3)]
best_regime = regime_names[np.argmax(improvements)]
worst_regime = regime_names[np.argmin(improvements)]

print(f"\n  Best improvement:  {best_regime} regime")
print(f"  Least improvement: {worst_regime} regime")

# plt.show()  # Commented out to avoid blocking