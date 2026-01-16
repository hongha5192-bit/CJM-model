"""
Analyze BAC performance by regime for simulations with and without URSI
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, recall_score
from itertools import permutations
from jumpmodels.jump import JumpModel
import warnings
warnings.filterwarnings('ignore')

def get_best_permutation(s_pred, s_true, K=3):
    """Find best permutation and return mapped predictions and the permutation"""
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

def calculate_regime_metrics(s_pred_mapped, s_true, K=3):
    """Calculate per-regime recall (which equals per-class BAC)"""
    recalls = []
    for regime in range(K):
        mask = s_true == regime
        if mask.sum() > 0:
            recall = np.mean(s_pred_mapped[mask] == regime)
            recalls.append(recall)
        else:
            recalls.append(0)
    return recalls

def fit_and_evaluate(sim_path, lambda_val, features_to_use, K=3):
    """Fit CJM and return per-regime performance"""
    # Load simulation
    df = pd.read_parquet(sim_path)

    # Extract features
    X_raw = df[features_to_use].values
    ret_ser = df['returns'].values
    s_true = df['s_true'].values

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # Fit CJM
    model = JumpModel(
        n_components=K,
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
    s_pred_mapped, best_perm, overall_bac = get_best_permutation(s_pred, s_true, K)

    # Calculate per-regime metrics
    regime_recalls = calculate_regime_metrics(s_pred_mapped, s_true, K)

    # Get confusion matrix
    cm = confusion_matrix(s_true, s_pred_mapped)

    return {
        'overall_bac': overall_bac,
        'regime_recalls': regime_recalls,
        'confusion_matrix': cm,
        'best_perm': best_perm,
        'true_counts': np.bincount(s_true, minlength=K)
    }

# Configuration
n_simulations = 10
lambda_values = [5, 10, 20, 50, 100]
regime_names = {0: 'Bullish', 1: 'Neutral', 2: 'Bearish'}

# Features
features_without_ursi = ['ADX', 'DMI_Plus', 'DMI_Minus', 'URSI', 'BB_PCTB', 'BBWP']
features_with_ursi = ['ADX', 'DMI_Plus', 'DMI_Minus', 'URSI', 'BB_PCTB', 'BBWP', 'URSI_0_50', 'URSI_0_20']

print("="*80)
print("ANALYZING BAC BY REGIME")
print("="*80)

# Store results
results_without_ursi = {lam: {'recalls': [[] for _ in range(3)], 'overall': []} for lam in lambda_values}
results_with_ursi = {lam: {'recalls': [[] for _ in range(3)], 'overall': []} for lam in lambda_values}

# Process simulations without URSI
print("\n1. Processing simulations WITHOUT URSI breadth indicators...")
sim_dir_without = Path('outputs/step2_sim/K3_price_based')
sim_files_without = sorted(sim_dir_without.glob('sim_*.parquet'))[:n_simulations]

for lambda_val in lambda_values:
    print(f"   Lambda = {lambda_val}...")
    for sim_file in sim_files_without:
        try:
            result = fit_and_evaluate(sim_file, lambda_val, features_without_ursi)
            results_without_ursi[lambda_val]['overall'].append(result['overall_bac'])
            for regime in range(3):
                results_without_ursi[lambda_val]['recalls'][regime].append(result['regime_recalls'][regime])
        except Exception as e:
            print(f"     Error: {e}")

# Process simulations with URSI
print("\n2. Processing simulations WITH URSI breadth indicators...")
sim_dir_with = Path('outputs/step2_sim/K3_with_ursi_breadth')
sim_files_with = sorted(sim_dir_with.glob('sim_*.parquet'))[:n_simulations]

for lambda_val in lambda_values:
    print(f"   Lambda = {lambda_val}...")
    for sim_file in sim_files_with:
        try:
            result = fit_and_evaluate(sim_file, lambda_val, features_with_ursi)
            results_with_ursi[lambda_val]['overall'].append(result['overall_bac'])
            for regime in range(3):
                results_with_ursi[lambda_val]['recalls'][regime].append(result['regime_recalls'][regime])
        except Exception as e:
            print(f"     Error: {e}")

# Calculate statistics
print("\n" + "="*80)
print("PER-REGIME BAC (RECALL) RESULTS")
print("="*80)

# Create detailed comparison
for lambda_val in lambda_values:
    print(f"\n--- Lambda = {lambda_val} ---")

    print("\nWithout URSI:")
    overall_mean = np.mean(results_without_ursi[lambda_val]['overall'])
    print(f"  Overall BAC: {overall_mean:.3f}")
    for regime in range(3):
        recalls = results_without_ursi[lambda_val]['recalls'][regime]
        if recalls:
            mean_recall = np.mean(recalls)
            std_recall = np.std(recalls)
            print(f"  {regime_names[regime]:8s}: {mean_recall:.3f} ± {std_recall:.3f}")

    print("\nWith URSI:")
    overall_mean = np.mean(results_with_ursi[lambda_val]['overall'])
    print(f"  Overall BAC: {overall_mean:.3f}")
    for regime in range(3):
        recalls = results_with_ursi[lambda_val]['recalls'][regime]
        if recalls:
            mean_recall = np.mean(recalls)
            std_recall = np.std(recalls)
            print(f"  {regime_names[regime]:8s}: {mean_recall:.3f} ± {std_recall:.3f}")

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Plot for each lambda (showing best and worst)
lambdas_to_plot = [5, 20, 100]  # Best with URSI, middle, worst

for idx, lambda_val in enumerate(lambdas_to_plot):
    ax = axes[0, idx]

    # Prepare data for grouped bar chart
    regimes = ['Bullish', 'Neutral', 'Bearish']
    x = np.arange(len(regimes))
    width = 0.35

    # Calculate means
    means_without = [np.mean(results_without_ursi[lambda_val]['recalls'][r]) for r in range(3)]
    means_with = [np.mean(results_with_ursi[lambda_val]['recalls'][r]) for r in range(3)]
    stds_without = [np.std(results_without_ursi[lambda_val]['recalls'][r]) for r in range(3)]
    stds_with = [np.std(results_with_ursi[lambda_val]['recalls'][r]) for r in range(3)]

    # Create bars
    bars1 = ax.bar(x - width/2, means_without, width, yerr=stds_without,
                   label='Without URSI', color='#FF6B6B', alpha=0.7, capsize=5)
    bars2 = ax.bar(x + width/2, means_with, width, yerr=stds_with,
                   label='With URSI', color='#4ECDC4', alpha=0.7, capsize=5)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    # Formatting
    ax.set_xlabel('Regime', fontsize=11, fontweight='bold')
    ax.set_ylabel('Recall (Per-regime BAC)', fontsize=11, fontweight='bold')
    ax.set_title(f'Lambda = {lambda_val}', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(regimes)
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.333, linestyle='--', color='gray', alpha=0.3, linewidth=1)
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')

# Bottom row: Heatmaps of recall by lambda
for idx, (results, title) in enumerate([(results_without_ursi, 'Without URSI'),
                                         (results_with_ursi, 'With URSI')]):
    ax = axes[1, idx]

    # Create matrix
    recall_matrix = np.zeros((3, len(lambda_values)))
    for j, lambda_val in enumerate(lambda_values):
        for regime in range(3):
            recalls = results[lambda_val]['recalls'][regime]
            if recalls:
                recall_matrix[regime, j] = np.mean(recalls)

    # Create heatmap
    im = ax.imshow(recall_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Recall', rotation=270, labelpad=15)

    # Set ticks
    ax.set_xticks(np.arange(len(lambda_values)))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels([f'λ={l}' for l in lambda_values])
    ax.set_yticklabels(['Bullish', 'Neutral', 'Bearish'])

    # Add text annotations
    for i in range(3):
        for j in range(len(lambda_values)):
            text = ax.text(j, i, f'{recall_matrix[i, j]:.2f}',
                         ha="center", va="center", color="black", fontsize=10)

    ax.set_title(f'Recall Heatmap - {title}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Lambda', fontsize=11, fontweight='bold')
    ax.set_ylabel('Regime', fontsize=11, fontweight='bold')

# Improvement heatmap
ax = axes[1, 2]
improvement_matrix = np.zeros((3, len(lambda_values)))

for j, lambda_val in enumerate(lambda_values):
    for regime in range(3):
        recalls_without = results_without_ursi[lambda_val]['recalls'][regime]
        recalls_with = results_with_ursi[lambda_val]['recalls'][regime]
        if recalls_without and recalls_with:
            improvement_matrix[regime, j] = np.mean(recalls_with) - np.mean(recalls_without)

# Create heatmap with diverging colormap
im = ax.imshow(improvement_matrix, cmap='RdBu', aspect='auto', vmin=-0.3, vmax=0.3)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Improvement', rotation=270, labelpad=15)

# Set ticks
ax.set_xticks(np.arange(len(lambda_values)))
ax.set_yticks(np.arange(3))
ax.set_xticklabels([f'λ={l}' for l in lambda_values])
ax.set_yticklabels(['Bullish', 'Neutral', 'Bearish'])

# Add text annotations
for i in range(3):
    for j in range(len(lambda_values)):
        value = improvement_matrix[i, j]
        text = ax.text(j, i, f'{value:+.2f}',
                     ha="center", va="center",
                     color="white" if abs(value) > 0.15 else "black",
                     fontsize=10, fontweight='bold')

ax.set_title('Improvement from URSI', fontsize=12, fontweight='bold')
ax.set_xlabel('Lambda', fontsize=11, fontweight='bold')
ax.set_ylabel('Regime', fontsize=11, fontweight='bold')

# Overall title
fig.suptitle('Per-Regime BAC Analysis: Impact of URSI Breadth Indicators',
            fontsize=16, fontweight='bold')

plt.tight_layout()

# Save figure
output_path = 'outputs/bac_by_regime_analysis.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nVisualization saved to: {output_path}")

# Summary table
print("\n" + "="*80)
print("REGIME DETECTION SUMMARY (Best Lambda for Each)")
print("="*80)

# Find best lambda for each configuration
best_lambda_without = max(lambda_values,
                         key=lambda l: np.mean(results_without_ursi[l]['overall']))
best_lambda_with = max(lambda_values,
                      key=lambda l: np.mean(results_with_ursi[l]['overall']))

print(f"\nWithout URSI (Best λ={best_lambda_without}):")
for regime in range(3):
    recalls = results_without_ursi[best_lambda_without]['recalls'][regime]
    if recalls:
        print(f"  {regime_names[regime]:8s}: {np.mean(recalls):.3f} ± {np.std(recalls):.3f}")

print(f"\nWith URSI (Best λ={best_lambda_with}):")
for regime in range(3):
    recalls = results_with_ursi[best_lambda_with]['recalls'][regime]
    if recalls:
        print(f"  {regime_names[regime]:8s}: {np.mean(recalls):.3f} ± {np.std(recalls):.3f}")

print(f"\nImprovement at optimal lambda:")
for regime in range(3):
    without_recall = np.mean(results_without_ursi[best_lambda_without]['recalls'][regime])
    with_recall = np.mean(results_with_ursi[best_lambda_with]['recalls'][regime])
    improvement = with_recall - without_recall
    print(f"  {regime_names[regime]:8s}: {improvement:+.3f} ({improvement/without_recall*100:+.1f}%)")

plt.show()