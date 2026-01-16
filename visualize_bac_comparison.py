"""
Visualize BAC comparison across lambda values with and without URSI breadth indicators
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load results
results_without_ursi = pd.read_csv('outputs/step3_lambda/lambda_scores_K3_price_based.csv')
results_with_ursi = pd.read_csv('outputs/step3_lambda/lambda_scores_K3_with_ursi.csv')

# Create figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# ========== Left plot: BAC comparison ==========
ax1 = axes[0]

# Plot without URSI
ax1.errorbar(results_without_ursi['lambda'],
             results_without_ursi['mean_bac'],
             yerr=results_without_ursi['std_bac'],
             marker='o', markersize=8,
             label='Without URSI_0_50/0_20 (6 features)',
             linewidth=2, capsize=5, capthick=2,
             color='#FF6B6B', markerfacecolor='#FF6B6B',
             markeredgecolor='darkred', markeredgewidth=1.5)

# Plot with URSI
ax1.errorbar(results_with_ursi['lambda'],
             results_with_ursi['mean_bac'],
             yerr=results_with_ursi['std_bac'],
             marker='s', markersize=8,
             label='With URSI_0_50/0_20 (8 features)',
             linewidth=2, capsize=5, capthick=2,
             color='#4ECDC4', markerfacecolor='#4ECDC4',
             markeredgecolor='darkgreen', markeredgewidth=1.5)

# Mark best points
best_without_idx = results_without_ursi['mean_bac'].idxmax()
best_with_idx = results_with_ursi['mean_bac'].idxmax()

ax1.scatter(results_without_ursi.loc[best_without_idx, 'lambda'],
           results_without_ursi.loc[best_without_idx, 'mean_bac'],
           s=200, marker='*', color='darkred', zorder=5,
           label=f"Best without URSI (λ={results_without_ursi.loc[best_without_idx, 'lambda']:.0f})")

ax1.scatter(results_with_ursi.loc[best_with_idx, 'lambda'],
           results_with_ursi.loc[best_with_idx, 'mean_bac'],
           s=200, marker='*', color='darkgreen', zorder=5,
           label=f"Best with URSI (λ={results_with_ursi.loc[best_with_idx, 'lambda']:.0f})")

# Add reference lines
ax1.axhline(y=0.333, linestyle='--', color='gray', alpha=0.5, linewidth=1,
           label='Random baseline (K=3)')
ax1.axhline(y=0.5, linestyle=':', color='gray', alpha=0.5, linewidth=1,
           label='BAC = 0.5')

# Formatting
ax1.set_xlabel('Lambda (Jump Penalty)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Balanced Accuracy (BAC)', fontsize=12, fontweight='bold')
ax1.set_title('BAC vs Lambda: Impact of URSI Breadth Indicators', fontsize=14, fontweight='bold')
ax1.set_xscale('log')
ax1.set_xlim(4, 120)
ax1.set_ylim(0.3, 0.7)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='best', frameon=True, fancybox=True, shadow=True)

# Add improvement annotation
improvement = (results_with_ursi.loc[best_with_idx, 'mean_bac'] -
               results_without_ursi.loc[best_without_idx, 'mean_bac']) / \
              results_without_ursi.loc[best_without_idx, 'mean_bac'] * 100

ax1.annotate(f'+{improvement:.1f}% improvement',
            xy=(7, 0.58), fontsize=11, fontweight='bold',
            color='green', bbox=dict(boxstyle="round,pad=0.3",
                                    facecolor='lightgreen', alpha=0.5))

# ========== Right plot: Performance difference ==========
ax2 = axes[1]

# Calculate differences
lambdas = results_with_ursi['lambda'].values
bac_diff = results_with_ursi['mean_bac'].values - results_without_ursi['mean_bac'].values

# Create bar plot
bars = ax2.bar(range(len(lambdas)), bac_diff * 100,
               color=['green' if d > 0 else 'red' for d in bac_diff],
               alpha=0.7, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for i, (bar, diff) in enumerate(zip(bars, bac_diff * 100)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height > 0 else -0.5),
            f'{diff:.1f}%', ha='center', va='bottom' if height > 0 else 'top',
            fontweight='bold', fontsize=10)

# Formatting
ax2.set_xticks(range(len(lambdas)))
ax2.set_xticklabels([f'λ={int(l)}' for l in lambdas])
ax2.set_xlabel('Lambda Values', fontsize=12, fontweight='bold')
ax2.set_ylabel('BAC Improvement (%)', fontsize=12, fontweight='bold')
ax2.set_title('Performance Gain from Adding URSI_0_50/0_20', fontsize=14, fontweight='bold')
ax2.axhline(y=0, color='black', linewidth=1, linestyle='-')
ax2.grid(True, alpha=0.3, axis='y')

# Add average improvement line
avg_improvement = np.mean(bac_diff * 100)
ax2.axhline(y=avg_improvement, color='blue', linestyle='--', linewidth=2, alpha=0.7,
           label=f'Average: {avg_improvement:.1f}%')
ax2.legend()

# Overall title
fig.suptitle('Lambda Scan Results: Effect of URSI Market Breadth Indicators',
            fontsize=16, fontweight='bold', y=1.02)

# Adjust layout
plt.tight_layout()

# Save figure
output_path = 'outputs/bac_lambda_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Visualization saved to: {output_path}")

# Also create a detailed table
print("\n" + "="*80)
print("DETAILED COMPARISON TABLE")
print("="*80)

comparison_df = pd.DataFrame({
    'Lambda': lambdas,
    'BAC without URSI': results_without_ursi['mean_bac'].values,
    'Std without URSI': results_without_ursi['std_bac'].values,
    'BAC with URSI': results_with_ursi['mean_bac'].values,
    'Std with URSI': results_with_ursi['std_bac'].values,
    'Improvement': bac_diff,
    'Improvement (%)': bac_diff * 100
})

# Format for display
for col in ['BAC without URSI', 'Std without URSI', 'BAC with URSI', 'Std with URSI', 'Improvement']:
    comparison_df[col] = comparison_df[col].apply(lambda x: f"{x:.4f}")
comparison_df['Improvement (%)'] = comparison_df['Improvement (%)'].apply(lambda x: f"{x:+.1f}%")

print(comparison_df.to_string(index=False))

# Summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print(f"Without URSI breadth indicators (6 features):")
print(f"  Best Lambda: {results_without_ursi.loc[best_without_idx, 'lambda']:.0f}")
print(f"  Best BAC: {results_without_ursi.loc[best_without_idx, 'mean_bac']:.4f} ± "
      f"{results_without_ursi.loc[best_without_idx, 'std_bac']:.4f}")
print(f"  BAC Range: {results_without_ursi['mean_bac'].min():.4f} - "
      f"{results_without_ursi['mean_bac'].max():.4f} "
      f"(diff: {results_without_ursi['mean_bac'].max() - results_without_ursi['mean_bac'].min():.4f})")

print(f"\nWith URSI breadth indicators (8 features):")
print(f"  Best Lambda: {results_with_ursi.loc[best_with_idx, 'lambda']:.0f}")
print(f"  Best BAC: {results_with_ursi.loc[best_with_idx, 'mean_bac']:.4f} ± "
      f"{results_with_ursi.loc[best_with_idx, 'std_bac']:.4f}")
print(f"  BAC Range: {results_with_ursi['mean_bac'].min():.4f} - "
      f"{results_with_ursi['mean_bac'].max():.4f} "
      f"(diff: {results_with_ursi['mean_bac'].max() - results_with_ursi['mean_bac'].min():.4f})")

print(f"\nImprovement:")
print(f"  Best BAC improvement: {improvement:.1f}%")
print(f"  Average improvement across all lambdas: {avg_improvement:.1f}%")
print(f"  Lambda sensitivity increased by: "
      f"{(results_with_ursi['mean_bac'].max() - results_with_ursi['mean_bac'].min()) / (results_without_ursi['mean_bac'].max() - results_without_ursi['mean_bac'].min()):.1f}x")

plt.show()