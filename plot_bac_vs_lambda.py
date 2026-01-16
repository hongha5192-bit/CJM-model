"""
Create BAC vs Lambda chart for 50 simulations with URSI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Data from 50 simulations with URSI
lambdas = [5, 10, 15, 20, 50, 100]
bac_values = [0.5605, 0.5460, 0.5391, 0.5302, 0.5069, 0.4769]
std_values = [0.0625, 0.0646, 0.0607, 0.0619, 0.0598, 0.0512]

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Main line plot with error bars
ax.errorbar(lambdas, bac_values, yerr=std_values,
           marker='o', markersize=12, linewidth=3, capsize=7, capthick=2,
           color='#2E86AB', markerfacecolor='#A23B72', markeredgecolor='#2E86AB',
           markeredgewidth=2, elinewidth=2, alpha=0.9,
           label='BAC with URSI features')

# Highlight best point
best_idx = np.argmax(bac_values)
ax.scatter(lambdas[best_idx], bac_values[best_idx],
          s=300, marker='*', color='gold', edgecolor='darkred', linewidth=2,
          zorder=5, label=f'Best: λ={lambdas[best_idx]}, BAC={bac_values[best_idx]:.3f}')

# Add value annotations for each point
for i, (lam, bac) in enumerate(zip(lambdas, bac_values)):
    ax.annotate(f'{bac:.3f}',
               xy=(lam, bac),
               xytext=(0, 10),
               textcoords='offset points',
               ha='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Reference lines
ax.axhline(y=0.333, linestyle='--', color='red', alpha=0.5, linewidth=1.5,
          label='Random baseline (K=3)')
ax.axhline(y=0.5, linestyle=':', color='gray', alpha=0.5, linewidth=1.5,
          label='BAC = 0.5')

# Formatting
ax.set_xlabel('Lambda (λ)', fontsize=14, fontweight='bold')
ax.set_ylabel('Balanced Accuracy (BAC)', fontsize=14, fontweight='bold')
ax.set_title('BAC vs Lambda - 50 Simulations with URSI Features',
            fontsize=16, fontweight='bold', pad=20)
ax.set_xscale('log')
ax.set_xticks(lambdas)
ax.set_xticklabels([str(l) for l in lambdas])
ax.set_xlim(4, 120)
ax.set_ylim(0.3, 0.65)
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=11)

# Add background shading for different performance regions
ax.axhspan(0.333, 0.45, alpha=0.1, color='red', label='Poor')
ax.axhspan(0.45, 0.55, alpha=0.1, color='yellow', label='Moderate')
ax.axhspan(0.55, 0.65, alpha=0.1, color='green', label='Good')

# Add text describing the trend
ax.text(30, 0.35, 'Performance decreases\nwith higher lambda',
       fontsize=11, ha='center', style='italic',
       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()

# Save figure
output_path = 'outputs/bac_vs_lambda_chart.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Chart saved to: {output_path}")

# Print summary table
print("\n" + "="*50)
print("BAC vs LAMBDA RESULTS (50 Simulations)")
print("="*50)
print(f"{'Lambda':>10} | {'BAC':>10} | {'Std Dev':>10}")
print("-"*50)
for lam, bac, std in zip(lambdas, bac_values, std_values):
    print(f"{lam:>10} | {bac:>10.4f} | ±{std:>9.4f}")
print("-"*50)
print(f"{'Best λ=5':>10} | {max(bac_values):>10.4f} | ±{std_values[np.argmax(bac_values)]:>9.4f}")
print("="*50)

# plt.show()  # Commented to avoid blocking