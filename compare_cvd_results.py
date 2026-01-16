"""
Compare BAC results with and without CVD_20 feature
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("="*80)
print("COMPARISON: BAC WITH vs WITHOUT CVD_20")
print("="*80)

# Load results WITH CVD_20 (9 features)
cvd_results = pd.read_csv('outputs/step3_lambda/lambda_scores_K3_with_cvd20.csv')
print("\nResults WITH CVD_20 (9 features):")
print(cvd_results)

# Load results WITHOUT CVD_20 (8 features with URSI breadth)
ursi_results = pd.read_csv('outputs/step3_lambda/lambda_scores_K3_with_ursi.csv')
print("\nResults WITHOUT CVD_20 (8 features with URSI breadth):")
print(ursi_results)

# Create comparison plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: BAC vs Lambda for both
ax1.errorbar(cvd_results['lambda'], cvd_results['mean_bac'],
             yerr=cvd_results['std_bac'],
             marker='o', linestyle='-', linewidth=2, markersize=8,
             label='With CVD_20 (9 features)', color='blue', capsize=5)

# Find matching lambdas in URSI results
for lambda_val in [5, 10, 20]:
    if lambda_val in ursi_results['lambda'].values:
        ursi_row = ursi_results[ursi_results['lambda'] == lambda_val].iloc[0]
        ax1.errorbar([lambda_val], [ursi_row['mean_bac']],
                    yerr=[ursi_row['std_bac']],
                    marker='s', linestyle='', markersize=8,
                    color='red', capsize=5, alpha=0.7)

# Add red line for URSI results
ax1.plot(ursi_results['lambda'], ursi_results['mean_bac'],
         'r--', linewidth=2, alpha=0.7, label='Without CVD_20 (8 features)')

ax1.set_xlabel('Lambda', fontsize=12)
ax1.set_ylabel('Mean BAC', fontsize=12)
ax1.set_title('BAC Performance: With vs Without CVD_20', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='best')
ax1.set_xscale('log')

# Plot 2: Bar comparison for matching lambdas
matching_lambdas = [5, 10, 20]
cvd_scores = []
ursi_scores = []
improvements = []

for lam in matching_lambdas:
    cvd_row = cvd_results[cvd_results['lambda'] == lam]
    ursi_row = ursi_results[ursi_results['lambda'] == lam]

    if len(cvd_row) > 0 and len(ursi_row) > 0:
        cvd_bac = cvd_row['mean_bac'].values[0]
        ursi_bac = ursi_row['mean_bac'].values[0]
        cvd_scores.append(cvd_bac)
        ursi_scores.append(ursi_bac)
        improvements.append((cvd_bac - ursi_bac) / ursi_bac * 100)

x = np.arange(len(matching_lambdas))
width = 0.35

bars1 = ax2.bar(x - width/2, ursi_scores, width, label='Without CVD_20', color='coral')
bars2 = ax2.bar(x + width/2, cvd_scores, width, label='With CVD_20', color='skyblue')

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}', ha='center', va='bottom', fontsize=10)

for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}', ha='center', va='bottom', fontsize=10)

# Add improvement percentages
for i, (lam, imp) in enumerate(zip(matching_lambdas, improvements)):
    color = 'green' if imp > 0 else 'red'
    ax2.text(i, max(cvd_scores[i], ursi_scores[i]) + 0.01,
             f'{imp:+.1f}%', ha='center', fontsize=10, color=color, fontweight='bold')

ax2.set_xlabel('Lambda', fontsize=12)
ax2.set_ylabel('Mean BAC', fontsize=12)
ax2.set_title('Direct Comparison at Specific Lambda Values', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels([f'λ={l}' for l in matching_lambdas])
ax2.legend()
ax2.grid(True, axis='y', alpha=0.3)
ax2.set_ylim([0.45, 0.6])

plt.tight_layout()
plt.savefig('outputs/cvd_comparison_results.png', dpi=100, bbox_inches='tight')
print("\nPlot saved to: outputs/cvd_comparison_results.png")

# Print detailed comparison
print("\n" + "="*80)
print("DETAILED COMPARISON AT MATCHING LAMBDA VALUES")
print("="*80)

for i, lam in enumerate(matching_lambdas):
    cvd_row = cvd_results[cvd_results['lambda'] == lam]
    ursi_row = ursi_results[ursi_results['lambda'] == lam]

    if len(cvd_row) > 0 and len(ursi_row) > 0:
        cvd_bac = cvd_row['mean_bac'].values[0]
        cvd_std = cvd_row['std_bac'].values[0]
        ursi_bac = ursi_row['mean_bac'].values[0]
        ursi_std = ursi_row['std_bac'].values[0]
        improvement = (cvd_bac - ursi_bac) / ursi_bac * 100

        print(f"\nLambda = {lam}:")
        print(f"  Without CVD_20: BAC = {ursi_bac:.4f} ± {ursi_std:.4f}")
        print(f"  With CVD_20:    BAC = {cvd_bac:.4f} ± {cvd_std:.4f}")
        print(f"  Change:         {improvement:+.2f}%")

# Summary statistics
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

best_cvd_idx = cvd_results['mean_bac'].idxmax()
best_cvd_lambda = cvd_results.loc[best_cvd_idx, 'lambda']
best_cvd_bac = cvd_results.loc[best_cvd_idx, 'mean_bac']

best_ursi_idx = ursi_results['mean_bac'].idxmax()
best_ursi_lambda = ursi_results.loc[best_ursi_idx, 'lambda']
best_ursi_bac = ursi_results.loc[best_ursi_idx, 'mean_bac']

print(f"\nBest performance WITHOUT CVD_20:")
print(f"  Lambda = {best_ursi_lambda}, BAC = {best_ursi_bac:.4f}")

print(f"\nBest performance WITH CVD_20:")
print(f"  Lambda = {best_cvd_lambda}, BAC = {best_cvd_bac:.4f}")

print(f"\nOverall improvement: {(best_cvd_bac - best_ursi_bac) / best_ursi_bac * 100:+.2f}%")

# Key findings
print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

avg_improvement = np.mean(improvements)
if avg_improvement > 0:
    print(f"✗ Adding CVD_20 DECREASED performance by an average of {abs(avg_improvement):.1f}%")
else:
    print(f"✓ Adding CVD_20 did not improve performance (average change: {avg_improvement:.1f}%)")

print("\nPossible reasons:")
print("- CVD_20 may be redundant with existing volume-based features")
print("- The 20-bar window might not be optimal for this dataset")
print("- CVD_20 might need different preprocessing or normalization")
print("- The simulated CVD_20 might not capture real market microstructure")

print("\n" + "="*80)