"""
Compare lambda scan results between 10 and 50 simulations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data from results
lambdas = [5, 10, 15, 20, 50, 100]

# 10 simulations (previous run)
bac_10sims = [0.5833, 0.5616, 0.5582, 0.5517, 0.5163, 0.4637]
std_10sims = [0.0580, 0.0621, 0.0530, 0.0609, 0.0654, 0.0558]

# 50 simulations (current run)
bac_50sims = [0.5605, 0.5460, 0.5391, 0.5302, 0.5069, 0.4769]
std_50sims = [0.0625, 0.0646, 0.0607, 0.0619, 0.0598, 0.0512]

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: BAC comparison
ax1.errorbar(lambdas, bac_10sims, yerr=std_10sims,
            marker='o', markersize=8, label='10 simulations',
            linewidth=2, capsize=5, alpha=0.7, color='blue')
ax1.errorbar(lambdas, bac_50sims, yerr=std_50sims,
            marker='s', markersize=8, label='50 simulations',
            linewidth=2, capsize=5, alpha=0.7, color='red')

ax1.axhline(y=0.333, linestyle='--', color='gray', alpha=0.3, label='Random')
ax1.axhline(y=0.5, linestyle=':', color='gray', alpha=0.3)

ax1.set_xlabel('Lambda', fontsize=11, fontweight='bold')
ax1.set_ylabel('Balanced Accuracy', fontsize=11, fontweight='bold')
ax1.set_title('BAC Comparison: 10 vs 50 Simulations', fontsize=12, fontweight='bold')
ax1.set_xscale('log')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_xlim(4, 120)
ax1.set_ylim(0.4, 0.65)

# Plot 2: Difference analysis
ax2.bar(range(len(lambdas)), np.array(bac_50sims) - np.array(bac_10sims),
       color=['green' if d > 0 else 'red' for d in np.array(bac_50sims) - np.array(bac_10sims)],
       alpha=0.6, edgecolor='black', linewidth=1.5)

ax2.set_xticks(range(len(lambdas)))
ax2.set_xticklabels([f'λ={l}' for l in lambdas])
ax2.set_xlabel('Lambda', fontsize=11, fontweight='bold')
ax2.set_ylabel('BAC Difference (50 sims - 10 sims)', fontsize=11, fontweight='bold')
ax2.set_title('Effect of More Simulations', fontsize=12, fontweight='bold')
ax2.axhline(y=0, color='black', linewidth=1)
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, diff in enumerate(np.array(bac_50sims) - np.array(bac_10sims)):
    ax2.text(i, diff + 0.002 if diff > 0 else diff - 0.002, f'{diff:.3f}',
            ha='center', va='bottom' if diff > 0 else 'top', fontweight='bold')

plt.suptitle('Lambda Scan: Effect of Simulation Sample Size', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/lambda_scan_10vs50_comparison.png', dpi=120, bbox_inches='tight')

# Print summary statistics
print("="*70)
print("LAMBDA SCAN COMPARISON: 10 vs 50 SIMULATIONS")
print("="*70)

print("\n50 Simulations Results (With URSI):")
print("-"*40)
for i, lam in enumerate(lambdas):
    print(f"λ={lam:3d}: BAC = {bac_50sims[i]:.4f} ± {std_50sims[i]:.4f}")

print(f"\nBest Lambda: {lambdas[np.argmax(bac_50sims)]}")
print(f"Best BAC: {max(bac_50sims):.4f}")
print(f"BAC Range: {max(bac_50sims) - min(bac_50sims):.4f}")

print("\n10 vs 50 Simulations Comparison:")
print("-"*40)
print(f"Mean absolute difference: {np.mean(np.abs(np.array(bac_50sims) - np.array(bac_10sims))):.4f}")
print(f"Max difference: {max(np.abs(np.array(bac_50sims) - np.array(bac_10sims))):.4f}")

# Check if ranking changed
rank_10 = np.argsort(bac_10sims)[::-1]
rank_50 = np.argsort(bac_50sims)[::-1]
print(f"\nLambda ranking (best to worst):")
print(f"  10 sims: {[lambdas[i] for i in rank_10]}")
print(f"  50 sims: {[lambdas[i] for i in rank_50]}")
print(f"  Ranking changed: {'No' if np.array_equal(rank_10, rank_50) else 'Yes'}")

print("\nConclusion:")
print("-"*40)
print("• Optimal lambda remains 5 with both 10 and 50 simulations")
print("• 50 simulations provide more conservative estimates (slightly lower BAC)")
print("• Standard deviations are similar, confirming stability")
print("• The clear downward trend with increasing lambda is consistent")
print("="*70)