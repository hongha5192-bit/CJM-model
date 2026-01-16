"""
Display and visualize the HMM transition matrix
"""
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load HMM parameters
params_path = Path('outputs/step1_params/hmm_K3_features.json')
with open(params_path, 'r') as f:
    params = json.load(f)

# Extract transition matrix
P = np.array(params['P'])

# State labels based on regime interpretation
state_labels = {
    0: "Bullish",
    1: "Neutral",
    2: "Bearish"
}

print("="*80)
print("HMM TRANSITION MATRIX (K=3)")
print("="*80)
print("\nP[i,j] = Probability of transitioning FROM state i TO state j\n")

# Create DataFrame for better display
df_P = pd.DataFrame(
    P,
    index=[f"From {state_labels[i]}" for i in range(3)],
    columns=[f"To {state_labels[i]}" for i in range(3)]
)

print(df_P.to_string(float_format=lambda x: f"{x:.6f}"))

print("\n" + "="*80)
print("KEY STATISTICS")
print("="*80)

# Persistence probabilities (diagonal)
print("\n1. PERSISTENCE PROBABILITIES (staying in same state):")
for i in range(3):
    print(f"   {state_labels[i]:8s}: {P[i,i]:.6f} ({P[i,i]*100:.2f}%)")

# Average duration in each state
print("\n2. AVERAGE DURATION IN EACH STATE:")
for i in range(3):
    avg_duration = 1 / (1 - P[i,i]) if P[i,i] < 1 else float('inf')
    print(f"   {state_labels[i]:8s}: {avg_duration:.1f} days")

# Most likely transitions
print("\n3. SIGNIFICANT TRANSITIONS (>1% probability):")
for i in range(3):
    for j in range(3):
        if i != j and P[i,j] > 0.01:
            print(f"   {state_labels[i]:8s} → {state_labels[j]:8s}: {P[i,j]:.6f} ({P[i,j]*100:.2f}%)")

# Stationary distribution
pi = np.array(params['pi'])
print("\n4. STATIONARY DISTRIBUTION (long-run probabilities):")
for i in range(3):
    print(f"   {state_labels[i]:8s}: {pi[i]:.6f} ({pi[i]*100:.2f}%)")

print("\n" + "="*80)
print("TRANSITION MATRIX HEATMAP")
print("="*80)

# Create heatmap
fig, ax = plt.subplots(figsize=(10, 8))

# Create heatmap with annotations
sns.heatmap(P,
            annot=True,
            fmt='.4f',
            cmap='RdYlGn_r',
            xticklabels=[state_labels[i] for i in range(3)],
            yticklabels=[f"From {state_labels[i]}" for i in range(3)],
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Transition Probability'},
            linewidths=1,
            linecolor='gray',
            square=True)

ax.set_xlabel('To State', fontsize=12, fontweight='bold')
ax.set_ylabel('From State', fontsize=12, fontweight='bold')
ax.set_title('HMM Transition Matrix (K=3 Regimes)', fontsize=14, fontweight='bold')

# Rotate labels for clarity
plt.xticks(rotation=0)
plt.yticks(rotation=0)

# Add text box with key insights
textstr = '\n'.join([
    'Key Insights:',
    f'• Bullish persistence: {P[0,0]*100:.1f}%',
    f'• Neutral persistence: {P[1,1]*100:.1f}%',
    f'• Bearish persistence: {P[2,2]*100:.1f}%',
    '',
    'Average Duration:',
    f'• Bullish: {1/(1-P[0,0]):.1f} days',
    f'• Neutral: {1/(1-P[1,1]):.1f} days',
    f'• Bearish: {1/(1-P[2,2]):.1f} days'
])

props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
ax.text(1.02, 0.5, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='center', bbox=props)

plt.tight_layout()

# Save figure
output_path = 'outputs/transition_matrix_visualization.png'
plt.savefig(output_path, dpi=100, bbox_inches='tight')
print(f"\nVisualization saved to: {output_path}")

# plt.show()  # Commented to avoid blocking

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)

print("""
1. HIGH PERSISTENCE: All three states show very high persistence (>95%)
   - This indicates regimes are stable and don't switch frequently
   - Markets tend to stay in the same regime for extended periods

2. REGIME TRANSITIONS:
   - Bullish → Neutral: 3.42% (most likely bullish transition)
   - Neutral → Bullish: 1.97%
   - Neutral → Bearish: 2.48%
   - Bearish → Neutral: 3.37% (most likely bearish transition)
   - Direct Bullish ↔ Bearish: ~0% (extremely rare)

3. TRANSITION PATTERN:
   - Transitions typically go through Neutral state
   - Direct jumps between Bullish and Bearish are virtually impossible
   - This suggests a "stepping stone" pattern in regime changes

4. MARKET DYNAMICS:
   - Long-run: 24.9% Bullish, 43.2% Neutral, 31.8% Bearish
   - Market spends most time in Neutral state
   - Bearish periods are slightly more common than Bullish
""")

print("="*80)