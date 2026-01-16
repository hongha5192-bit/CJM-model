"""
Visualize final CJM results
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load results
df = pd.read_csv('outputs/step4_apply/regimes_daily.csv')
df['date'] = pd.to_datetime(df['date'])

# Create figure with subplots
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Plot 1: Price with regime coloring
ax1 = axes[0]
for regime in [0, 1]:
    mask = df['label'] == regime
    color = 'red' if regime == 0 else 'green'
    label = f'Regime {regime} ({"Bear" if regime == 0 else "Bull"})'
    ax1.scatter(df[mask]['date'], df[mask]['close'],
                c=color, alpha=0.3, s=2, label=label)
ax1.plot(df['date'], df['close'], 'k-', alpha=0.5, linewidth=0.5)
ax1.set_ylabel('VNINDEX Close Price')
ax1.set_title('VNINDEX with CJM Regime Classification (2018-2025)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Daily returns by regime
ax2 = axes[1]
for regime in [0, 1]:
    mask = df['label'] == regime
    color = 'red' if regime == 0 else 'green'
    ax2.scatter(df[mask]['date'], df[mask]['ret'] * 100,
                c=color, alpha=0.4, s=2)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.set_ylabel('Daily Return (%)')
ax2.set_title('Daily Returns by Regime')
ax2.grid(True, alpha=0.3)

# Plot 3: Regime probability
ax3 = axes[2]
ax3.fill_between(df['date'], 0, df['proba_0'],
                 color='red', alpha=0.3, label='P(Bear)')
ax3.fill_between(df['date'], df['proba_0'], 1,
                 color='green', alpha=0.3, label='P(Bull)')
ax3.axhline(y=0.5, color='black', linestyle='--', linewidth=0.5)
ax3.set_ylabel('Regime Probability')
ax3.set_xlabel('Date')
ax3.set_title('Regime Probabilities Over Time')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/regime_visualization.png', dpi=150)
plt.close()

# Print summary statistics
print("=" * 60)
print("CJM REGIME ANALYSIS SUMMARY")
print("=" * 60)

for regime in [0, 1]:
    mask = df['label'] == regime
    n_days = mask.sum()
    pct = 100 * n_days / len(df)
    mean_ret = df[mask]['ret'].mean() * 252  # Annualized
    vol = df[mask]['ret'].std() * np.sqrt(252)  # Annualized
    sharpe = mean_ret / vol if vol > 0 else 0

    print(f"\nRegime {regime} ({'Bear/Volatile' if regime == 0 else 'Bull/Calm'}):")
    print(f"  Days: {n_days} ({pct:.1f}%)")
    print(f"  Annualized Return: {mean_ret:.1f}%")
    print(f"  Annualized Volatility: {vol:.1f}%")
    print(f"  Sharpe Ratio: {sharpe:.2f}")

# Recent regime analysis
recent = df.tail(20)
recent_regime = recent['label'].mode()[0]
print(f"\nRecent Market State (Last 20 days):")
print(f"  Dominant Regime: {recent_regime} ({'Bear' if recent_regime == 0 else 'Bull'})")
print(f"  Average Confidence: {recent['proba_0'].mean():.1%} for Bear")

print("\nVisualization saved to: outputs/regime_visualization.png")
print("=" * 60)