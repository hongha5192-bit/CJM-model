"""
Analyze URSI_0_50 and URSI_0_20 statistics by HMM regime
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Load the HMM results
print("Loading HMM results and data...")
hmm_results_path = Path('outputs/step1_params/hmm_states_K3.csv')
if not hmm_results_path.exists():
    raise FileNotFoundError(f"Cannot find HMM results file at {hmm_results_path}")

hmm_df = pd.read_csv(hmm_results_path)
print(f"Loaded HMM results: {hmm_df.shape}")
print(f"Columns: {hmm_df.columns.tolist()}")

# Load the features data with URSI_0_50 and URSI_0_20
features_path = Path('data/vnindex_features.parquet')
if not features_path.exists():
    print("Features file not found, looking for alternative...")
    # Try to load from the original data and compute features
    data_path = Path('data/vnindex_full.csv')
    if data_path.exists():
        raw_df = pd.read_csv(data_path)
        print(f"Loaded raw data: {raw_df.shape}")

        # Check if URSI columns exist
        if 'URSI_0_50' in raw_df.columns and 'URSI_0_20' in raw_df.columns:
            features_df = raw_df[['TRADINGDATE', 'URSI_0_50', 'URSI_0_20']].copy()
            features_df['TRADINGDATE'] = pd.to_datetime(features_df['TRADINGDATE'])
        else:
            print("URSI columns not found in raw data")
            print(f"Available columns: {raw_df.columns.tolist()}")
            raise ValueError("Cannot find URSI_0_50 and URSI_0_20 columns")
    else:
        raise FileNotFoundError("Cannot find features or raw data file")
else:
    features_df = pd.read_parquet(features_path)
    print(f"Loaded features: {features_df.shape}")

# Ensure date columns are datetime
if 'date' in hmm_df.columns:
    hmm_df['date'] = pd.to_datetime(hmm_df['date'])
elif 'TRADINGDATE' in hmm_df.columns:
    hmm_df['date'] = pd.to_datetime(hmm_df['TRADINGDATE'])

if 'TRADINGDATE' in features_df.columns:
    features_df['date'] = pd.to_datetime(features_df['TRADINGDATE'])
elif 'date' in features_df.columns:
    features_df['date'] = pd.to_datetime(features_df['date'])

# Merge the dataframes
print("\nMerging HMM labels with features...")
merged_df = pd.merge(hmm_df, features_df, on='date', how='inner')
print(f"Merged data shape: {merged_df.shape}")

# Check if we have the regime labels
if 'hmm_state' in merged_df.columns:
    regime_col = 'hmm_state'
elif 'regime' in merged_df.columns:
    regime_col = 'regime'
elif 'label' in merged_df.columns:
    regime_col = 'label'
elif 'state' in merged_df.columns:
    regime_col = 'state'
else:
    print(f"Available columns: {merged_df.columns.tolist()}")
    raise ValueError("Cannot find regime/label/state column")

print(f"Using regime column: {regime_col}")

# Calculate statistics for URSI_0_50 and URSI_0_20 by regime
print("\n" + "="*80)
print("URSI_0_50 and URSI_0_20 Statistics by HMM Regime")
print("="*80)

regime_stats = {}
regime_names = {0: 'Bullish', 1: 'Neutral', 2: 'Bearish'}

for feature in ['URSI_0_50', 'URSI_0_20']:
    if feature not in merged_df.columns:
        print(f"\nWarning: {feature} not found in data")
        continue

    print(f"\n{feature}:")
    print("-"*40)

    stats_dict = {}
    for regime in sorted(merged_df[regime_col].unique()):
        regime_data = merged_df[merged_df[regime_col] == regime][feature].dropna()

        if len(regime_data) > 0:
            mean = regime_data.mean()
            std = regime_data.std()
            median = regime_data.median()
            q25 = regime_data.quantile(0.25)
            q75 = regime_data.quantile(0.75)

            stats_dict[regime] = {
                'mean': mean,
                'std': std,
                'median': median,
                'q25': q25,
                'q75': q75,
                'count': len(regime_data)
            }

            regime_name = regime_names.get(regime, f'Regime {regime}')
            print(f"  {regime_name} (State {regime}):")
            print(f"    Mean:   {mean:.2f}")
            print(f"    Std:    {std:.2f}")
            print(f"    Median: {median:.2f}")
            print(f"    Q25-Q75: [{q25:.2f}, {q75:.2f}]")
            print(f"    Count:  {len(regime_data)}")

    regime_stats[feature] = stats_dict

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for i, feature in enumerate(['URSI_0_50', 'URSI_0_20']):
    if feature not in merged_df.columns:
        continue

    # Box plot
    ax = axes[i, 0]
    data_for_plot = []
    labels_for_plot = []

    for regime in sorted(merged_df[regime_col].unique()):
        regime_data = merged_df[merged_df[regime_col] == regime][feature].dropna()
        if len(regime_data) > 0:
            data_for_plot.append(regime_data.values)
            labels_for_plot.append(f"{regime_names.get(regime, f'R{regime}')}\n(n={len(regime_data)})")

    bp = ax.boxplot(data_for_plot, labels=labels_for_plot)
    ax.set_title(f'{feature} Distribution by Regime')
    ax.set_ylabel(feature)
    ax.grid(True, alpha=0.3)

    # Add mean line
    means = [np.mean(d) for d in data_for_plot]
    ax.plot(range(1, len(means)+1), means, 'ro-', label='Mean')
    ax.legend()

    # Histogram
    ax = axes[i, 1]
    for regime in sorted(merged_df[regime_col].unique()):
        regime_data = merged_df[merged_df[regime_col] == regime][feature].dropna()
        if len(regime_data) > 0:
            ax.hist(regime_data, bins=30, alpha=0.5,
                   label=f"{regime_names.get(regime, f'Regime {regime}')} (μ={regime_data.mean():.1f})",
                   density=True)

    ax.set_xlabel(feature)
    ax.set_ylabel('Density')
    ax.set_title(f'{feature} Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Time series colored by regime
    ax = axes[i, 2]
    colors = {0: 'green', 1: 'yellow', 2: 'red'}

    for regime in sorted(merged_df[regime_col].unique()):
        regime_mask = merged_df[regime_col] == regime
        regime_data = merged_df[regime_mask]
        if len(regime_data) > 0:
            ax.scatter(regime_data['date'], regime_data[feature],
                      c=colors.get(regime, 'gray'), alpha=0.5, s=1,
                      label=regime_names.get(regime, f'Regime {regime}'))

    ax.set_xlabel('Date')
    ax.set_ylabel(feature)
    ax.set_title(f'{feature} Time Series by Regime')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add horizontal lines for means
    for regime in sorted(merged_df[regime_col].unique()):
        regime_data = merged_df[merged_df[regime_col] == regime][feature].dropna()
        if len(regime_data) > 0:
            ax.axhline(y=regime_data.mean(), color=colors.get(regime, 'gray'),
                      linestyle='--', alpha=0.5, linewidth=1)

plt.suptitle('URSI_0_50 and URSI_0_20 Analysis by HMM Regime', fontsize=14)
plt.tight_layout()

# Save the plot
output_path = 'outputs/ursi_by_hmm_regime.png'
plt.savefig(output_path, dpi=100, bbox_inches='tight')
print(f"\nVisualization saved to: {output_path}")

# Create a summary table
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)

summary_data = []
for feature in ['URSI_0_50', 'URSI_0_20']:
    if feature not in regime_stats:
        continue
    for regime in sorted(regime_stats[feature].keys()):
        stats = regime_stats[feature][regime]
        summary_data.append({
            'Feature': feature,
            'Regime': regime_names.get(regime, f'Regime {regime}'),
            'Mean': f"{stats['mean']:.2f}",
            'Std': f"{stats['std']:.2f}",
            'Median': f"{stats['median']:.2f}",
            'Q25': f"{stats['q25']:.2f}",
            'Q75': f"{stats['q75']:.2f}",
            'Count': stats['count']
        })

if summary_data:
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    # Save to CSV
    summary_df.to_csv('outputs/ursi_regime_statistics.csv', index=False)
    print("\nStatistics saved to: outputs/ursi_regime_statistics.csv")

# Statistical tests
print("\n" + "="*80)
print("STATISTICAL TESTS (ANOVA)")
print("="*80)

from scipy import stats as scipy_stats

for feature in ['URSI_0_50', 'URSI_0_20']:
    if feature not in merged_df.columns:
        continue

    print(f"\n{feature}:")

    # Prepare groups for ANOVA
    groups = []
    for regime in sorted(merged_df[regime_col].unique()):
        regime_data = merged_df[merged_df[regime_col] == regime][feature].dropna()
        if len(regime_data) > 0:
            groups.append(regime_data.values)

    if len(groups) >= 2:
        # Perform ANOVA
        f_stat, p_value = scipy_stats.f_oneway(*groups)
        print(f"  F-statistic: {f_stat:.4f}")
        print(f"  p-value: {p_value:.4e}")

        if p_value < 0.05:
            print(f"  Result: Significant difference between regimes (p < 0.05)")
        else:
            print(f"  Result: No significant difference between regimes (p >= 0.05)")

print("\n" + "="*80)