#!/usr/bin/env python3
"""
Compare simulated features vs real data to identify outliers and significant differences
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_real_data():
    """Load real VNINDEX data with features"""
    data_path = Path('data/processed/vnindex_daily_features_20180102_20250103.parquet')
    df = pd.read_parquet(data_path)

    # Get feature columns (excluding price/return columns and metadata)
    feature_cols = ['ADX', 'URSI', 'BBWP', 'BB_PCTB']

    # Note: Real data doesn't have DMI_Plus/DMI_Minus separately
    # but we can check if they exist
    if 'DMI_Plus' in df.columns:
        feature_cols.append('DMI_Plus')
    if 'DMI_Minus' in df.columns:
        feature_cols.append('DMI_Minus')

    return df[feature_cols]

def load_simulated_data():
    """Load all simulated data"""
    sim_dir = Path('outputs/step2_sim/K3_price_based')

    all_sims = []
    for sim_file in sorted(sim_dir.glob('sim_*.parquet')):
        df = pd.read_parquet(sim_file)
        all_sims.append(df)

    # Concatenate all simulations
    combined = pd.concat(all_sims, ignore_index=True)

    # Get feature columns
    feature_cols = ['ADX', 'DMI_Plus', 'DMI_Minus', 'URSI', 'BBWP', 'BB_PCTB']
    return combined[feature_cols]

def calculate_statistics(df, label):
    """Calculate comprehensive statistics for features"""
    stats_dict = {}

    for col in df.columns:
        stats_dict[col] = {
            'label': label,
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'q01': df[col].quantile(0.01),
            'q05': df[col].quantile(0.05),
            'q25': df[col].quantile(0.25),
            'median': df[col].median(),
            'q75': df[col].quantile(0.75),
            'q95': df[col].quantile(0.95),
            'q99': df[col].quantile(0.99),
            'max': df[col].max(),
            'skewness': df[col].skew(),
            'kurtosis': df[col].kurtosis()
        }

    return pd.DataFrame(stats_dict).T

def perform_ks_test(real_data, sim_data):
    """Perform Kolmogorov-Smirnov test for distribution comparison"""
    results = {}

    common_features = set(real_data.columns) & set(sim_data.columns)

    for feature in common_features:
        # Remove NaN values
        real_vals = real_data[feature].dropna()
        sim_vals = sim_data[feature].dropna()

        # Perform KS test
        ks_stat, p_value = stats.ks_2samp(real_vals, sim_vals)

        results[feature] = {
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'significant_at_0.05': p_value < 0.05,
            'significant_at_0.01': p_value < 0.01
        }

    return pd.DataFrame(results).T

def plot_distributions(real_data, sim_data, output_dir):
    """Create distribution comparison plots"""
    common_features = sorted(set(real_data.columns) & set(sim_data.columns))

    # Create figure with subplots
    n_features = len(common_features)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, feature in enumerate(common_features):
        if idx >= 6:
            break

        ax = axes[idx]

        # Plot histograms
        real_vals = real_data[feature].dropna()
        sim_vals = sim_data[feature].dropna()

        # Create bins that cover both distributions
        min_val = min(real_vals.min(), sim_vals.min())
        max_val = max(real_vals.max(), sim_vals.max())
        bins = np.linspace(min_val, max_val, 30)

        ax.hist(real_vals, bins=bins, alpha=0.5, label='Real', density=True, color='blue')
        ax.hist(sim_vals, bins=bins, alpha=0.5, label='Simulated', density=True, color='red')

        ax.set_title(f'{feature} Distribution')
        ax.set_xlabel(feature)
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_features, 6):
        axes[idx].set_visible(False)

    plt.suptitle('Feature Distributions: Real vs Simulated', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save figure
    output_path = output_dir / 'feature_distributions_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Distribution plots saved to: {output_path}")

def plot_qq_plots(real_data, sim_data, output_dir):
    """Create Q-Q plots for distribution comparison"""
    common_features = sorted(set(real_data.columns) & set(sim_data.columns))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, feature in enumerate(common_features):
        if idx >= 6:
            break

        ax = axes[idx]

        # Get data
        real_vals = real_data[feature].dropna().values
        sim_vals = sim_data[feature].dropna().values

        # Sample to same size for Q-Q plot
        min_size = min(len(real_vals), len(sim_vals))
        if min_size > 5000:  # Subsample for efficiency
            real_sample = np.random.choice(real_vals, 5000, replace=False)
            sim_sample = np.random.choice(sim_vals, 5000, replace=False)
        else:
            real_sample = real_vals[:min_size]
            sim_sample = sim_vals[:min_size]

        # Sort both samples
        real_sorted = np.sort(real_sample)
        sim_sorted = np.sort(sim_sample)

        # Q-Q plot
        ax.scatter(real_sorted, sim_sorted, alpha=0.5, s=1)

        # Add diagonal reference line
        min_val = min(real_sorted.min(), sim_sorted.min())
        max_val = max(real_sorted.max(), sim_sorted.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)

        ax.set_title(f'{feature} Q-Q Plot')
        ax.set_xlabel('Real Data Quantiles')
        ax.set_ylabel('Simulated Data Quantiles')
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(common_features), 6):
        axes[idx].set_visible(False)

    plt.suptitle('Q-Q Plots: Real vs Simulated', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save figure
    output_path = output_dir / 'qq_plots_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Q-Q plots saved to: {output_path}")

def identify_outliers(real_stats, sim_stats):
    """Identify significant differences between real and simulated data"""
    outliers = []

    for feature in real_stats.index:
        if feature not in sim_stats.index:
            continue

        real = real_stats.loc[feature]
        sim = sim_stats.loc[feature]

        # Check mean difference (relative)
        mean_diff = abs(sim['mean'] - real['mean']) / (real['mean'] + 1e-10)

        # Check std difference (relative)
        std_diff = abs(sim['std'] - real['std']) / (real['std'] + 1e-10)

        # Check range difference
        real_range = real['max'] - real['min']
        sim_range = sim['max'] - sim['min']
        range_diff = abs(sim_range - real_range) / (real_range + 1e-10)

        # Check tail behavior
        tail_diff_lower = abs(sim['q01'] - real['q01']) / (abs(real['q01']) + 1e-10)
        tail_diff_upper = abs(sim['q99'] - real['q99']) / (abs(real['q99']) + 1e-10)

        outliers.append({
            'feature': feature,
            'mean_diff_%': mean_diff * 100,
            'std_diff_%': std_diff * 100,
            'range_diff_%': range_diff * 100,
            'lower_tail_diff_%': tail_diff_lower * 100,
            'upper_tail_diff_%': tail_diff_upper * 100,
            'real_mean': real['mean'],
            'sim_mean': sim['mean'],
            'real_std': real['std'],
            'sim_std': sim['std'],
            'real_min': real['min'],
            'sim_min': sim['min'],
            'real_max': real['max'],
            'sim_max': sim['max']
        })

    return pd.DataFrame(outliers)

def main():
    """Main analysis function"""
    print("="*80)
    print("COMPARING SIMULATED vs REAL FEATURE DISTRIBUTIONS")
    print("="*80)

    # Create output directory
    output_dir = Path('outputs/feature_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading real data...")
    real_data = load_real_data()
    print(f"Real data shape: {real_data.shape}")
    print(f"Real features: {list(real_data.columns)}")

    print("\nLoading simulated data...")
    sim_data = load_simulated_data()
    print(f"Simulated data shape: {sim_data.shape}")
    print(f"Simulated features: {list(sim_data.columns)}")

    # Calculate statistics
    print("\nCalculating statistics...")
    real_stats = calculate_statistics(real_data, 'Real')
    sim_stats = calculate_statistics(sim_data, 'Simulated')

    # Combine statistics for comparison
    common_features = sorted(set(real_stats.index) & set(sim_stats.index))

    print("\n" + "="*80)
    print("STATISTICAL COMPARISON")
    print("="*80)

    for feature in common_features:
        print(f"\n{feature}:")
        print(f"  Real:      Mean={real_stats.loc[feature, 'mean']:.3f}, "
              f"Std={real_stats.loc[feature, 'std']:.3f}, "
              f"Range=[{real_stats.loc[feature, 'min']:.3f}, {real_stats.loc[feature, 'max']:.3f}]")
        print(f"  Simulated: Mean={sim_stats.loc[feature, 'mean']:.3f}, "
              f"Std={sim_stats.loc[feature, 'std']:.3f}, "
              f"Range=[{sim_stats.loc[feature, 'min']:.3f}, {sim_stats.loc[feature, 'max']:.3f}]")

    # Perform KS test
    print("\n" + "="*80)
    print("KOLMOGOROV-SMIRNOV TEST RESULTS")
    print("="*80)
    ks_results = perform_ks_test(real_data, sim_data)
    print(ks_results[['ks_statistic', 'p_value', 'significant_at_0.05']])

    # Identify outliers
    print("\n" + "="*80)
    print("OUTLIER ANALYSIS")
    print("="*80)
    outliers_df = identify_outliers(real_stats, sim_stats)

    # Flag significant differences
    outliers_df['is_outlier'] = (
        (outliers_df['mean_diff_%'] > 20) |
        (outliers_df['std_diff_%'] > 30) |
        (outliers_df['range_diff_%'] > 50)
    )

    print("\nFeatures with significant differences:")
    for _, row in outliers_df[outliers_df['is_outlier']].iterrows():
        print(f"\n{row['feature']}:")
        print(f"  Mean difference: {row['mean_diff_%']:.1f}%")
        print(f"  Std difference: {row['std_diff_%']:.1f}%")
        print(f"  Range difference: {row['range_diff_%']:.1f}%")

    # Save detailed statistics
    stats_comparison = pd.concat([
        real_stats.add_suffix('_real'),
        sim_stats.add_suffix('_sim')
    ], axis=1)
    stats_comparison.to_csv(output_dir / 'statistics_comparison.csv')

    # Save KS test results
    ks_results.to_csv(output_dir / 'ks_test_results.csv')

    # Save outlier analysis
    outliers_df.to_csv(output_dir / 'outlier_analysis.csv')

    # Create plots
    print("\nGenerating comparison plots...")
    plot_distributions(real_data, sim_data, output_dir)
    plot_qq_plots(real_data, sim_data, output_dir)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    # Overall assessment
    n_significant = ks_results['significant_at_0.05'].sum()
    n_features = len(ks_results)

    if n_significant == 0:
        print("✅ All features have statistically similar distributions (KS test p > 0.05)")
    elif n_significant < n_features / 2:
        print(f"⚠️  {n_significant}/{n_features} features show significant differences")
    else:
        print(f"❌ {n_significant}/{n_features} features show significant differences")

    # Specific issues
    print("\nKey findings:")
    for feature in common_features:
        real_mean = real_stats.loc[feature, 'mean']
        sim_mean = sim_stats.loc[feature, 'mean']
        diff = abs(sim_mean - real_mean) / (real_mean + 1e-10) * 100

        if diff > 20:
            print(f"  - {feature}: Large mean difference ({diff:.1f}%)")

    print(f"\nDetailed results saved to: {output_dir}")

if __name__ == "__main__":
    main()