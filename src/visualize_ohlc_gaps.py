#!/usr/bin/env python3
"""
Visualize OHLC gap analysis by regime
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def visualize_ohlc_gaps():
    """Create visualizations for OHLC gap analysis"""

    print("="*80)
    print("VISUALIZING OHLC GAP ANALYSIS")
    print("="*80)

    # Load the data
    print("\nLoading data...")
    df = pd.read_csv('outputs/ohlc_analysis/daily_ohlc_with_gaps.csv')
    df['date'] = pd.to_datetime(df['date'])

    print(f"Data shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # State labels
    state_labels = {
        0: "Bullish",
        1: "Neutral",
        2: "Bearish"
    }

    df['regime_label'] = df['regime'].map(state_labels)

    # Create output directory
    output_dir = Path('outputs/ohlc_analysis/visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    colors = {'Bullish': '#2ecc71', 'Neutral': '#f39c12', 'Bearish': '#e74c3c'}

    # ========================================================================
    # 1. Distribution plots - Box plots
    # ========================================================================
    print("\nCreating distribution plots...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # High-Close gap
    df_plot = df.copy()
    sns.boxplot(data=df_plot, x='regime_label', y='high_close_gap_pct',
                ax=axes[0], palette=colors, order=['Bullish', 'Neutral', 'Bearish'])
    axes[0].set_title('High - Close Gap Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Regime')
    axes[0].set_ylabel('Gap (%)')
    axes[0].set_ylim(0, 3)

    # Low-Close gap
    sns.boxplot(data=df_plot, x='regime_label', y='low_close_gap_pct',
                ax=axes[1], palette=colors, order=['Bullish', 'Neutral', 'Bearish'])
    axes[1].set_title('Close - Low Gap Distribution', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Regime')
    axes[1].set_ylabel('Gap (%)')
    axes[1].set_ylim(0, 3)

    # Total daily range
    df_plot['daily_range_pct'] = df_plot['high_close_gap_pct'] + df_plot['low_close_gap_pct']
    sns.boxplot(data=df_plot, x='regime_label', y='daily_range_pct',
                ax=axes[2], palette=colors, order=['Bullish', 'Neutral', 'Bearish'])
    axes[2].set_title('Total Daily Range Distribution', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Regime')
    axes[2].set_ylabel('Range (%)')
    axes[2].set_ylim(0, 4)

    plt.tight_layout()
    plt.savefig(output_dir / 'ohlc_gaps_boxplots.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # 2. Mean and Std comparison bar charts
    # ========================================================================
    print("Creating comparison bar charts...")

    # Calculate statistics
    stats = df.groupby('regime_label').agg({
        'high_close_gap_pct': ['mean', 'std'],
        'low_close_gap_pct': ['mean', 'std']
    }).reset_index()

    stats.columns = ['regime', 'hc_mean', 'hc_std', 'lc_mean', 'lc_std']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Mean comparison
    x = np.arange(3)
    width = 0.35

    ax = axes[0]
    bars1 = ax.bar(x - width/2, stats['hc_mean'], width, label='High-Close Gap',
                   color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, stats['lc_mean'], width, label='Close-Low Gap',
                   color='#e67e22', alpha=0.8)

    ax.set_ylabel('Mean Gap (%)', fontsize=11)
    ax.set_title('Mean OHLC Gaps by Regime', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Bullish', 'Neutral', 'Bearish'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}%', ha='center', va='bottom', fontsize=9)

    # Std comparison
    ax = axes[1]
    bars1 = ax.bar(x - width/2, stats['hc_std'], width, label='High-Close Gap',
                   color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, stats['lc_std'], width, label='Close-Low Gap',
                   color='#e67e22', alpha=0.8)

    ax.set_ylabel('Std Dev (%)', fontsize=11)
    ax.set_title('OHLC Gap Volatility by Regime', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Bullish', 'Neutral', 'Bearish'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'ohlc_gaps_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # 3. Histogram distributions
    # ========================================================================
    print("Creating histogram distributions...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    regimes = ['Bullish', 'Neutral', 'Bearish']

    for idx, regime in enumerate(regimes):
        regime_data = df[df['regime_label'] == regime]

        # High-Close gap histogram
        ax = axes[0, idx]
        ax.hist(regime_data['high_close_gap_pct'], bins=30, alpha=0.7,
               color=colors[regime], edgecolor='black', linewidth=0.5)
        ax.axvline(regime_data['high_close_gap_pct'].mean(), color='red',
                  linestyle='--', linewidth=2, label=f"Mean: {regime_data['high_close_gap_pct'].mean():.3f}%")
        ax.set_title(f'{regime}: High-Close Gap', fontsize=11, fontweight='bold')
        ax.set_xlabel('Gap (%)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_xlim(0, 4)

        # Low-Close gap histogram
        ax = axes[1, idx]
        ax.hist(regime_data['low_close_gap_pct'], bins=30, alpha=0.7,
               color=colors[regime], edgecolor='black', linewidth=0.5)
        ax.axvline(regime_data['low_close_gap_pct'].mean(), color='red',
                  linestyle='--', linewidth=2, label=f"Mean: {regime_data['low_close_gap_pct'].mean():.3f}%")
        ax.set_title(f'{regime}: Close-Low Gap', fontsize=11, fontweight='bold')
        ax.set_xlabel('Gap (%)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_xlim(0, 4)

    plt.tight_layout()
    plt.savefig(output_dir / 'ohlc_gaps_histograms.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # 4. Daily range comparison
    # ========================================================================
    print("Creating daily range comparison...")

    df['daily_range_pct'] = df['high_close_gap_pct'] + df['low_close_gap_pct']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Violin plot
    sns.violinplot(data=df, x='regime_label', y='daily_range_pct',
                   ax=axes[0], palette=colors, order=['Bullish', 'Neutral', 'Bearish'])
    axes[0].set_title('Daily Range Distribution by Regime (Violin Plot)',
                     fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Regime')
    axes[0].set_ylabel('Daily Range (%)')
    axes[0].set_ylim(0, 5)

    # Bar chart with error bars
    daily_range_stats = df.groupby('regime_label')['daily_range_pct'].agg(['mean', 'std']).reset_index()
    daily_range_stats = daily_range_stats.set_index('regime_label').loc[['Bullish', 'Neutral', 'Bearish']]

    x_pos = np.arange(len(regimes))
    bars = axes[1].bar(x_pos, daily_range_stats['mean'],
                      yerr=daily_range_stats['std'],
                      color=[colors[r] for r in regimes],
                      alpha=0.8, capsize=10, ecolor='black', linewidth=2)
    axes[1].set_title('Average Daily Range by Regime',
                     fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Daily Range (%)')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(regimes)
    axes[1].grid(axis='y', alpha=0.3)

    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}%\n±{daily_range_stats["std"].iloc[i]:.3f}',
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'daily_range_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # 5. Asymmetry analysis (ratio of high-close to close-low)
    # ========================================================================
    print("Creating asymmetry analysis...")

    df['gap_ratio'] = df['high_close_gap_pct'] / (df['low_close_gap_pct'] + 1e-10)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Box plot of gap ratio
    sns.boxplot(data=df[df['gap_ratio'] < 5], x='regime_label', y='gap_ratio',
                ax=axes[0], palette=colors, order=['Bullish', 'Neutral', 'Bearish'])
    axes[0].axhline(y=1, color='black', linestyle='--', linewidth=2, alpha=0.7,
                   label='Symmetric (ratio=1)')
    axes[0].set_title('Gap Asymmetry by Regime\n(High-Close / Close-Low)',
                     fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Regime')
    axes[0].set_ylabel('Gap Ratio')
    axes[0].legend()
    axes[0].set_ylim(0, 3)

    # Mean gap ratio bar chart
    gap_ratio_means = df.groupby('regime_label')['gap_ratio'].mean()
    gap_ratio_means = gap_ratio_means.loc[['Bullish', 'Neutral', 'Bearish']]

    bars = axes[1].bar(x_pos, gap_ratio_means,
                      color=[colors[r] for r in regimes],
                      alpha=0.8)
    axes[1].axhline(y=1, color='black', linestyle='--', linewidth=2, alpha=0.7,
                   label='Symmetric (ratio=1)')
    axes[1].set_title('Average Gap Ratio by Regime\n(<1: Close near low, >1: Close near high)',
                     fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Average Gap Ratio')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(regimes)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    # Add value labels and interpretation
    for i, bar in enumerate(bars):
        height = bar.get_height()
        interpretation = "Bearish bias" if height < 0.9 else "Bullish bias" if height > 1.1 else "Balanced"
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}\n({interpretation})',
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'gap_asymmetry_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # 6. Summary statistics table as image
    # ========================================================================
    print("Creating summary statistics table...")

    summary_stats = []
    for regime in regimes:
        regime_data = df[df['regime_label'] == regime]
        summary_stats.append({
            'Regime': regime,
            'Count': len(regime_data),
            'HC Mean': f"{regime_data['high_close_gap_pct'].mean():.4f}%",
            'HC Std': f"{regime_data['high_close_gap_pct'].std():.4f}%",
            'LC Mean': f"{regime_data['low_close_gap_pct'].mean():.4f}%",
            'LC Std': f"{regime_data['low_close_gap_pct'].std():.4f}%",
            'Range Mean': f"{regime_data['daily_range_pct'].mean():.4f}%",
            'Range Std': f"{regime_data['daily_range_pct'].std():.4f}%"
        })

    df_summary = pd.DataFrame(summary_stats)

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=df_summary.values,
                    colLabels=df_summary.columns,
                    cellLoc='center',
                    loc='center',
                    colColours=['#f0f0f0']*len(df_summary.columns))

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Color code regime rows
    for i in range(len(regimes)):
        table[(i+1, 0)].set_facecolor(colors[regimes[i]])
        table[(i+1, 0)].set_text_props(weight='bold', color='white')

    plt.title('OHLC Gap Statistics Summary by Regime', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_dir / 'summary_statistics_table.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nAll visualizations saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  1. ohlc_gaps_boxplots.png - Distribution box plots")
    print("  2. ohlc_gaps_comparison.png - Mean and std comparison")
    print("  3. ohlc_gaps_histograms.png - Detailed histograms by regime")
    print("  4. daily_range_comparison.png - Daily range analysis")
    print("  5. gap_asymmetry_analysis.png - Asymmetry analysis")
    print("  6. summary_statistics_table.png - Summary table")
    print("\n" + "="*80)

if __name__ == "__main__":
    visualize_ohlc_gaps()
