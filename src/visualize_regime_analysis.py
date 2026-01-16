#!/usr/bin/env python
"""
Visualize CJM K=3 regimes on VNINDEX chart and analyze yearly distribution
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_regime_data():
    """Load regime predictions and price data"""

    # Load regime predictions
    regime_path = Path('outputs/step4_apply/regimes_daily_K3.csv')
    df = pd.read_csv(regime_path, parse_dates=['date'])

    # Load full price data for OHLC
    price_path = Path('data/vnindex_full.csv')
    df_price = pd.read_csv(price_path, parse_dates=['TRADINGDATE'])

    # Select and rename columns BEFORE merge to avoid duplicates
    df_price = df_price[['TRADINGDATE', 'OPENINDEX', 'HIGHESTINDEX',
                         'LOWESTINDEX', 'CLOSEINDEX', 'TOTALMATCHVOLUME']].copy()

    df_price = df_price.rename(columns={
        'TRADINGDATE': 'date',
        'OPENINDEX': 'open',
        'HIGHESTINDEX': 'high',
        'LOWESTINDEX': 'low',
        'CLOSEINDEX': 'close_price',
        'TOTALMATCHVOLUME': 'volume'
    })

    # Merge dataframes
    df = df.merge(df_price[['date', 'open', 'high', 'low', 'close_price', 'volume']],
                  on='date', how='left')

    # Add year column
    df['year'] = df['date'].dt.year

    return df


def create_regime_chart(df):
    """Create candlestick chart with regime overlay"""

    # Define colors for regimes
    regime_colors = {
        0: '#FFE6E6',  # Light pink for neutral
        1: '#E6FFE6',  # Light green for bullish
        2: '#FFE6E6'   # Light red for bearish
    }

    regime_names = {
        0: 'Neutral',
        1: 'Bullish',
        2: 'Bearish'
    }

    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 12),
                                          gridspec_kw={'height_ratios': [3, 1, 1]})

    # Plot 1: Price with regime backgrounds
    ax1.set_title('VNINDEX with CJM K=3 Regime Classification (2018-2025)',
                  fontsize=16, fontweight='bold')

    # Add regime backgrounds
    current_regime = df.iloc[0]['regime']
    start_idx = 0

    for i in range(1, len(df)):
        if df.iloc[i]['regime'] != current_regime or i == len(df) - 1:
            # Draw rectangle for current regime
            end_idx = i if i < len(df) - 1 else len(df)

            if current_regime == 0:
                color = 'lightgray'
                alpha = 0.3
            elif current_regime == 1:
                color = 'lightgreen'
                alpha = 0.3
            else:  # regime 2
                color = 'lightcoral'
                alpha = 0.3

            ax1.axvspan(df.iloc[start_idx]['date'], df.iloc[end_idx-1]['date'],
                       facecolor=color, alpha=alpha)

            # Update for next regime
            current_regime = df.iloc[i]['regime']
            start_idx = i

    # Plot price line
    ax1.plot(df['date'], df['close'], 'k-', linewidth=1.5, label='VNINDEX')

    # Add moving averages for reference
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma50'] = df['close'].rolling(50).mean()
    ax1.plot(df['date'], df['ma20'], 'b-', linewidth=0.8, alpha=0.5, label='MA20')
    ax1.plot(df['date'], df['ma50'], 'r-', linewidth=0.8, alpha=0.5, label='MA50')

    ax1.set_ylabel('VNINDEX Points', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')

    # Add regime labels
    patches = [
        mpatches.Patch(color='lightgray', alpha=0.3, label='Regime 0: Neutral'),
        mpatches.Patch(color='lightgreen', alpha=0.3, label='Regime 1: Bullish'),
        mpatches.Patch(color='lightcoral', alpha=0.3, label='Regime 2: Bearish')
    ]
    ax1.legend(handles=patches, loc='upper right')

    # Plot 2: Regime indicator
    ax2.set_title('Regime State', fontsize=12)
    ax2.fill_between(df['date'], 0, df['regime'],
                     where=(df['regime']==0), color='gray', alpha=0.5, label='Neutral')
    ax2.fill_between(df['date'], 0, df['regime'],
                     where=(df['regime']==1), color='green', alpha=0.5, label='Bullish')
    ax2.fill_between(df['date'], 0, df['regime'],
                     where=(df['regime']==2), color='red', alpha=0.5, label='Bearish')
    ax2.set_ylabel('Regime', fontsize=10)
    ax2.set_ylim(-0.2, 2.2)
    ax2.set_yticks([0, 1, 2])
    ax2.grid(True, alpha=0.3)

    # Plot 3: Daily returns colored by regime
    ax3.set_title('Daily Returns by Regime', fontsize=12)
    colors = df['regime'].map({0: 'gray', 1: 'green', 2: 'red'})
    ax3.bar(df['date'], df['ret']*100, color=colors, alpha=0.6, width=1)
    ax3.set_ylabel('Return (%)', fontsize=10)
    ax3.set_xlabel('Date', fontsize=10)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_path = Path('outputs/visualizations')
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / 'regime_chart_full.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved regime chart to: {output_path / 'regime_chart_full.png'}")


def create_yearly_distribution(df):
    """Create regime distribution by year"""

    # Calculate regime distribution by year
    yearly_regime = df.groupby(['year', 'regime']).size().unstack(fill_value=0)
    yearly_regime_pct = yearly_regime.div(yearly_regime.sum(axis=1), axis=0) * 100

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot 1: Stacked bar chart (percentage)
    yearly_regime_pct.plot(kind='bar', stacked=True, ax=ax1,
                           color=['gray', 'green', 'red'], alpha=0.8)
    ax1.set_title('Regime Distribution by Year (%)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Percentage of Days', fontsize=12)
    ax1.legend(title='Regime', labels=['0: Neutral', '1: Bullish', '2: Bearish'])
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)

    # Add percentage labels on bars
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.0f%%', label_type='center')

    # Subplot 2: Grouped bar chart (absolute days)
    yearly_regime.plot(kind='bar', ax=ax2, color=['gray', 'green', 'red'], alpha=0.8)
    ax2.set_title('Regime Distribution by Year (Days)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Number of Days', fontsize=12)
    ax2.legend(title='Regime', labels=['0: Neutral', '1: Bullish', '2: Bearish'])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_path = Path('outputs/visualizations')
    plt.savefig(output_path / 'regime_yearly_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved yearly distribution to: {output_path / 'regime_yearly_distribution.png'}")

    # Print statistics
    print("\n📊 Regime Distribution by Year:")
    print("=" * 60)
    print("\nPercentage Distribution:")
    print(yearly_regime_pct.round(1))
    print("\nAbsolute Days:")
    print(yearly_regime)

    return yearly_regime_pct


def create_regime_statistics_table(df):
    """Create detailed regime statistics table"""

    # Calculate statistics by year and regime
    stats = []

    for year in df['year'].unique():
        year_data = df[df['year'] == year]

        for regime in [0, 1, 2]:
            regime_data = year_data[year_data['regime'] == regime]

            if len(regime_data) > 0:
                stats.append({
                    'Year': year,
                    'Regime': regime,
                    'Days': len(regime_data),
                    'Pct': len(regime_data) / len(year_data) * 100,
                    'Avg_Return': regime_data['ret'].mean() * 100,
                    'Volatility': regime_data['ret'].std() * 100,
                    'Max_Return': regime_data['ret'].max() * 100,
                    'Min_Return': regime_data['ret'].min() * 100
                })

    stats_df = pd.DataFrame(stats)

    # Create pivot tables for visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1. Heatmap of percentage distribution
    pivot_pct = stats_df.pivot(index='Year', columns='Regime', values='Pct')
    sns.heatmap(pivot_pct, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[0,0],
                cbar_kws={'label': '% of Days'})
    axes[0,0].set_title('Regime Distribution Heatmap (% of Days)', fontsize=12, fontweight='bold')
    axes[0,0].set_xlabel('Regime')
    axes[0,0].set_ylabel('Year')

    # 2. Average returns by regime and year
    pivot_ret = stats_df.pivot(index='Year', columns='Regime', values='Avg_Return')
    sns.heatmap(pivot_ret, annot=True, fmt='.2f', cmap='RdYlGn', center=0, ax=axes[0,1],
                cbar_kws={'label': 'Avg Daily Return %'})
    axes[0,1].set_title('Average Daily Returns by Regime and Year (%)', fontsize=12, fontweight='bold')
    axes[0,1].set_xlabel('Regime')
    axes[0,1].set_ylabel('Year')

    # 3. Volatility by regime and year
    pivot_vol = stats_df.pivot(index='Year', columns='Regime', values='Volatility')
    sns.heatmap(pivot_vol, annot=True, fmt='.2f', cmap='Blues', ax=axes[1,0],
                cbar_kws={'label': 'Daily Volatility %'})
    axes[1,0].set_title('Volatility by Regime and Year (%)', fontsize=12, fontweight='bold')
    axes[1,0].set_xlabel('Regime')
    axes[1,0].set_ylabel('Year')

    # 4. Summary statistics
    axes[1,1].axis('tight')
    axes[1,1].axis('off')

    summary_text = f"""
    Summary Statistics (2018-2025):

    Regime 0 (Neutral): {df[df['regime']==0].shape[0]} days ({df[df['regime']==0].shape[0]/len(df)*100:.1f}%)
    • Average Return: {df[df['regime']==0]['ret'].mean()*100:.3f}%
    • Volatility: {df[df['regime']==0]['ret'].std()*100:.3f}%

    Regime 1 (Bullish): {df[df['regime']==1].shape[0]} days ({df[df['regime']==1].shape[0]/len(df)*100:.1f}%)
    • Average Return: {df[df['regime']==1]['ret'].mean()*100:.3f}%
    • Volatility: {df[df['regime']==1]['ret'].std()*100:.3f}%

    Regime 2 (Bearish): {df[df['regime']==2].shape[0]} days ({df[df['regime']==2].shape[0]/len(df)*100:.1f}%)
    • Average Return: {df[df['regime']==2]['ret'].mean()*100:.3f}%
    • Volatility: {df[df['regime']==2]['ret'].std()*100:.3f}%

    Most Bullish Year: {pivot_pct[1].idxmax()} ({pivot_pct[1].max():.1f}% bullish days)
    Most Bearish Year: {pivot_pct[2].idxmax()} ({pivot_pct[2].max():.1f}% bearish days)
    Most Neutral Year: {pivot_pct[0].idxmax()} ({pivot_pct[0].max():.1f}% neutral days)
    """

    axes[1,1].text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
                   fontfamily='monospace')

    plt.suptitle('CJM K=3 Regime Analysis by Year (2018-2025)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save
    output_path = Path('outputs/visualizations')
    plt.savefig(output_path / 'regime_statistics_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved statistics heatmap to: {output_path / 'regime_statistics_heatmap.png'}")

    return stats_df


def create_regime_transitions_by_year(df):
    """Analyze regime transitions by year"""

    # Calculate transitions by year
    transition_stats = []

    for year in df['year'].unique():
        year_data = df[df['year'] == year].reset_index(drop=True)

        if len(year_data) > 1:
            # Count transitions
            transitions = (year_data['regime'].diff() != 0).sum() - 1

            # Count specific transitions
            trans_matrix = np.zeros((3, 3))
            for i in range(len(year_data) - 1):
                from_state = int(year_data.iloc[i]['regime'])
                to_state = int(year_data.iloc[i+1]['regime'])
                trans_matrix[from_state, to_state] += 1

            transition_stats.append({
                'Year': year,
                'Total_Transitions': transitions,
                'Days': len(year_data),
                'Transition_Rate': transitions / len(year_data) * 100,
                'Avg_Regime_Duration': len(year_data) / (transitions + 1)
            })

    trans_df = pd.DataFrame(transition_stats)

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart of transitions by year
    axes[0].bar(trans_df['Year'], trans_df['Total_Transitions'], color='steelblue', alpha=0.7)
    axes[0].set_title('Number of Regime Transitions by Year', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Year')
    axes[0].set_ylabel('Number of Transitions')
    axes[0].grid(True, alpha=0.3)

    # Average regime duration by year
    axes[1].bar(trans_df['Year'], trans_df['Avg_Regime_Duration'], color='coral', alpha=0.7)
    axes[1].set_title('Average Regime Duration by Year (Days)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Year')
    axes[1].set_ylabel('Average Duration (Days)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_path = Path('outputs/visualizations')
    plt.savefig(output_path / 'regime_transitions_by_year.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved transitions analysis to: {output_path / 'regime_transitions_by_year.png'}")

    # Print statistics
    print("\n🔄 Regime Transitions by Year:")
    print("=" * 60)
    print(trans_df.to_string(index=False))

    return trans_df


def main():
    """Main function to create all visualizations"""

    print("\n" + "="*60)
    print("Creating Regime Visualizations and Analysis")
    print("="*60)

    # Load data
    df = load_regime_data()
    print(f"\n✅ Loaded {len(df)} days of data")

    # Create visualizations
    print("\n📊 Generating visualizations...")

    # 1. Full regime chart
    create_regime_chart(df)

    # 2. Yearly distribution
    yearly_dist = create_yearly_distribution(df)

    # 3. Statistics heatmaps
    stats_df = create_regime_statistics_table(df)

    # 4. Transitions analysis
    trans_df = create_regime_transitions_by_year(df)

    print("\n" + "="*60)
    print("✅ All visualizations created successfully!")
    print("="*60)
    print("\nGenerated files in outputs/visualizations/:")
    print("  • regime_chart_full.png - Full timeline with regimes")
    print("  • regime_yearly_distribution.png - Distribution by year")
    print("  • regime_statistics_heatmap.png - Detailed statistics")
    print("  • regime_transitions_by_year.png - Transition analysis")


if __name__ == '__main__':
    main()