"""
Visualize 20-bar Rolling CVD over years with comprehensive analysis
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def visualize_cvd20_yearly():
    """Create comprehensive yearly visualization of 20-bar rolling CVD"""

    print("="*80)
    print("20-BAR ROLLING CVD - YEARLY VISUALIZATION")
    print("="*80)

    # Load data
    print("\n1. Loading rolling CVD data...")
    df = pd.read_csv('outputs/vnindex_rolling_cvd.csv', parse_dates=['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.to_period('Q')

    print(f"   Data range: {df['date'].min().date()} to {df['date'].max().date()}")

    # Calculate additional metrics
    df['CVD_20_ma'] = df['CVD_20'].rolling(50).mean()  # 50-day MA of CVD_20
    df['CVD_20_std'] = df['CVD_20'].rolling(100).std()
    df['CVD_20_zscore'] = (df['CVD_20'] - df['CVD_20'].rolling(100).mean()) / df['CVD_20_std']

    # Calculate yearly statistics
    print("\n2. Calculating yearly statistics for 20-bar CVD...")

    yearly_stats = []
    for year in sorted(df['year'].unique()):
        year_data = df[df['year'] == year].copy()

        if len(year_data) > 0:
            yearly_stats.append({
                'Year': year,
                'Days': len(year_data),
                'Mean_CVD20': year_data['CVD_20'].mean() / 1e9,
                'Std_CVD20': year_data['CVD_20'].std() / 1e9,
                'Max_CVD20': year_data['CVD_20'].max() / 1e9,
                'Min_CVD20': year_data['CVD_20'].min() / 1e9,
                'Positive_Days': (year_data['CVD_20'] > 0).sum(),
                'Negative_Days': (year_data['CVD_20'] < 0).sum(),
                'Positive_Pct': (year_data['CVD_20'] > 0).mean() * 100,
                'Avg_BuySell_Ratio': year_data['BuySell_Ratio_20'].mean(),
                'Price_Return': (year_data['CLOSEINDEX'].iloc[-1] / year_data['CLOSEINDEX'].iloc[0] - 1) * 100
            })

    yearly_df = pd.DataFrame(yearly_stats)

    # Print summary
    print("\n3. Yearly Summary of 20-bar CVD:")
    print("-" * 100)
    print(f"{'Year':<6} {'Mean CVD20':<12} {'Std Dev':<10} {'Max CVD20':<10} {'Min CVD20':<10} {'Positive%':<10} {'B/S Ratio':<10} {'Return%':<10}")
    print("-" * 100)

    for _, row in yearly_df.iterrows():
        print(f"{int(row['Year']):<6} {row['Mean_CVD20']:>11.2f}B {row['Std_CVD20']:>9.2f}B "
              f"{row['Max_CVD20']:>9.1f}B {row['Min_CVD20']:>9.1f}B {row['Positive_Pct']:>9.1f}% "
              f"{row['Avg_BuySell_Ratio']:>9.3f} {row['Price_Return']:>9.1f}%")

    # Create comprehensive visualization
    print("\n4. Creating visualizations...")

    fig = plt.figure(figsize=(16, 20))
    gs = fig.add_gridspec(6, 2, height_ratios=[2.5, 1.5, 1.5, 1.5, 1.5, 1.5],
                          hspace=0.3, wspace=0.25)

    # 1. Full timeline of 20-bar CVD with price overlay
    ax1 = fig.add_subplot(gs[0, :])
    ax1_price = ax1.twinx()

    # Plot CVD_20
    ax1.plot(df['date'], df['CVD_20']/1e9, 'g-', linewidth=0.8, alpha=0.7, label='20-bar CVD')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

    # Add filled areas
    ax1.fill_between(df['date'], 0, df['CVD_20']/1e9,
                     where=(df['CVD_20'] > 0), alpha=0.2, color='green')
    ax1.fill_between(df['date'], 0, df['CVD_20']/1e9,
                     where=(df['CVD_20'] < 0), alpha=0.2, color='red')

    # Plot price
    ax1_price.plot(df['date'], df['CLOSEINDEX'], 'b-', linewidth=1, alpha=0.6, label='VNINDEX')

    # Add year boundaries
    for year in df['year'].unique():
        year_start = df[df['year'] == year]['date'].min()
        ax1.axvline(x=year_start, color='gray', alpha=0.3, linestyle='--', linewidth=0.5)
        ax1.text(year_start, ax1.get_ylim()[1]*0.95, str(year), fontsize=8, alpha=0.5)

    ax1.set_title('20-bar Rolling CVD vs VNINDEX (2016-2025)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('20-bar CVD (Billions)', color='g')
    ax1_price.set_ylabel('VNINDEX', color='b')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    ax1_price.legend(loc='upper right')

    # Format x-axis
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # 2. Yearly boxplot of CVD_20 values
    ax2 = fig.add_subplot(gs[1, 0])

    yearly_data = []
    yearly_labels = []
    for year in sorted(df['year'].unique()):
        year_data = df[df['year'] == year]['CVD_20'].dropna() / 1e9
        if len(year_data) > 0:
            yearly_data.append(year_data.values)
            yearly_labels.append(str(year))

    bp = ax2.boxplot(yearly_data, tick_labels=yearly_labels, patch_artist=True,
                     showmeans=True, meanline=True)

    # Color boxes based on mean
    for i, (patch, data) in enumerate(zip(bp['boxes'], yearly_data)):
        if np.mean(data) > 0:
            patch.set_facecolor('green')
        else:
            patch.set_facecolor('red')
        patch.set_alpha(0.5)

    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_title('20-bar CVD Distribution by Year', fontweight='bold')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('20-bar CVD (Billions)')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)

    # 3. Yearly mean and volatility
    ax3 = fig.add_subplot(gs[1, 1])

    ax3_std = ax3.twinx()

    bars1 = ax3.bar(yearly_df['Year'], yearly_df['Mean_CVD20'],
                    color=['green' if x > 0 else 'red' for x in yearly_df['Mean_CVD20']],
                    alpha=0.6, label='Mean CVD_20')

    line1 = ax3_std.plot(yearly_df['Year'], yearly_df['Std_CVD20'],
                         'orange', marker='o', linewidth=2, label='Std Dev')

    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_title('Yearly Mean and Volatility of 20-bar CVD', fontweight='bold')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Mean CVD_20 (Billions)', color='black')
    ax3_std.set_ylabel('Std Dev (Billions)', color='orange')
    ax3.legend(loc='upper left')
    ax3_std.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    # 4. Monthly heatmap of CVD_20
    ax4 = fig.add_subplot(gs[2, :])

    # Create monthly average pivot table
    monthly_pivot = df.pivot_table(values='CVD_20',
                                   index=df['year'],
                                   columns=df['month'],
                                   aggfunc='mean')
    monthly_pivot = monthly_pivot / 1e9  # Convert to billions

    # Create heatmap
    sns.heatmap(monthly_pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                ax=ax4, cbar_kws={'label': 'Mean 20-bar CVD (Billions)'},
                linewidths=0.5, linecolor='gray')
    ax4.set_title('Monthly Average 20-bar CVD Heatmap (Billions)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Month')
    ax4.set_ylabel('Year')

    # 5. CVD_20 Z-score over time
    ax5 = fig.add_subplot(gs[3, 0])

    ax5.plot(df['date'], df['CVD_20_zscore'], 'purple', linewidth=0.8, alpha=0.7)
    ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax5.axhline(y=2, color='red', linestyle='--', alpha=0.3, label='Overbought')
    ax5.axhline(y=-2, color='green', linestyle='--', alpha=0.3, label='Oversold')

    ax5.fill_between(df['date'], -2, 2, alpha=0.1, color='gray')

    # Highlight extreme periods
    extreme_high = df[df['CVD_20_zscore'] > 2]
    extreme_low = df[df['CVD_20_zscore'] < -2]

    ax5.scatter(extreme_high['date'], extreme_high['CVD_20_zscore'],
               color='red', s=20, alpha=0.5, label=f'Extreme High ({len(extreme_high)} days)')
    ax5.scatter(extreme_low['date'], extreme_low['CVD_20_zscore'],
               color='green', s=20, alpha=0.5, label=f'Extreme Low ({len(extreme_low)} days)')

    ax5.set_title('20-bar CVD Z-Score (100-day normalized)', fontweight='bold')
    ax5.set_xlabel('Date')
    ax5.set_ylabel('Z-Score')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # 6. Positive/Negative days ratio by year
    ax6 = fig.add_subplot(gs[3, 1])

    width = 0.35
    years_pos = yearly_df['Year'].values
    x = np.arange(len(years_pos))

    bars1 = ax6.bar(x - width/2, yearly_df['Positive_Days'], width,
                    label='Positive CVD Days', color='green', alpha=0.7)
    bars2 = ax6.bar(x + width/2, yearly_df['Negative_Days'], width,
                    label='Negative CVD Days', color='red', alpha=0.7)

    ax6.set_xlabel('Year')
    ax6.set_ylabel('Number of Days')
    ax6.set_title('Positive vs Negative CVD_20 Days by Year', fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(years_pos.astype(int), rotation=45)
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # 7. Rolling Buy/Sell Ratio (20-bar)
    ax7 = fig.add_subplot(gs[4, 0])

    ax7.plot(df['date'], df['BuySell_Ratio_20'], 'purple', linewidth=0.8, alpha=0.7)
    ax7.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    ax7.axhline(y=1.2, color='green', linestyle=':', alpha=0.3)
    ax7.axhline(y=0.8, color='red', linestyle=':', alpha=0.3)

    ax7.fill_between(df['date'], 1, df['BuySell_Ratio_20'],
                     where=(df['BuySell_Ratio_20'] > 1), alpha=0.2, color='green')
    ax7.fill_between(df['date'], 1, df['BuySell_Ratio_20'],
                     where=(df['BuySell_Ratio_20'] < 1), alpha=0.2, color='red')

    ax7.set_title('20-bar Buy/Sell Volume Ratio Timeline', fontweight='bold')
    ax7.set_xlabel('Date')
    ax7.set_ylabel('Buy/Sell Ratio')
    ax7.grid(True, alpha=0.3)

    # 8. Scatter plot: CVD_20 Mean vs Price Return by Year
    ax8 = fig.add_subplot(gs[4, 1])

    colors_scatter = []
    for year in yearly_df['Year']:
        if year < 2020:
            colors_scatter.append('blue')
        elif year < 2023:
            colors_scatter.append('orange')
        else:
            colors_scatter.append('green')

    ax8.scatter(yearly_df['Mean_CVD20'], yearly_df['Price_Return'],
               c=colors_scatter, s=100, alpha=0.7, edgecolors='black')

    # Add year labels
    for idx, row in yearly_df.iterrows():
        ax8.annotate(str(int(row['Year'])),
                    (row['Mean_CVD20'], row['Price_Return']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax8.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax8.axvline(x=0, color='black', linestyle='--', alpha=0.3)

    # Add correlation
    corr = yearly_df['Mean_CVD20'].corr(yearly_df['Price_Return'])
    ax8.text(0.05, 0.95, f'Correlation: {corr:.3f}',
            transform=ax8.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax8.set_xlabel('Mean 20-bar CVD (Billions)')
    ax8.set_ylabel('Yearly Price Return (%)')
    ax8.set_title('CVD_20 vs Price Performance by Year', fontweight='bold')
    ax8.grid(True, alpha=0.3)

    # 9. Quarterly CVD_20 trend
    ax9 = fig.add_subplot(gs[5, :])

    quarterly = df.groupby('quarter').agg({
        'CVD_20': 'mean',
        'BuySell_Ratio_20': 'mean',
        'CLOSEINDEX': 'last'
    }).reset_index()

    quarterly['CVD_20'] = quarterly['CVD_20'] / 1e9
    quarters_str = quarterly['quarter'].astype(str)

    ax9_ratio = ax9.twinx()

    line1 = ax9.plot(range(len(quarterly)), quarterly['CVD_20'],
                     'g-', marker='o', linewidth=2, label='Mean CVD_20')
    line2 = ax9_ratio.plot(range(len(quarterly)), quarterly['BuySell_Ratio_20'],
                          'orange', marker='s', linewidth=2, alpha=0.7, label='Buy/Sell Ratio')

    ax9.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax9_ratio.axhline(y=1, color='black', linestyle=':', alpha=0.3)

    ax9.set_xlabel('Quarter')
    ax9.set_ylabel('Mean 20-bar CVD (Billions)', color='g')
    ax9_ratio.set_ylabel('Buy/Sell Ratio', color='orange')
    ax9.set_title('Quarterly 20-bar CVD and Buy/Sell Ratio Trends', fontweight='bold')

    # Set x-axis labels (every 4th quarter)
    ax9.set_xticks(range(0, len(quarterly), 4))
    ax9.set_xticklabels([quarters_str.iloc[i] if i < len(quarters_str) else ''
                        for i in range(0, len(quarterly), 4)], rotation=45)

    ax9.legend(loc='upper left')
    ax9_ratio.legend(loc='upper right')
    ax9.grid(True, alpha=0.3)

    plt.suptitle('20-bar Rolling CVD Analysis - VNINDEX (2016-2025)',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    # Save figure
    output_path = 'outputs/cvd20_yearly_visualization.png'
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"\n   Visualization saved to: {output_path}")

    # Additional analysis
    print("\n" + "="*80)
    print("KEY INSIGHTS FROM 20-BAR CVD ANALYSIS")
    print("="*80)

    # Current state
    current_cvd20 = df['CVD_20'].iloc[-1] / 1e9
    current_zscore = df['CVD_20_zscore'].iloc[-1]
    current_ratio = df['BuySell_Ratio_20'].iloc[-1]

    print(f"\n1. Current State (as of {df['date'].iloc[-1].date()}):")
    print(f"   20-bar CVD: {current_cvd20:.3f} Billion")
    print(f"   Z-Score: {current_zscore:.2f}")
    print(f"   Buy/Sell Ratio: {current_ratio:.3f}")

    # Best/Worst years
    best_year = yearly_df.loc[yearly_df['Mean_CVD20'].idxmax()]
    worst_year = yearly_df.loc[yearly_df['Mean_CVD20'].idxmin()]

    print(f"\n2. Best/Worst Years:")
    print(f"   Best Year: {int(best_year['Year'])} (Mean CVD: {best_year['Mean_CVD20']:.2f}B, Return: {best_year['Price_Return']:.1f}%)")
    print(f"   Worst Year: {int(worst_year['Year'])} (Mean CVD: {worst_year['Mean_CVD20']:.2f}B, Return: {worst_year['Price_Return']:.1f}%)")

    # Extremes
    total_extreme_high = (df['CVD_20_zscore'] > 2).sum()
    total_extreme_low = (df['CVD_20_zscore'] < -2).sum()

    print(f"\n3. Extreme Periods:")
    print(f"   Days with Z-score > 2: {total_extreme_high} ({total_extreme_high/len(df)*100:.2f}%)")
    print(f"   Days with Z-score < -2: {total_extreme_low} ({total_extreme_low/len(df)*100:.2f}%)")

    # Overall statistics
    overall_positive = (df['CVD_20'] > 0).mean() * 100
    overall_mean = df['CVD_20'].mean() / 1e9

    print(f"\n4. Overall Statistics:")
    print(f"   Percentage of positive CVD_20 days: {overall_positive:.1f}%")
    print(f"   Overall mean CVD_20: {overall_mean:.3f} Billion")
    print(f"   Correlation with price returns: {corr:.3f}")

    print("\n" + "="*80)

    return yearly_df

if __name__ == "__main__":
    yearly_stats = visualize_cvd20_yearly()