"""
Create focused CVD trend visualizations
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

def create_cvd_trend_analysis():
    """Create focused CVD trend visualizations"""

    print("="*80)
    print("CVD TREND ANALYSIS")
    print("="*80)

    # Load data
    df = pd.read_csv('outputs/vnindex_cvd_analysis.csv', parse_dates=['date'])
    df['year'] = df['date'].dt.year

    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # 1. CVD Evolution with Trend
    ax1 = axes[0, 0]
    ax1.plot(df['date'], df['CVD']/1e9, 'b-', linewidth=1, alpha=0.6, label='CVD')

    # Add trend line
    x_numeric = np.arange(len(df))
    z = np.polyfit(x_numeric, df['CVD'].values/1e9, 1)
    p = np.poly1d(z)
    ax1.plot(df['date'], p(x_numeric), "r--", linewidth=2, label=f'Trend (slope: {z[0]:.3f}B/day)')

    ax1.set_title('CVD Evolution with Linear Trend', fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('CVD (Billions)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. CVD Momentum (Rate of Change)
    ax2 = axes[0, 1]

    # Calculate CVD momentum (30-day change)
    df['CVD_Momentum'] = df['CVD'].diff(30) / 1e9

    ax2.plot(df['date'], df['CVD_Momentum'], 'g-', linewidth=1)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.fill_between(df['date'], 0, df['CVD_Momentum'],
                     where=(df['CVD_Momentum'] > 0), alpha=0.3, color='green', label='Accumulation')
    ax2.fill_between(df['date'], 0, df['CVD_Momentum'],
                     where=(df['CVD_Momentum'] < 0), alpha=0.3, color='red', label='Distribution')

    ax2.set_title('CVD Momentum (30-day Change)', fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('30-day CVD Change (B)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Rolling Buy/Sell Ratio
    ax3 = axes[1, 0]

    # Calculate rolling buy/sell ratio (60-day window)
    window = 60
    df['Rolling_Buy'] = df['BUY_VOLUME'].rolling(window).sum()
    df['Rolling_Sell'] = df['SELL_VOLUME'].rolling(window).sum()
    df['Rolling_Ratio'] = df['Rolling_Buy'] / df['Rolling_Sell']

    ax3.plot(df['date'], df['Rolling_Ratio'], 'purple', linewidth=1.5)
    ax3.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    ax3.fill_between(df['date'], 1, df['Rolling_Ratio'],
                     where=(df['Rolling_Ratio'] > 1), alpha=0.3, color='green')
    ax3.fill_between(df['date'], 1, df['Rolling_Ratio'],
                     where=(df['Rolling_Ratio'] < 1), alpha=0.3, color='red')

    ax3.set_title(f'Rolling Buy/Sell Ratio ({window}-day)', fontweight='bold')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Buy/Sell Ratio')
    ax3.set_ylim(0.5, 2.0)
    ax3.grid(True, alpha=0.3)

    # 4. CVD vs Price Correlation
    ax4 = axes[1, 1]

    # Calculate rolling correlation
    window = 120
    df['Price_Returns'] = df['CLOSEINDEX'].pct_change()
    df['CVD_Returns'] = df['CVD'].pct_change()

    rolling_corr = df['Price_Returns'].rolling(window).corr(df['CVD_Returns'])

    ax4.plot(df['date'], rolling_corr, 'orange', linewidth=1.5)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.axhline(y=0.5, color='green', linestyle=':', alpha=0.5)
    ax4.axhline(y=-0.5, color='red', linestyle=':', alpha=0.5)
    ax4.fill_between(df['date'], 0, rolling_corr,
                     where=(rolling_corr > 0), alpha=0.3, color='green')
    ax4.fill_between(df['date'], 0, rolling_corr,
                     where=(rolling_corr < 0), alpha=0.3, color='red')

    ax4.set_title(f'Price-CVD Rolling Correlation ({window}-day)', fontweight='bold')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Correlation')
    ax4.set_ylim(-1, 1)
    ax4.grid(True, alpha=0.3)

    # 5. Volume Delta Extremes
    ax5 = axes[2, 0]

    # Identify extreme days (top/bottom 5%)
    percentile_95 = df['VOLUME_DELTA'].quantile(0.95)
    percentile_5 = df['VOLUME_DELTA'].quantile(0.05)

    # Plot all volume delta
    colors = ['green' if x > 0 else 'red' for x in df['VOLUME_DELTA']]
    ax5.scatter(df['date'], df['VOLUME_DELTA']/1e9, c=colors, alpha=0.3, s=1)

    # Highlight extremes
    extreme_buy = df[df['VOLUME_DELTA'] > percentile_95]
    extreme_sell = df[df['VOLUME_DELTA'] < percentile_5]

    ax5.scatter(extreme_buy['date'], extreme_buy['VOLUME_DELTA']/1e9,
               color='darkgreen', s=50, marker='^', label='Extreme Buy (>95%ile)')
    ax5.scatter(extreme_sell['date'], extreme_sell['VOLUME_DELTA']/1e9,
               color='darkred', s=50, marker='v', label='Extreme Sell (<5%ile)')

    ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax5.set_title('Volume Delta with Extremes Highlighted', fontweight='bold')
    ax5.set_xlabel('Date')
    ax5.set_ylabel('Volume Delta (B)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. CVD Regime Analysis
    ax6 = axes[2, 1]

    # Define CVD regimes based on momentum
    df['CVD_MA50'] = df['CVD'].rolling(50).mean()
    df['CVD_MA200'] = df['CVD'].rolling(200).mean()

    # Calculate regime
    df['CVD_Regime'] = 'Neutral'
    df.loc[(df['CVD'] > df['CVD_MA50']) & (df['CVD_MA50'] > df['CVD_MA200']), 'CVD_Regime'] = 'Bullish'
    df.loc[(df['CVD'] < df['CVD_MA50']) & (df['CVD_MA50'] < df['CVD_MA200']), 'CVD_Regime'] = 'Bearish'

    # Count regime days by year
    regime_counts = df.groupby(['year', 'CVD_Regime']).size().unstack(fill_value=0)

    # Create stacked bar chart
    regime_counts.plot(kind='bar', stacked=True, ax=ax6,
                       color=['red', 'green', 'yellow'], alpha=0.7)

    ax6.set_title('CVD Regime Distribution by Year', fontweight='bold')
    ax6.set_xlabel('Year')
    ax6.set_ylabel('Trading Days')
    ax6.legend(title='CVD Regime')
    ax6.grid(True, alpha=0.3)
    ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45)

    plt.suptitle('CVD Trend Analysis - VNINDEX', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save figure
    output_path = 'outputs/cvd_trend_analysis.png'
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")

    # Print key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)

    # Current state
    current_cvd = df['CVD'].iloc[-1] / 1e9
    cvd_ma50 = df['CVD_MA50'].iloc[-1] / 1e9
    cvd_ma200 = df['CVD_MA200'].iloc[-1] / 1e9
    current_regime = df['CVD_Regime'].iloc[-1]

    print(f"\n1. Current CVD State:")
    print(f"   Current CVD: {current_cvd:.2f} Billion")
    print(f"   50-day MA: {cvd_ma50:.2f} Billion")
    print(f"   200-day MA: {cvd_ma200:.2f} Billion")
    print(f"   Current Regime: {current_regime}")

    # Trend analysis
    print(f"\n2. Long-term Trend:")
    print(f"   Daily CVD growth rate: {z[0]:.3f} Billion/day")
    print(f"   Annualized growth: {z[0] * 252:.1f} Billion/year")

    # Extremes
    print(f"\n3. Volume Delta Extremes:")
    print(f"   Days with extreme buying (>95%ile): {len(extreme_buy)} ({len(extreme_buy)/len(df)*100:.2f}%)")
    print(f"   Days with extreme selling (<5%ile): {len(extreme_sell)} ({len(extreme_sell)/len(df)*100:.2f}%)")

    # Correlation
    recent_corr = rolling_corr.iloc[-1]
    avg_corr = rolling_corr.mean()
    print(f"\n4. Price-CVD Relationship:")
    print(f"   Current correlation: {recent_corr:.3f}")
    print(f"   Average correlation: {avg_corr:.3f}")

    # Buy/Sell momentum
    current_ratio = df['Rolling_Ratio'].iloc[-1]
    print(f"\n5. Current Buy/Sell Momentum:")
    print(f"   60-day Buy/Sell Ratio: {current_ratio:.3f}")
    if current_ratio > 1.2:
        print("   Status: Strong Buying Pressure")
    elif current_ratio > 1.0:
        print("   Status: Moderate Buying Pressure")
    elif current_ratio > 0.8:
        print("   Status: Moderate Selling Pressure")
    else:
        print("   Status: Strong Selling Pressure")

    print("\n" + "="*80)

    return df

if __name__ == "__main__":
    df_analysis = create_cvd_trend_analysis()