"""
Visualize Volume Delta and CVD over years with comprehensive analysis
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def create_yearly_visualization():
    """Create comprehensive yearly visualization of Volume Delta and CVD"""

    print("="*80)
    print("VOLUME DELTA & CVD YEARLY VISUALIZATION")
    print("="*80)

    # Load data
    print("\n1. Loading CVD data...")
    df = pd.read_csv('outputs/vnindex_cvd_analysis.csv', parse_dates=['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    print(f"   Data range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"   Total trading days: {len(df)}")

    # Calculate yearly statistics
    print("\n2. Calculating yearly statistics...")

    yearly_stats = []
    for year in sorted(df['year'].unique()):
        year_data = df[df['year'] == year]

        # Calculate metrics
        total_buy = year_data['BUY_VOLUME'].sum()
        total_sell = year_data['SELL_VOLUME'].sum()
        net_delta = year_data['VOLUME_DELTA'].sum()
        avg_daily_delta = year_data['VOLUME_DELTA'].mean()

        # Price performance
        if len(year_data) > 0:
            year_return = (year_data['CLOSEINDEX'].iloc[-1] / year_data['CLOSEINDEX'].iloc[0] - 1) * 100
        else:
            year_return = 0

        # CVD change
        cvd_start = year_data['CVD'].iloc[0] if len(year_data) > 0 else 0
        cvd_end = year_data['CVD'].iloc[-1] if len(year_data) > 0 else 0
        cvd_change = cvd_end - cvd_start

        yearly_stats.append({
            'Year': year,
            'Trading_Days': len(year_data),
            'Total_Buy_Vol': total_buy,
            'Total_Sell_Vol': total_sell,
            'Net_Delta': net_delta,
            'Buy_Sell_Ratio': total_buy / total_sell if total_sell > 0 else 0,
            'Avg_Daily_Delta': avg_daily_delta,
            'Price_Return_%': year_return,
            'CVD_Change': cvd_change,
            'Avg_Daily_Volume': year_data['TOTALMATCHVOLUME'].mean()
        })

    yearly_df = pd.DataFrame(yearly_stats)

    # Print yearly summary
    print("\n3. Yearly Summary:")
    print("-" * 100)
    print(f"{'Year':<6} {'Days':<6} {'Buy/Sell Ratio':<15} {'Net Delta (B)':<15} {'Price Return':<12} {'CVD Change (B)':<15}")
    print("-" * 100)

    for _, row in yearly_df.iterrows():
        print(f"{int(row['Year']):<6} {row['Trading_Days']:<6} "
              f"{row['Buy_Sell_Ratio']:<15.3f} {row['Net_Delta']/1e9:<15.2f} "
              f"{row['Price_Return_%']:<12.2f}% {row['CVD_Change']/1e9:<15.2f}")

    # Create comprehensive visualization
    print("\n4. Creating visualizations...")

    fig = plt.figure(figsize=(16, 20))

    # Create subplots
    gs = fig.add_gridspec(6, 2, height_ratios=[2, 1.5, 1.5, 1.5, 1.5, 1.2],
                          width_ratios=[2, 1], hspace=0.3, wspace=0.25)

    # 1. Price with CVD overlay (full width, top)
    ax1 = fig.add_subplot(gs[0, :])
    ax1_cvd = ax1.twinx()

    # Plot price
    ax1.plot(df['date'], df['CLOSEINDEX'], 'b-', linewidth=1.5, label='VNINDEX')
    ax1.fill_between(df['date'], df['CLOSEINDEX'], alpha=0.1, color='blue')

    # Plot CVD
    ax1_cvd.plot(df['date'], df['CVD']/1e9, 'g-', linewidth=1.5, alpha=0.7, label='CVD (Billions)')

    # Add year boundaries
    for year in df['year'].unique():
        year_start = df[df['year'] == year]['date'].min()
        ax1.axvline(x=year_start, color='gray', alpha=0.3, linestyle='--', linewidth=0.5)

    ax1.set_title('VNINDEX Price and Cumulative Volume Delta (2016-2025)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('VNINDEX', color='b')
    ax1_cvd.set_ylabel('CVD (Billions)', color='g')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    ax1_cvd.legend(loc='upper right')

    # Format x-axis
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_minor_locator(mdates.MonthLocator((1, 7)))

    # 2. Daily Volume Delta (full width)
    ax2 = fig.add_subplot(gs[1, :])

    # Create color array based on positive/negative
    colors = ['green' if x > 0 else 'red' for x in df['VOLUME_DELTA']]
    ax2.bar(df['date'], df['VOLUME_DELTA']/1e9, color=colors, alpha=0.6, width=1)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    # Add 30-day moving average
    ma30 = df['VOLUME_DELTA'].rolling(30).mean()
    ax2.plot(df['date'], ma30/1e9, 'orange', linewidth=1.5, alpha=0.8, label='30-day MA')

    ax2.set_title('Daily Volume Delta (Billions)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Volume Delta (B)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Yearly Buy/Sell Ratio
    ax3 = fig.add_subplot(gs[2, 0])

    bars = ax3.bar(yearly_df['Year'], yearly_df['Buy_Sell_Ratio'],
                   color=['green' if x > 1 else 'red' for x in yearly_df['Buy_Sell_Ratio']],
                   alpha=0.7)
    ax3.axhline(y=1, color='black', linestyle='--', alpha=0.5)

    # Add value labels
    for bar, ratio in zip(bars, yearly_df['Buy_Sell_Ratio']):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{ratio:.3f}', ha='center', va='bottom', fontsize=9)

    ax3.set_title('Yearly Buy/Sell Volume Ratio', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Buy/Sell Ratio')
    ax3.set_ylim(0.8, max(yearly_df['Buy_Sell_Ratio']) * 1.1)
    ax3.grid(True, alpha=0.3)

    # 4. Yearly Net Volume Delta
    ax4 = fig.add_subplot(gs[2, 1])

    bars = ax4.bar(yearly_df['Year'], yearly_df['Net_Delta']/1e9,
                   color=['green' if x > 0 else 'red' for x in yearly_df['Net_Delta']],
                   alpha=0.7)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    ax4.set_title('Yearly Net Volume Delta', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Net Delta (Billions)')
    ax4.grid(True, alpha=0.3)

    # 5. Monthly Volume Delta Heatmap
    ax5 = fig.add_subplot(gs[3, :])

    # Create monthly pivot table
    monthly_pivot = df.pivot_table(values='VOLUME_DELTA',
                                   index=df['year'],
                                   columns=df['month'],
                                   aggfunc='sum')
    monthly_pivot = monthly_pivot / 1e9  # Convert to billions

    # Create heatmap
    sns.heatmap(monthly_pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                ax=ax5, cbar_kws={'label': 'Net Volume Delta (Billions)'},
                linewidths=0.5, linecolor='gray')
    ax5.set_title('Monthly Net Volume Delta Heatmap (Billions)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Month')
    ax5.set_ylabel('Year')

    # 6. CVD Growth Rate
    ax6 = fig.add_subplot(gs[4, 0])

    # Calculate quarterly CVD growth
    df['quarter'] = df['date'].dt.to_period('Q')
    quarterly = df.groupby('quarter').agg({
        'CVD': 'last',
        'VOLUME_DELTA': 'sum'
    }).reset_index()
    quarterly['CVD_Growth'] = quarterly['CVD'].pct_change() * 100

    # Plot
    quarters_str = quarterly['quarter'].astype(str)
    ax6.plot(range(len(quarterly)), quarterly['CVD_Growth'], 'o-', color='purple', linewidth=2)
    ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax6.fill_between(range(len(quarterly)), 0, quarterly['CVD_Growth'],
                     where=(quarterly['CVD_Growth'] > 0), alpha=0.3, color='green')
    ax6.fill_between(range(len(quarterly)), 0, quarterly['CVD_Growth'],
                     where=(quarterly['CVD_Growth'] < 0), alpha=0.3, color='red')

    ax6.set_title('Quarterly CVD Growth Rate (%)', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Quarter')
    ax6.set_ylabel('Growth Rate (%)')
    ax6.set_xticks(range(0, len(quarterly), 4))
    ax6.set_xticklabels([quarters_str.iloc[i] if i < len(quarters_str) else ''
                         for i in range(0, len(quarterly), 4)], rotation=45)
    ax6.grid(True, alpha=0.3)

    # 7. Volume Delta Distribution by Year
    ax7 = fig.add_subplot(gs[4, 1])

    # Create violin plot data
    violin_data = []
    violin_labels = []
    for year in sorted(df['year'].unique())[-5:]:  # Last 5 years
        year_data = df[df['year'] == year]['VOLUME_DELTA'].values / 1e9
        violin_data.append(year_data)
        violin_labels.append(str(year))

    parts = ax7.violinplot(violin_data, positions=range(len(violin_labels)),
                           showmeans=True, showmedians=True)
    ax7.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    ax7.set_title('Volume Delta Distribution (Last 5 Years)', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Year')
    ax7.set_ylabel('Daily Volume Delta (B)')
    ax7.set_xticks(range(len(violin_labels)))
    ax7.set_xticklabels(violin_labels)
    ax7.grid(True, alpha=0.3)

    # 8. Summary Statistics Table
    ax8 = fig.add_subplot(gs[5, :])
    ax8.axis('tight')
    ax8.axis('off')

    # Prepare table data
    table_data = []
    table_data.append(['Metric', 'Value'])
    table_data.append(['Total Buy Volume', f'{df["BUY_VOLUME"].sum()/1e12:.2f} Trillion'])
    table_data.append(['Total Sell Volume', f'{df["SELL_VOLUME"].sum()/1e12:.2f} Trillion'])
    table_data.append(['Overall Buy/Sell Ratio', f'{df["BUY_VOLUME"].sum()/df["SELL_VOLUME"].sum():.3f}'])
    table_data.append(['Current CVD', f'{df["CVD"].iloc[-1]/1e9:.2f} Billion'])
    table_data.append(['Average Daily Delta', f'{df["VOLUME_DELTA"].mean()/1e9:.3f} Billion'])
    table_data.append(['Best Year (Buy/Sell)', f'{yearly_df.loc[yearly_df["Buy_Sell_Ratio"].idxmax(), "Year"]:.0f} ({yearly_df["Buy_Sell_Ratio"].max():.3f})'])
    table_data.append(['Worst Year (Buy/Sell)', f'{yearly_df.loc[yearly_df["Buy_Sell_Ratio"].idxmin(), "Year"]:.0f} ({yearly_df["Buy_Sell_Ratio"].min():.3f})'])

    # Create table
    table = ax8.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style the header row
    for i in range(2):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax8.set_title('Summary Statistics (2016-2025)', fontsize=12, fontweight='bold', pad=20)

    # Overall title
    fig.suptitle('Volume Delta & CVD Analysis - VNINDEX (2016-2025)',
                fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()

    # Save figure
    output_path = 'outputs/cvd_yearly_visualization.png'
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"\n   Visualization saved to: {output_path}")

    # Save yearly statistics to CSV
    yearly_output = 'outputs/cvd_yearly_statistics.csv'
    yearly_df.to_csv(yearly_output, index=False)
    print(f"   Yearly statistics saved to: {yearly_output}")

    print("\n" + "="*80)
    print("YEARLY VISUALIZATION COMPLETE")
    print("="*80)

    return yearly_df

if __name__ == "__main__":
    yearly_stats = create_yearly_visualization()