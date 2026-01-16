"""
Calculate Rolling CVD with 20-bar window for better regime detection
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def calculate_rolling_cvd(window=20):
    """
    Calculate rolling CVD instead of cumulative from beginning

    Parameters:
    window: Number of bars for rolling sum (default 20)
    """

    print("="*80)
    print(f"ROLLING CVD CALCULATION (Window = {window} bars)")
    print("="*80)

    # Load the existing CVD data
    print(f"\n1. Loading data and calculating {window}-bar rolling CVD...")
    df = pd.read_csv('outputs/vnindex_cvd_analysis.csv', parse_dates=['date'])

    # Calculate rolling CVD (sum of volume delta over window)
    df[f'CVD_{window}'] = df['VOLUME_DELTA'].rolling(window=window, min_periods=1).sum()

    # Also calculate different windows for comparison
    windows = [10, 20, 50, 100]
    for w in windows:
        df[f'CVD_{w}'] = df['VOLUME_DELTA'].rolling(window=w, min_periods=1).sum()

    # Calculate rolling CVD momentum (rate of change)
    df[f'CVD_{window}_momentum'] = df[f'CVD_{window}'].diff()

    # Calculate rolling Buy/Sell ratio over window
    df[f'BuySell_Ratio_{window}'] = (
        df['BUY_VOLUME'].rolling(window=window, min_periods=1).sum() /
        df['SELL_VOLUME'].rolling(window=window, min_periods=1).sum()
    )

    print(f"   Calculated rolling CVD for windows: {windows}")

    # Display statistics
    print(f"\n2. Rolling CVD Statistics (Last 100 days):")
    print("-" * 60)

    recent_df = df.tail(100)
    for w in windows:
        cvd_col = f'CVD_{w}'
        mean_val = recent_df[cvd_col].mean() / 1e9
        std_val = recent_df[cvd_col].std() / 1e9
        current_val = df[cvd_col].iloc[-1] / 1e9

        print(f"\n{w}-bar Rolling CVD:")
        print(f"   Current: {current_val:.3f} Billion")
        print(f"   Mean: {mean_val:.3f} Billion")
        print(f"   Std Dev: {std_val:.3f} Billion")
        print(f"   Z-score: {(current_val - mean_val) / std_val if std_val > 0 else 0:.2f}")

    # Create visualization
    print("\n3. Creating visualizations...")

    fig, axes = plt.subplots(4, 2, figsize=(14, 14))

    # 1. Compare different rolling windows
    ax1 = axes[0, 0]
    for w in [10, 20, 50]:
        ax1.plot(df['date'].tail(500), df[f'CVD_{w}'].tail(500)/1e9,
                label=f'{w}-bar CVD', linewidth=1.5, alpha=0.8)

    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_title('Rolling CVD Comparison (Last 500 days)', fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Rolling CVD (Billions)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 20-bar CVD with signals
    ax2 = axes[0, 1]

    # Plot 20-bar CVD
    recent_dates = df['date'].tail(250)
    recent_cvd = df['CVD_20'].tail(250) / 1e9

    ax2.plot(recent_dates, recent_cvd, 'b-', linewidth=1.5, label='20-bar CVD')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    # Add colored regions
    ax2.fill_between(recent_dates, 0, recent_cvd,
                     where=(recent_cvd > 0), alpha=0.3, color='green', label='Net Buying')
    ax2.fill_between(recent_dates, 0, recent_cvd,
                     where=(recent_cvd < 0), alpha=0.3, color='red', label='Net Selling')

    ax2.set_title('20-bar Rolling CVD with Buy/Sell Zones', fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('20-bar CVD (Billions)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Rolling CVD vs Price
    ax3 = axes[1, 0]
    ax3_price = ax3.twinx()

    # Plot both on same chart
    ax3.plot(df['date'].tail(250), df['CVD_20'].tail(250)/1e9,
             'g-', linewidth=1.5, alpha=0.8, label='20-bar CVD')
    ax3_price.plot(df['date'].tail(250), df['CLOSEINDEX'].tail(250),
                   'b-', linewidth=1.5, alpha=0.8, label='VNINDEX')

    ax3.set_xlabel('Date')
    ax3.set_ylabel('20-bar CVD (Billions)', color='g')
    ax3_price.set_ylabel('VNINDEX', color='b')
    ax3.set_title('20-bar CVD vs Price (Last 250 days)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left')
    ax3_price.legend(loc='upper right')

    # 4. CVD Momentum (Rate of Change)
    ax4 = axes[1, 1]

    momentum = df['CVD_20_momentum'].tail(250) / 1e9
    ax4.bar(df['date'].tail(250), momentum,
            color=['green' if x > 0 else 'red' for x in momentum],
            alpha=0.6)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)

    ax4.set_title('20-bar CVD Momentum (Daily Change)', fontweight='bold')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('CVD Change (Billions)')
    ax4.grid(True, alpha=0.3)

    # 5. Rolling Buy/Sell Ratio
    ax5 = axes[2, 0]

    ratio_20 = df['BuySell_Ratio_20'].tail(250)
    ax5.plot(df['date'].tail(250), ratio_20, 'purple', linewidth=1.5)
    ax5.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    ax5.axhline(y=1.2, color='green', linestyle=':', alpha=0.3, label='Strong Buy')
    ax5.axhline(y=0.8, color='red', linestyle=':', alpha=0.3, label='Strong Sell')

    ax5.fill_between(df['date'].tail(250), 1, ratio_20,
                     where=(ratio_20 > 1), alpha=0.2, color='green')
    ax5.fill_between(df['date'].tail(250), 1, ratio_20,
                     where=(ratio_20 < 1), alpha=0.2, color='red')

    ax5.set_title('20-bar Buy/Sell Volume Ratio', fontweight='bold')
    ax5.set_xlabel('Date')
    ax5.set_ylabel('Buy/Sell Ratio')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. CVD Z-Score (Normalized)
    ax6 = axes[2, 1]

    # Calculate rolling z-score
    cvd_mean = df['CVD_20'].rolling(100).mean()
    cvd_std = df['CVD_20'].rolling(100).std()
    cvd_zscore = (df['CVD_20'] - cvd_mean) / cvd_std

    ax6.plot(df['date'].tail(250), cvd_zscore.tail(250), 'orange', linewidth=1.5)
    ax6.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax6.axhline(y=2, color='red', linestyle='--', alpha=0.3, label='Overbought')
    ax6.axhline(y=-2, color='green', linestyle='--', alpha=0.3, label='Oversold')

    ax6.fill_between(df['date'].tail(250), -2, 2, alpha=0.1, color='gray')

    ax6.set_title('20-bar CVD Z-Score (100-day normalized)', fontweight='bold')
    ax6.set_xlabel('Date')
    ax6.set_ylabel('Z-Score')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # 7. Histogram of 20-bar CVD values
    ax7 = axes[3, 0]

    cvd_values = df['CVD_20'].dropna() / 1e9
    ax7.hist(cvd_values, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax7.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax7.axvline(x=cvd_values.mean(), color='orange', linestyle='-',
                alpha=0.7, label=f'Mean: {cvd_values.mean():.2f}B')

    current_val = df['CVD_20'].iloc[-1] / 1e9
    ax7.axvline(x=current_val, color='green', linestyle='-',
                alpha=0.7, linewidth=2, label=f'Current: {current_val:.2f}B')

    ax7.set_title('Distribution of 20-bar CVD Values', fontweight='bold')
    ax7.set_xlabel('20-bar CVD (Billions)')
    ax7.set_ylabel('Frequency')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 8. Comparison: Full CVD vs Rolling CVD
    ax8 = axes[3, 1]

    # Normalize both for comparison
    full_cvd_norm = (df['CVD'] - df['CVD'].mean()) / df['CVD'].std()
    rolling_cvd_norm = (df['CVD_20'] - df['CVD_20'].mean()) / df['CVD_20'].std()

    ax8.plot(df['date'].tail(500), full_cvd_norm.tail(500),
             'blue', linewidth=1, alpha=0.6, label='Full CVD (normalized)')
    ax8.plot(df['date'].tail(500), rolling_cvd_norm.tail(500),
             'red', linewidth=1.5, alpha=0.8, label='20-bar CVD (normalized)')

    ax8.set_title('Full CVD vs 20-bar Rolling CVD (Normalized)', fontweight='bold')
    ax8.set_xlabel('Date')
    ax8.set_ylabel('Normalized Value')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    plt.suptitle('Rolling CVD Analysis - VNINDEX', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save figure
    output_path = 'outputs/rolling_cvd_analysis.png'
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"   Visualization saved to: {output_path}")

    # Save data with rolling CVD
    print("\n4. Saving results...")

    # Select columns to save
    save_columns = ['date', 'CLOSEINDEX', 'TOTALMATCHVOLUME',
                    'VOLUME_DELTA', 'CVD', 'CVD_10', 'CVD_20', 'CVD_50', 'CVD_100',
                    'CVD_20_momentum', 'BuySell_Ratio_20']

    output_df = df[save_columns].copy()
    output_csv = 'outputs/vnindex_rolling_cvd.csv'
    output_df.to_csv(output_csv, index=False)
    print(f"   Results saved to: {output_csv}")

    # Print current signals
    print("\n5. Current Trading Signals (20-bar CVD):")
    print("-" * 60)

    current_cvd = df['CVD_20'].iloc[-1] / 1e9
    current_ratio = df['BuySell_Ratio_20'].iloc[-1]
    current_momentum = df['CVD_20_momentum'].iloc[-1] / 1e9
    current_zscore = cvd_zscore.iloc[-1]

    print(f"   20-bar CVD: {current_cvd:.3f} Billion")
    print(f"   Buy/Sell Ratio: {current_ratio:.3f}")
    print(f"   CVD Momentum: {current_momentum:.3f} Billion/day")
    print(f"   Z-Score: {current_zscore:.2f}")

    # Generate signal
    if current_cvd > 0 and current_momentum > 0 and current_ratio > 1.1:
        signal = "STRONG BUY - Accumulation phase"
    elif current_cvd > 0 and current_ratio > 1:
        signal = "BUY - Moderate buying pressure"
    elif current_cvd < 0 and current_momentum < 0 and current_ratio < 0.9:
        signal = "STRONG SELL - Distribution phase"
    elif current_cvd < 0 and current_ratio < 1:
        signal = "SELL - Moderate selling pressure"
    else:
        signal = "NEUTRAL - Mixed signals"

    print(f"\n   Signal: {signal}")

    print("\n" + "="*80)
    print("ROLLING CVD CALCULATION COMPLETE")
    print("="*80)

    return df

if __name__ == "__main__":
    df_rolling = calculate_rolling_cvd(window=20)