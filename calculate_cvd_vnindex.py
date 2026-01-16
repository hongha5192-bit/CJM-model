"""
Calculate Cumulative Volume Delta (CVD) for VNINDEX using Intrabar Pressure Method
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def calculate_volume_delta_intrabar(df):
    """
    Calculate Volume Delta using the Intrabar Pressure Method

    Formula:
    - If (Close - Low) > (High - Close): Buying pressure dominates → positive delta
    - If (Close - Low) < (High - Close): Selling pressure dominates → negative delta
    - If equal: Use previous bar's direction
    """

    # Initialize arrays
    volume_delta = np.zeros(len(df))
    buy_volume = np.zeros(len(df))
    sell_volume = np.zeros(len(df))

    for i in range(len(df)):
        high = df['HIGHESTINDEX'].iloc[i]
        low = df['LOWESTINDEX'].iloc[i]
        close = df['CLOSEINDEX'].iloc[i]
        open_price = df['OPENINDEX'].iloc[i]
        volume = df['TOTALMATCHVOLUME'].iloc[i]

        # Skip if no volume
        if pd.isna(volume) or volume == 0:
            continue

        # Calculate ranges
        total_range = high - low

        if total_range > 0:
            # Intrabar pressure method
            lower_range = close - low  # Distance from low to close
            upper_range = high - close  # Distance from close to high

            if lower_range > upper_range:
                # Close is nearer to high = buying pressure
                buy_volume[i] = volume
                volume_delta[i] = volume
            elif lower_range < upper_range:
                # Close is nearer to low = selling pressure
                sell_volume[i] = volume
                volume_delta[i] = -volume
            else:
                # Equal distance - use bar polarity as tiebreaker
                if close > open_price:
                    buy_volume[i] = volume
                    volume_delta[i] = volume
                elif close < open_price:
                    sell_volume[i] = volume
                    volume_delta[i] = -volume
                else:
                    # If still equal, use previous bar direction
                    if i > 0:
                        if volume_delta[i-1] > 0:
                            buy_volume[i] = volume
                            volume_delta[i] = volume
                        else:
                            sell_volume[i] = volume
                            volume_delta[i] = -volume
        else:
            # No range (high = low) - use close vs open
            if close > open_price:
                buy_volume[i] = volume
                volume_delta[i] = volume
            elif close < open_price:
                sell_volume[i] = volume
                volume_delta[i] = -volume
            else:
                # Doji with no range - use previous direction
                if i > 0 and volume_delta[i-1] > 0:
                    buy_volume[i] = volume
                    volume_delta[i] = volume
                else:
                    sell_volume[i] = volume
                    volume_delta[i] = -volume

    return volume_delta, buy_volume, sell_volume

def calculate_trading_value(df):
    """
    Estimate trading value using volume and price
    Trading Value = Volume * Average Price
    Average Price = (High + Low + Close) / 3 (Typical Price)
    """
    typical_price = (df['HIGHESTINDEX'] + df['LOWESTINDEX'] + df['CLOSEINDEX']) / 3
    trading_value = df['TOTALMATCHVOLUME'] * typical_price

    # Convert to billions VND for readability
    trading_value_billions = trading_value / 1e9

    return trading_value, trading_value_billions

def main():
    """Main function to calculate and visualize CVD"""

    print("="*80)
    print("CUMULATIVE VOLUME DELTA (CVD) CALCULATION FOR VNINDEX")
    print("="*80)

    # 1. Load data
    print("\n1. Loading VNINDEX data...")
    df = pd.read_csv('data/vnindex_full.csv', parse_dates=['date'])
    print(f"   Loaded {len(df)} rows from 2016-05-27 to {df['date'].max()}")

    # Check for missing volume data
    missing_volume = df['TOTALMATCHVOLUME'].isna().sum()
    print(f"   Missing volume data: {missing_volume} rows")

    # 2. Calculate trading value
    print("\n2. Calculating Trading Value...")
    df['TRADING_VALUE'], df['TRADING_VALUE_BILLIONS'] = calculate_trading_value(df)

    # Display trading value statistics
    print(f"   Average Daily Trading Value: {df['TRADING_VALUE_BILLIONS'].mean():.2f} billion VND")
    print(f"   Median Daily Trading Value: {df['TRADING_VALUE_BILLIONS'].median():.2f} billion VND")
    print(f"   Max Daily Trading Value: {df['TRADING_VALUE_BILLIONS'].max():.2f} billion VND")
    print(f"   Min Daily Trading Value: {df['TRADING_VALUE_BILLIONS'].min():.2f} billion VND")

    # 3. Calculate Volume Delta using Intrabar Pressure Method
    print("\n3. Calculating Volume Delta (Intrabar Pressure Method)...")
    volume_delta, buy_volume, sell_volume = calculate_volume_delta_intrabar(df)

    df['VOLUME_DELTA'] = volume_delta
    df['BUY_VOLUME'] = buy_volume
    df['SELL_VOLUME'] = sell_volume

    # 4. Calculate Cumulative Volume Delta
    print("\n4. Calculating Cumulative Volume Delta (CVD)...")
    df['CVD'] = df['VOLUME_DELTA'].cumsum()

    # Calculate CVD smoothing (7-period RMA)
    df['CVD_SMOOTH'] = df['CVD'].ewm(alpha=1/7, adjust=False).mean()

    # Display statistics
    print(f"   Final CVD value: {df['CVD'].iloc[-1]:,.0f}")
    print(f"   CVD range: [{df['CVD'].min():,.0f}, {df['CVD'].max():,.0f}]")

    # Calculate buy/sell volume statistics
    total_buy = df['BUY_VOLUME'].sum()
    total_sell = df['SELL_VOLUME'].sum()
    print(f"\n   Total Buy Volume: {total_buy:,.0f}")
    print(f"   Total Sell Volume: {total_sell:,.0f}")
    print(f"   Net Volume Delta: {total_buy - total_sell:,.0f}")
    print(f"   Buy/Sell Ratio: {total_buy/total_sell:.3f}")

    # 5. Identify divergences
    print("\n5. Identifying Price/CVD Divergences...")

    # Calculate price changes
    df['PRICE_CHANGE'] = df['CLOSEINDEX'].diff()
    df['CVD_CHANGE'] = df['CVD'].diff()

    # Identify divergences (price and CVD move in opposite directions)
    df['DIVERGENCE'] = 0
    df.loc[(df['PRICE_CHANGE'] > 0) & (df['CVD_CHANGE'] < 0), 'DIVERGENCE'] = -1  # Bearish divergence
    df.loc[(df['PRICE_CHANGE'] < 0) & (df['CVD_CHANGE'] > 0), 'DIVERGENCE'] = 1   # Bullish divergence

    bullish_div = (df['DIVERGENCE'] == 1).sum()
    bearish_div = (df['DIVERGENCE'] == -1).sum()
    print(f"   Bullish divergences: {bullish_div}")
    print(f"   Bearish divergences: {bearish_div}")

    # 6. Create visualizations
    print("\n6. Creating visualizations...")

    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    # Plot 1: Price with volume
    ax1 = axes[0]
    ax1_vol = ax1.twinx()

    ax1.plot(df['date'], df['CLOSEINDEX'], 'b-', linewidth=1, label='Close Price')
    ax1_vol.bar(df['date'], df['TOTALMATCHVOLUME'], alpha=0.3, color='gray', label='Volume')

    ax1.set_ylabel('VNINDEX', color='b')
    ax1_vol.set_ylabel('Volume', color='gray')
    ax1.set_title('VNINDEX Price and Volume')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    ax1_vol.legend(loc='upper right')

    # Plot 2: Volume Delta
    ax2 = axes[1]
    colors = ['g' if x > 0 else 'r' for x in df['VOLUME_DELTA']]
    ax2.bar(df['date'], df['VOLUME_DELTA'], color=colors, alpha=0.6)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('Volume Delta')
    ax2.set_title('Daily Volume Delta (Green=Buy, Red=Sell)')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Cumulative Volume Delta
    ax3 = axes[2]
    ax3.plot(df['date'], df['CVD'], 'b-', linewidth=1.5, label='CVD')
    ax3.plot(df['date'], df['CVD_SMOOTH'], 'r--', linewidth=1, alpha=0.7, label='CVD Smooth (7-RMA)')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.fill_between(df['date'], 0, df['CVD'], where=(df['CVD'] > 0), alpha=0.3, color='green')
    ax3.fill_between(df['date'], 0, df['CVD'], where=(df['CVD'] < 0), alpha=0.3, color='red')
    ax3.set_ylabel('Cumulative Volume Delta')
    ax3.set_title('Cumulative Volume Delta (CVD)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Price with CVD overlay
    ax4 = axes[3]
    ax4_cvd = ax4.twinx()

    ax4.plot(df['date'], df['CLOSEINDEX'], 'b-', linewidth=1, label='VNINDEX')
    ax4_cvd.plot(df['date'], df['CVD'], 'g-', linewidth=1, alpha=0.7, label='CVD')

    # Mark divergences
    bullish_dates = df[df['DIVERGENCE'] == 1]['date']
    bearish_dates = df[df['DIVERGENCE'] == -1]['date']

    for date in bullish_dates[-20:]:  # Show last 20 divergences
        ax4.axvline(x=date, color='green', alpha=0.2, linestyle='--')

    for date in bearish_dates[-20:]:  # Show last 20 divergences
        ax4.axvline(x=date, color='red', alpha=0.2, linestyle='--')

    ax4.set_ylabel('VNINDEX', color='b')
    ax4_cvd.set_ylabel('CVD', color='g')
    ax4.set_xlabel('Date')
    ax4.set_title('Price vs CVD (Divergences marked)')
    ax4.legend(loc='upper left')
    ax4_cvd.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_path = 'outputs/cvd_analysis.png'
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"\n   Visualization saved to: {output_path}")

    # 7. Save results to CSV
    print("\n7. Saving results...")

    # Select columns to save
    save_columns = ['date', 'OPENINDEX', 'HIGHESTINDEX', 'LOWESTINDEX', 'CLOSEINDEX',
                    'TOTALMATCHVOLUME', 'TRADING_VALUE_BILLIONS',
                    'BUY_VOLUME', 'SELL_VOLUME', 'VOLUME_DELTA', 'CVD', 'CVD_SMOOTH', 'DIVERGENCE']

    output_df = df[save_columns].copy()
    output_csv = 'outputs/vnindex_cvd_analysis.csv'
    output_df.to_csv(output_csv, index=False)
    print(f"   Results saved to: {output_csv}")

    # 8. Recent analysis (last 30 days)
    print("\n8. Recent CVD Analysis (Last 30 days):")
    recent_df = df.tail(30)

    recent_buy = recent_df['BUY_VOLUME'].sum()
    recent_sell = recent_df['SELL_VOLUME'].sum()
    recent_net = recent_buy - recent_sell

    print(f"   Buy Volume: {recent_buy:,.0f}")
    print(f"   Sell Volume: {recent_sell:,.0f}")
    print(f"   Net Delta: {recent_net:,.0f}")
    print(f"   Buy/Sell Ratio: {recent_buy/recent_sell:.3f}")
    print(f"   CVD Change: {recent_df['CVD'].iloc[-1] - recent_df['CVD'].iloc[0]:,.0f}")

    # Check current market state
    current_cvd = df['CVD'].iloc[-1]
    cvd_ma20 = df['CVD'].rolling(20).mean().iloc[-1]

    if current_cvd > cvd_ma20:
        market_state = "BULLISH (CVD above 20-day average)"
    else:
        market_state = "BEARISH (CVD below 20-day average)"

    print(f"\n   Current Market State: {market_state}")
    print(f"   Current CVD: {current_cvd:,.0f}")
    print(f"   20-day CVD Average: {cvd_ma20:,.0f}")

    print("\n" + "="*80)
    print("CVD CALCULATION COMPLETE")
    print("="*80)

    return df

if __name__ == "__main__":
    df_with_cvd = main()