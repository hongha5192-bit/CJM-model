#!/usr/bin/env python3
"""
Visualize candlestick chart for one simulation
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from pathlib import Path
import mplfinance as mpf

def plot_candlestick_simple(df, output_path, title="Simulated Price Series"):
    """
    Create candlestick chart using matplotlib
    """
    fig, ax = plt.subplots(figsize=(20, 8))

    # Prepare colors
    colors = ['#2ecc71' if row['close'] >= row['open'] else '#e74c3c'
              for idx, row in df.iterrows()]

    # Plot candlesticks
    for idx, row in df.iterrows():
        x = idx
        open_price = row['open']
        high = row['high']
        low = row['low']
        close = row['close']

        color = '#2ecc71' if close >= open_price else '#e74c3c'

        # Draw high-low line (wick)
        ax.plot([x, x], [low, high], color='black', linewidth=0.5, alpha=0.8)

        # Draw open-close box (body)
        height = abs(close - open_price)
        bottom = min(open_price, close)

        if height < 0.001:  # Doji - very small body
            ax.plot([x-0.3, x+0.3], [open_price, open_price],
                   color='black', linewidth=1)
        else:
            rect = Rectangle((x-0.3, bottom), 0.6, height,
                           facecolor=color, edgecolor='black',
                           linewidth=0.5, alpha=0.8)
            ax.add_patch(rect)

    ax.set_xlabel('Time (days)', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add legend
    green_patch = mpatches.Patch(color='#2ecc71', label='Up Day (Close > Open)')
    red_patch = mpatches.Patch(color='#e74c3c', label='Down Day (Close < Open)')
    ax.legend(handles=[green_patch, red_patch], loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def plot_candlestick_with_regimes(df, output_path):
    """
    Create candlestick chart with regime background colors
    """
    fig, ax = plt.subplots(figsize=(20, 8))

    # State colors for background
    state_colors_bg = {
        0: '#2ecc71',  # Bullish - Green
        1: '#f39c12',  # Neutral - Orange
        2: '#e74c3c'   # Bearish - Red
    }

    state_labels = {
        0: "Bullish",
        1: "Neutral",
        2: "Bearish"
    }

    # Add regime background
    current_state = df['s_true'].iloc[0]
    start_idx = 0

    for i in range(1, len(df)):
        if df['s_true'].iloc[i] != current_state or i == len(df) - 1:
            end_idx = i if i < len(df) - 1 else i

            ax.axvspan(start_idx, end_idx,
                      alpha=0.15, color=state_colors_bg[current_state],
                      zorder=0)

            current_state = df['s_true'].iloc[i]
            start_idx = i

    # Plot candlesticks
    for idx, row in df.iterrows():
        x = idx
        open_price = row['open']
        high = row['high']
        low = row['low']
        close = row['close']

        color = '#2ecc71' if close >= open_price else '#e74c3c'

        # Draw high-low line (wick)
        ax.plot([x, x], [low, high], color='black', linewidth=0.5, alpha=0.8)

        # Draw open-close box (body)
        height = abs(close - open_price)
        bottom = min(open_price, close)

        if height < 0.001:  # Doji
            ax.plot([x-0.3, x+0.3], [open_price, open_price],
                   color='black', linewidth=1)
        else:
            rect = Rectangle((x-0.3, bottom), 0.6, height,
                           facecolor=color, edgecolor='black',
                           linewidth=0.5, alpha=0.9, zorder=2)
            ax.add_patch(rect)

    ax.set_xlabel('Time (days)', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.set_title('Simulated Price Series with HMM Regime Background',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, zorder=1)

    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', alpha=0.9, edgecolor='black', label='Up Day'),
        Patch(facecolor='#e74c3c', alpha=0.9, edgecolor='black', label='Down Day'),
        Patch(facecolor=state_colors_bg[0], alpha=0.15, label='Bullish Regime'),
        Patch(facecolor=state_colors_bg[1], alpha=0.15, label='Neutral Regime'),
        Patch(facecolor=state_colors_bg[2], alpha=0.15, label='Bearish Regime')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def plot_candlestick_subplots(df, output_path):
    """
    Create multi-panel view with candlestick, volume-like bars, and regime
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 12),
                                        sharex=True,
                                        gridspec_kw={'height_ratios': [3, 1, 1]})

    # ========================================================================
    # Panel 1: Candlestick Chart
    # ========================================================================
    for idx, row in df.iterrows():
        x = idx
        open_price = row['open']
        high = row['high']
        low = row['low']
        close = row['close']

        color = '#2ecc71' if close >= open_price else '#e74c3c'

        # Draw high-low line (wick)
        ax1.plot([x, x], [low, high], color='black', linewidth=0.5, alpha=0.8)

        # Draw open-close box (body)
        height = abs(close - open_price)
        bottom = min(open_price, close)

        if height < 0.001:  # Doji
            ax1.plot([x-0.3, x+0.3], [open_price, open_price],
                    color='black', linewidth=1)
        else:
            rect = Rectangle((x-0.3, bottom), 0.6, height,
                           facecolor=color, edgecolor='black',
                           linewidth=0.5, alpha=0.8)
            ax1.add_patch(rect)

    ax1.set_ylabel('Price', fontsize=11)
    ax1.set_title('Simulated Price Series - Candlestick View',
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # ========================================================================
    # Panel 2: Daily Returns
    # ========================================================================
    returns = df['returns'].values
    colors_ret = ['#2ecc71' if r >= 0 else '#e74c3c' for r in returns]

    ax2.bar(range(len(returns)), returns * 100, color=colors_ret,
           alpha=0.7, edgecolor='black', linewidth=0.3)
    ax2.axhline(y=0, color='black', linewidth=1)
    ax2.set_ylabel('Return (%)', fontsize=11)
    ax2.set_title('Daily Returns', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # ========================================================================
    # Panel 3: Regime States
    # ========================================================================
    state_colors = {0: '#2ecc71', 1: '#f39c12', 2: '#e74c3c'}
    state_labels = {0: "Bullish", 1: "Neutral", 2: "Bearish"}

    for state in [0, 1, 2]:
        state_data = df[df['s_true'] == state]
        ax3.scatter(state_data.index, state_data['s_true'],
                   color=state_colors[state], s=20, alpha=0.7,
                   label=state_labels[state])

    ax3.set_xlabel('Time (days)', fontsize=11)
    ax3.set_ylabel('Regime', fontsize=11)
    ax3.set_yticks([0, 1, 2])
    ax3.set_yticklabels(['Bullish', 'Neutral', 'Bearish'])
    ax3.set_title('HMM Regime States', fontsize=11, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def plot_monthly_view(df, output_path):
    """
    Create monthly candlestick aggregation
    """
    # Add month column
    df_copy = df.copy()
    df_copy['month'] = df_copy.index // 21  # Approximate 21 trading days per month

    # Aggregate to monthly OHLC
    monthly = df_copy.groupby('month').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        's_true': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
    }).reset_index()

    fig, ax = plt.subplots(figsize=(20, 8))

    # Plot monthly candlesticks
    for idx, row in monthly.iterrows():
        x = idx
        open_price = row['open']
        high = row['high']
        low = row['low']
        close = row['close']

        color = '#2ecc71' if close >= open_price else '#e74c3c'

        # Draw high-low line (wick)
        ax.plot([x, x], [low, high], color='black', linewidth=1.5, alpha=0.8)

        # Draw open-close box (body)
        height = abs(close - open_price)
        bottom = min(open_price, close)

        if height < 0.001:  # Doji
            ax.plot([x-0.3, x+0.3], [open_price, open_price],
                   color='black', linewidth=2)
        else:
            rect = Rectangle((x-0.3, bottom), 0.6, height,
                           facecolor=color, edgecolor='black',
                           linewidth=1, alpha=0.8)
            ax.add_patch(rect)

    ax.set_xlabel('Time (months)', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.set_title('Simulated Price Series - Monthly Aggregation',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add legend
    green_patch = mpatches.Patch(color='#2ecc71', label='Up Month')
    red_patch = mpatches.Patch(color='#e74c3c', label='Down Month')
    ax.legend(handles=[green_patch, red_patch], loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def main():
    print("="*80)
    print("VISUALIZING SIMULATION CANDLESTICK CHART")
    print("="*80)

    # Load one simulation
    sim_dir = Path('outputs/step2_sim/K3_price_based')
    sim_file = sim_dir / 'sim_0000.parquet'

    if not sim_file.exists():
        print(f"\nError: Simulation file not found: {sim_file}")
        print("Please run step2_simulate_price_based.py first")
        return

    print(f"\nLoading simulation: {sim_file}")
    df = pd.read_parquet(sim_file)

    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nPrice range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    print(f"Return mean: {df['returns'].mean()*100:.4f}%")
    print(f"Return std: {df['returns'].std()*100:.4f}%")

    # Count up/down days
    df['direction'] = df['close'] >= df['open']
    up_days = df['direction'].sum()
    down_days = len(df) - up_days
    print(f"\nUp days: {up_days} ({up_days/len(df)*100:.1f}%)")
    print(f"Down days: {down_days} ({down_days/len(df)*100:.1f}%)")

    # State distribution
    print("\nState distribution:")
    for state in [0, 1, 2]:
        count = (df['s_true'] == state).sum()
        pct = count / len(df) * 100
        state_names = {0: 'Bullish', 1: 'Neutral', 2: 'Bearish'}
        print(f"  {state_names[state]}: {count} days ({pct:.1f}%)")

    # Create output directory
    output_dir = Path('outputs/simulation_visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create visualizations
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)

    print("\n1. Simple candlestick chart...")
    plot_candlestick_simple(df, output_dir / 'candlestick_simple.png')

    print("\n2. Candlestick with regime backgrounds...")
    plot_candlestick_with_regimes(df, output_dir / 'candlestick_with_regimes.png')

    print("\n3. Multi-panel view...")
    plot_candlestick_subplots(df, output_dir / 'candlestick_subplots.png')

    print("\n4. Monthly aggregation...")
    plot_monthly_view(df, output_dir / 'candlestick_monthly.png')

    # Create zoomed views for different time periods
    print("\n5. Creating zoomed views...")

    # First 100 days
    df_zoom1 = df.iloc[:100].copy()
    plot_candlestick_simple(df_zoom1,
                           output_dir / 'candlestick_zoom_days_0_100.png',
                           title="Simulated Price Series - Days 0-100")

    # Middle 100 days
    mid_start = len(df) // 2 - 50
    mid_end = mid_start + 100
    df_zoom2 = df.iloc[mid_start:mid_end].copy()
    df_zoom2.index = range(len(df_zoom2))
    plot_candlestick_simple(df_zoom2,
                           output_dir / f'candlestick_zoom_days_{mid_start}_{mid_end}.png',
                           title=f"Simulated Price Series - Days {mid_start}-{mid_end}")

    # Last 100 days
    df_zoom3 = df.iloc[-100:].copy()
    df_zoom3.index = range(len(df_zoom3))
    plot_candlestick_simple(df_zoom3,
                           output_dir / f'candlestick_zoom_days_{len(df)-100}_{len(df)}.png',
                           title=f"Simulated Price Series - Days {len(df)-100}-{len(df)}")

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nAll visualizations saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  1. candlestick_simple.png - Basic candlestick chart (full series)")
    print("  2. candlestick_with_regimes.png - With regime background colors")
    print("  3. candlestick_subplots.png - Multi-panel view (price + returns + regimes)")
    print("  4. candlestick_monthly.png - Monthly aggregated view")
    print("  5. candlestick_zoom_days_0_100.png - Zoomed: first 100 days")
    print(f"  6. candlestick_zoom_days_{mid_start}_{mid_end}.png - Zoomed: middle 100 days")
    print(f"  7. candlestick_zoom_days_{len(df)-100}_{len(df)}.png - Zoomed: last 100 days")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
