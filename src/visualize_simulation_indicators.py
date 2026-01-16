#!/usr/bin/env python3
"""
Visualize price chart with technical indicators for simulation
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

def plot_price_with_indicators(df, output_path):
    """
    Create 4-panel chart: Price + URSI + ADX + BBWP
    """
    fig, axes = plt.subplots(4, 1, figsize=(20, 14), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1, 1, 1]})

    # ========================================================================
    # Panel 1: Candlestick Chart
    # ========================================================================
    ax1 = axes[0]

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

    ax1.set_ylabel('Price', fontsize=12, fontweight='bold')
    ax1.set_title('Simulated Price Series with Technical Indicators',
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # ========================================================================
    # Panel 2: URSI
    # ========================================================================
    ax2 = axes[1]

    ax2.plot(df.index, df['URSI'], color='#3498db', linewidth=1.5, label='URSI')
    ax2.axhline(y=70, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Overbought (70)')
    ax2.axhline(y=30, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Oversold (30)')
    ax2.axhline(y=50, color='gray', linestyle=':', linewidth=1, alpha=0.3)

    ax2.fill_between(df.index, 70, 100, color='red', alpha=0.1)
    ax2.fill_between(df.index, 0, 30, color='green', alpha=0.1)

    ax2.set_ylabel('URSI', fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ========================================================================
    # Panel 3: ADX
    # ========================================================================
    ax3 = axes[2]

    ax3.plot(df.index, df['ADX'], color='#9b59b6', linewidth=1.5, label='ADX')
    ax3.axhline(y=25, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Trend Threshold (25)')
    ax3.fill_between(df.index, 25, df['ADX'].max(), where=(df['ADX'] >= 25),
                     color='orange', alpha=0.1, interpolate=True)

    ax3.set_ylabel('ADX', fontsize=11, fontweight='bold')
    ax3.set_ylim(0, df['ADX'].max() * 1.1)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)

    # ========================================================================
    # Panel 4: BBWP
    # ========================================================================
    ax4 = axes[3]

    ax4.plot(df.index, df['BBWP'], color='#e67e22', linewidth=1.5, label='BBWP')
    ax4.axhline(y=80, color='red', linestyle='--', linewidth=1, alpha=0.5, label='High Volatility (80)')
    ax4.axhline(y=20, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Low Volatility (20)')
    ax4.axhline(y=50, color='gray', linestyle=':', linewidth=1, alpha=0.3)

    ax4.fill_between(df.index, 80, 100, color='red', alpha=0.1)
    ax4.fill_between(df.index, 0, 20, color='green', alpha=0.1)

    ax4.set_ylabel('BBWP', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Time (days)', fontsize=12, fontweight='bold')
    ax4.set_ylim(0, 100)
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def plot_price_with_indicators_regimes(df, output_path):
    """
    Create 4-panel chart with regime backgrounds
    """
    fig, axes = plt.subplots(4, 1, figsize=(20, 14), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1, 1, 1]})

    # State colors for background
    state_colors_bg = {
        0: '#2ecc71',  # Bullish - Green
        1: '#f39c12',  # Neutral - Orange
        2: '#e74c3c'   # Bearish - Red
    }

    # Add regime background to all panels
    for ax in axes:
        current_state = df['s_true'].iloc[0]
        start_idx = 0

        for i in range(1, len(df)):
            if df['s_true'].iloc[i] != current_state or i == len(df) - 1:
                end_idx = i if i < len(df) - 1 else i

                ax.axvspan(start_idx, end_idx,
                          alpha=0.1, color=state_colors_bg[current_state],
                          zorder=0)

                current_state = df['s_true'].iloc[i]
                start_idx = i

    # ========================================================================
    # Panel 1: Candlestick Chart
    # ========================================================================
    ax1 = axes[0]

    for idx, row in df.iterrows():
        x = idx
        open_price = row['open']
        high = row['high']
        low = row['low']
        close = row['close']

        color = '#2ecc71' if close >= open_price else '#e74c3c'

        # Draw high-low line (wick)
        ax1.plot([x, x], [low, high], color='black', linewidth=0.5, alpha=0.8, zorder=2)

        # Draw open-close box (body)
        height = abs(close - open_price)
        bottom = min(open_price, close)

        if height < 0.001:  # Doji
            ax1.plot([x-0.3, x+0.3], [open_price, open_price],
                    color='black', linewidth=1, zorder=2)
        else:
            rect = Rectangle((x-0.3, bottom), 0.6, height,
                           facecolor=color, edgecolor='black',
                           linewidth=0.5, alpha=0.9, zorder=2)
            ax1.add_patch(rect)

    ax1.set_ylabel('Price', fontsize=12, fontweight='bold')
    ax1.set_title('Simulated Price Series with Technical Indicators and Regime Background',
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, zorder=1)

    # ========================================================================
    # Panel 2: URSI
    # ========================================================================
    ax2 = axes[1]

    ax2.plot(df.index, df['URSI'], color='#3498db', linewidth=1.5, label='URSI', zorder=2)
    ax2.axhline(y=70, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Overbought (70)', zorder=1)
    ax2.axhline(y=30, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Oversold (30)', zorder=1)
    ax2.axhline(y=50, color='gray', linestyle=':', linewidth=1, alpha=0.3, zorder=1)

    ax2.set_ylabel('URSI', fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, zorder=1)

    # ========================================================================
    # Panel 3: ADX
    # ========================================================================
    ax3 = axes[2]

    ax3.plot(df.index, df['ADX'], color='#9b59b6', linewidth=1.5, label='ADX', zorder=2)
    ax3.axhline(y=25, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Trend Threshold (25)', zorder=1)

    ax3.set_ylabel('ADX', fontsize=11, fontweight='bold')
    ax3.set_ylim(0, df['ADX'].max() * 1.1)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3, zorder=1)

    # ========================================================================
    # Panel 4: BBWP
    # ========================================================================
    ax4 = axes[3]

    ax4.plot(df.index, df['BBWP'], color='#e67e22', linewidth=1.5, label='BBWP', zorder=2)
    ax4.axhline(y=80, color='red', linestyle='--', linewidth=1, alpha=0.5, label='High Volatility (80)', zorder=1)
    ax4.axhline(y=20, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Low Volatility (20)', zorder=1)
    ax4.axhline(y=50, color='gray', linestyle=':', linewidth=1, alpha=0.3, zorder=1)

    ax4.set_ylabel('BBWP', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Time (days)', fontsize=12, fontweight='bold')
    ax4.set_ylim(0, 100)
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3, zorder=1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def plot_indicators_statistics(df, output_path):
    """
    Show indicator statistics by regime
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))

    indicators = ['URSI', 'ADX', 'BBWP']
    state_names = {0: 'Bullish', 1: 'Neutral', 2: 'Bearish'}
    state_colors = {0: '#2ecc71', 1: '#f39c12', 2: '#e74c3c'}

    # For each indicator
    for i, indicator in enumerate(indicators):
        # Distribution by regime
        ax = axes[i, 0]
        for state in [0, 1, 2]:
            state_data = df[df['s_true'] == state][indicator]
            ax.hist(state_data, bins=30, alpha=0.6,
                   color=state_colors[state], label=state_names[state],
                   edgecolor='black', linewidth=0.5)
        ax.set_xlabel(indicator, fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{indicator} Distribution by Regime', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Box plot by regime
        ax = axes[i, 1]
        data_by_state = [df[df['s_true'] == state][indicator].values for state in [0, 1, 2]]
        bp = ax.boxplot(data_by_state, labels=[state_names[s] for s in [0, 1, 2]],
                       patch_artist=True, widths=0.6)
        for patch, state in zip(bp['boxes'], [0, 1, 2]):
            patch.set_facecolor(state_colors[state])
            patch.set_alpha(0.7)
        ax.set_ylabel(indicator, fontsize=10)
        ax.set_title(f'{indicator} by Regime (Boxplot)', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Time series
        ax = axes[i, 2]
        ax.plot(df.index, df[indicator], color='steelblue', linewidth=1, alpha=0.8)
        ax.set_xlabel('Time (days)', fontsize=10)
        ax.set_ylabel(indicator, fontsize=10)
        ax.set_title(f'{indicator} Time Series', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Technical Indicators Analysis', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def main():
    print("="*80)
    print("VISUALIZING TECHNICAL INDICATORS")
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

    # Print indicator statistics
    print("\n" + "="*80)
    print("INDICATOR STATISTICS")
    print("="*80)

    for indicator in ['URSI', 'ADX', 'BBWP']:
        print(f"\n{indicator}:")
        print(f"  Overall: {df[indicator].mean():.2f} ± {df[indicator].std():.2f}")
        print(f"  Range: [{df[indicator].min():.2f}, {df[indicator].max():.2f}]")

        print(f"  By regime:")
        for state in [0, 1, 2]:
            state_data = df[df['s_true'] == state][indicator]
            state_names = {0: 'Bullish', 1: 'Neutral', 2: 'Bearish'}
            print(f"    {state_names[state]}: {state_data.mean():.2f} ± {state_data.std():.2f}")

    # Create output directory
    output_dir = Path('outputs/simulation_visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create visualizations
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)

    print("\n1. Price with indicators...")
    plot_price_with_indicators(df, output_dir / 'indicators_price_chart.png')

    print("\n2. Price with indicators and regime backgrounds...")
    plot_price_with_indicators_regimes(df, output_dir / 'indicators_price_chart_regimes.png')

    print("\n3. Indicator statistics...")
    plot_indicators_statistics(df, output_dir / 'indicators_statistics.png')

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nAll visualizations saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  1. indicators_price_chart.png - Price with URSI, ADX, BBWP")
    print("  2. indicators_price_chart_regimes.png - With regime backgrounds")
    print("  3. indicators_statistics.png - Indicator distributions and statistics")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
