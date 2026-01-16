#!/usr/bin/env python3
"""
Visualize HMM regime labels over time
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import seaborn as sns

def visualize_hmm_regimes():
    """Create comprehensive visualizations of HMM regimes"""

    print("="*80)
    print("VISUALIZING HMM REGIME LABELS")
    print("="*80)

    # Load HMM states
    print("\nLoading HMM state assignments...")
    df_hmm = pd.read_csv('outputs/step1_params/hmm_states_K3.csv')
    df_hmm['date'] = pd.to_datetime(df_hmm['date'])

    # Load OHLC data
    print("Loading OHLC data...")
    df_ohlc = pd.read_csv('data/vnindex_full.csv')
    df_ohlc['date'] = pd.to_datetime(df_ohlc['date'])
    df_ohlc = df_ohlc.rename(columns={
        'CLOSEINDEX': 'close'
    })

    # Merge
    df = df_hmm.merge(df_ohlc[['date', 'close']], on='date', how='left')

    print(f"Data shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # State labels and colors
    state_labels = {
        0: "Bullish (State 0)",
        1: "Neutral (State 1)",
        2: "Bearish (State 2)"
    }

    state_colors = {
        0: '#2ecc71',  # Green
        1: '#f39c12',  # Orange
        2: '#e74c3c'   # Red
    }

    # Create output directory
    output_dir = Path('outputs/hmm_regime_visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")

    # ========================================================================
    # 1. Timeline with price and regime background
    # ========================================================================
    print("\nCreating timeline visualization...")

    fig, ax = plt.subplots(figsize=(16, 6))

    # Plot price
    ax.plot(df['date'], df['close'], color='black', linewidth=1.5, label='VNINDEX Close', zorder=3)

    # Add colored background for each regime
    current_state = df['hmm_state'].iloc[0]
    start_idx = 0

    for i in range(1, len(df)):
        if df['hmm_state'].iloc[i] != current_state or i == len(df) - 1:
            # End of current regime
            end_idx = i if i < len(df) - 1 else i

            ax.axvspan(df['date'].iloc[start_idx], df['date'].iloc[end_idx],
                      alpha=0.3, color=state_colors[current_state], zorder=1)

            # Update for next regime
            current_state = df['hmm_state'].iloc[i]
            start_idx = i

    # Formatting
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('VNINDEX Close Price', fontsize=12)
    ax.set_title('HMM Regime Classification Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')

    # Add custom legend for regimes
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=state_colors[i], alpha=0.3, label=state_labels[i])
                      for i in range(3)]
    legend_elements.insert(0, plt.Line2D([0], [0], color='black', linewidth=1.5, label='VNINDEX Close'))
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / 'hmm_regime_timeline.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # 2. Regime sequence plot (categorical)
    # ========================================================================
    print("Creating regime sequence plot...")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    # Top: Price
    ax1.plot(df['date'], df['close'], color='black', linewidth=1)
    ax1.set_ylabel('VNINDEX Close', fontsize=11)
    ax1.set_title('VNINDEX Price and HMM Regime Sequence', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Bottom: Regime as categorical
    for state in range(3):
        state_data = df[df['hmm_state'] == state]
        ax2.scatter(state_data['date'], state_data['hmm_state'],
                   color=state_colors[state], s=10, alpha=0.7, label=state_labels[state])

    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('Regime', fontsize=11)
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['Bullish (0)', 'Neutral (1)', 'Bearish (2)'])
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / 'hmm_regime_sequence.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # 3. Regime distribution by year
    # ========================================================================
    print("Creating regime distribution by year...")

    df['year'] = df['date'].dt.year

    # Calculate regime counts by year
    regime_by_year = df.groupby(['year', 'hmm_state']).size().unstack(fill_value=0)
    regime_pct_by_year = regime_by_year.div(regime_by_year.sum(axis=1), axis=0) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Stacked bar chart - counts
    regime_by_year.plot(kind='bar', stacked=True, ax=ax1,
                        color=[state_colors[i] for i in range(3)],
                        width=0.8)
    ax1.set_title('HMM Regime Distribution by Year (Counts)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Number of Days')
    ax1.legend([state_labels[i] for i in range(3)], loc='upper left')
    ax1.grid(axis='y', alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # Stacked bar chart - percentages
    regime_pct_by_year.plot(kind='bar', stacked=True, ax=ax2,
                            color=[state_colors[i] for i in range(3)],
                            width=0.8)
    ax2.set_title('HMM Regime Distribution by Year (Percentage)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Percentage (%)')
    ax2.legend([state_labels[i] for i in range(3)], loc='upper left')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 100)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / 'hmm_regime_by_year.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # 4. Regime duration analysis
    # ========================================================================
    print("Creating regime duration analysis...")

    # Calculate regime durations
    df['regime_change'] = (df['hmm_state'] != df['hmm_state'].shift(1)).astype(int)
    df['regime_group'] = df['regime_change'].cumsum()

    regime_durations = df.groupby(['regime_group', 'hmm_state']).size().reset_index(name='duration')

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for state in range(3):
        ax = axes[state]
        state_durations = regime_durations[regime_durations['hmm_state'] == state]['duration']

        ax.hist(state_durations, bins=30, color=state_colors[state], alpha=0.7, edgecolor='black')
        ax.axvline(state_durations.mean(), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {state_durations.mean():.1f} days')
        ax.axvline(state_durations.median(), color='blue', linestyle='--', linewidth=2,
                  label=f'Median: {state_durations.median():.1f} days')

        ax.set_title(f'{state_labels[state]}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Duration (days)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('HMM Regime Duration Distribution', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'hmm_regime_durations.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # 5. Regime transitions heatmap
    # ========================================================================
    print("Creating regime transitions heatmap...")

    # Calculate transition counts
    df['next_state'] = df['hmm_state'].shift(-1)
    transitions = df[df['regime_change'].shift(-1) == 1].groupby(['hmm_state', 'next_state']).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(transitions, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
                cbar_kws={'label': 'Number of Transitions'},
                xticklabels=[state_labels[i] for i in range(3)],
                yticklabels=[state_labels[i] for i in range(3)])

    ax.set_title('HMM Regime Transition Count Matrix', fontsize=12, fontweight='bold')
    ax.set_xlabel('To State')
    ax.set_ylabel('From State')

    plt.tight_layout()
    plt.savefig(output_dir / 'hmm_regime_transitions.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # 6. Summary statistics table
    # ========================================================================
    print("Creating summary statistics...")

    summary_stats = []
    for state in range(3):
        state_data = df[df['hmm_state'] == state]
        durations = regime_durations[regime_durations['hmm_state'] == state]['duration']

        summary_stats.append({
            'Regime': state_labels[state],
            'Total Days': len(state_data),
            'Percentage': f"{len(state_data)/len(df)*100:.1f}%",
            'Avg Duration': f"{durations.mean():.1f}",
            'Median Duration': f"{durations.median():.1f}",
            'Max Duration': f"{durations.max():.0f}",
            'Num Episodes': len(durations),
            'Avg Return': f"{state_data['ret'].mean()*100:.3f}%",
            'Std Return': f"{state_data['ret'].std()*100:.3f}%"
        })

    df_summary = pd.DataFrame(summary_stats)

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=df_summary.values,
                    colLabels=df_summary.columns,
                    cellLoc='center',
                    loc='center',
                    colColours=['#f0f0f0']*len(df_summary.columns))

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Color code regime names
    for i in range(3):
        table[(i+1, 0)].set_facecolor(state_colors[i])
        table[(i+1, 0)].set_text_props(weight='bold', color='white')

    plt.title('HMM Regime Summary Statistics', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_dir / 'hmm_regime_summary.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # Print summary
    # ========================================================================
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nAll visualizations saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  1. hmm_regime_timeline.png - Price chart with regime background")
    print("  2. hmm_regime_sequence.png - Price and regime sequence plot")
    print("  3. hmm_regime_by_year.png - Regime distribution by year")
    print("  4. hmm_regime_durations.png - Regime duration histograms")
    print("  5. hmm_regime_transitions.png - Transition count heatmap")
    print("  6. hmm_regime_summary.png - Summary statistics table")

    print("\n" + "="*80)
    print("REGIME STATISTICS SUMMARY")
    print("="*80)
    print(df_summary.to_string(index=False))
    print("\n" + "="*80)

if __name__ == "__main__":
    visualize_hmm_regimes()
