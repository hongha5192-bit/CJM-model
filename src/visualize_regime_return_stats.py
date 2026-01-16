#!/usr/bin/env python3
"""
Visualize mean return and std for each regime: HMM vs CJM
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def visualize_regime_return_stats():
    """Create comprehensive visualizations of regime return statistics"""

    print("="*80)
    print("VISUALIZING REGIME RETURN STATISTICS: HMM vs CJM")
    print("="*80)

    # Load data
    print("\nLoading data...")
    df_hmm = pd.read_csv('outputs/step1_params/hmm_states_K3.csv')
    df_hmm['date'] = pd.to_datetime(df_hmm['date'])

    df_cjm = pd.read_csv('outputs/step4_apply/regimes_daily_K3_relabeled.csv')
    df_cjm['date'] = pd.to_datetime(df_cjm['date'])

    # Merge
    df = df_hmm.merge(df_cjm[['date', 'regime_relabeled']], on='date', how='inner')
    df = df.rename(columns={'regime_relabeled': 'cjm_state'})

    print(f"Data shape: {df.shape}")

    # State labels and colors
    state_labels = ['Bullish', 'Neutral', 'Bearish']
    state_colors = {
        0: '#2ecc71',
        1: '#f39c12',
        2: '#e74c3c'
    }

    # Calculate statistics
    hmm_stats = []
    cjm_stats = []

    for state in range(3):
        # HMM
        hmm_data = df[df['hmm_state'] == state]['ret']
        hmm_stats.append({
            'regime': state_labels[state],
            'mean': hmm_data.mean() * 100,  # Convert to percentage
            'std': hmm_data.std() * 100,
            'count': len(hmm_data),
            'sharpe': hmm_data.mean() / hmm_data.std() if hmm_data.std() > 0 else 0
        })

        # CJM
        cjm_data = df[df['cjm_state'] == state]['ret']
        cjm_stats.append({
            'regime': state_labels[state],
            'mean': cjm_data.mean() * 100,
            'std': cjm_data.std() * 100,
            'count': len(cjm_data),
            'sharpe': cjm_data.mean() / cjm_data.std() if cjm_data.std() > 0 else 0
        })

    df_hmm_stats = pd.DataFrame(hmm_stats)
    df_cjm_stats = pd.DataFrame(cjm_stats)

    print("\nHMM Statistics:")
    print(df_hmm_stats)
    print("\nCJM Statistics:")
    print(df_cjm_stats)

    # Create output directory
    output_dir = Path('outputs/regime_return_stats')
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # 1. Combined bar chart - Mean Return
    # ========================================================================
    print("\nCreating combined visualizations...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(3)
    width = 0.35

    # Mean Return
    bars1 = ax1.bar(x - width/2, df_hmm_stats['mean'], width,
                    label='HMM', color='steelblue', alpha=0.8, edgecolor='black')
    bars2 = ax1.bar(x + width/2, df_cjm_stats['mean'], width,
                    label='CJM', color='coral', alpha=0.8, edgecolor='black')

    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_ylabel('Mean Daily Return (%)', fontsize=12)
    ax1.set_title('Mean Daily Return by Regime', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(state_labels)
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}%',
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=9, fontweight='bold')

    # Std Return
    bars3 = ax2.bar(x - width/2, df_hmm_stats['std'], width,
                    label='HMM', color='steelblue', alpha=0.8, edgecolor='black')
    bars4 = ax2.bar(x + width/2, df_cjm_stats['std'], width,
                    label='CJM', color='coral', alpha=0.8, edgecolor='black')

    ax2.set_ylabel('Std Dev of Daily Return (%)', fontsize=12)
    ax2.set_title('Return Volatility by Regime', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(state_labels)
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'mean_std_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # 2. Risk-Return Scatter Plot
    # ========================================================================
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot HMM
    for i, state in enumerate(state_labels):
        ax.scatter(df_hmm_stats.loc[i, 'std'], df_hmm_stats.loc[i, 'mean'],
                  s=300, color=state_colors[i], marker='o', alpha=0.7,
                  edgecolor='black', linewidth=2, label=f'HMM {state}')

    # Plot CJM
    for i, state in enumerate(state_labels):
        ax.scatter(df_cjm_stats.loc[i, 'std'], df_cjm_stats.loc[i, 'mean'],
                  s=300, color=state_colors[i], marker='s', alpha=0.7,
                  edgecolor='black', linewidth=2, label=f'CJM {state}')

    # Add lines connecting HMM and CJM for same regime
    for i in range(3):
        ax.plot([df_hmm_stats.loc[i, 'std'], df_cjm_stats.loc[i, 'std']],
               [df_hmm_stats.loc[i, 'mean'], df_cjm_stats.loc[i, 'mean']],
               'k--', alpha=0.3, linewidth=1)

    # Add labels
    for i, state in enumerate(state_labels):
        # HMM label
        ax.annotate(f'HMM\n{state}',
                   xy=(df_hmm_stats.loc[i, 'std'], df_hmm_stats.loc[i, 'mean']),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        # CJM label
        ax.annotate(f'CJM\n{state}',
                   xy=(df_cjm_stats.loc[i, 'std'], df_cjm_stats.loc[i, 'mean']),
                   xytext=(-10, -20), textcoords='offset points',
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

    ax.set_xlabel('Std Dev of Daily Return (%)', fontsize=12)
    ax.set_ylabel('Mean Daily Return (%)', fontsize=12)
    ax.set_title('Risk-Return Profile by Regime: HMM vs CJM\n(Circle=HMM, Square=CJM)',
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9, ncol=2)

    plt.tight_layout()
    plt.savefig(output_dir / 'risk_return_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # 3. Sharpe Ratio Comparison
    # ========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, df_hmm_stats['sharpe'], width,
                   label='HMM', color='steelblue', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, df_cjm_stats['sharpe'], width,
                   label='CJM', color='coral', alpha=0.8, edgecolor='black')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Sharpe Ratio (Daily)', fontsize=12)
    ax.set_title('Risk-Adjusted Return by Regime (Sharpe Ratio)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(state_labels)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom' if height >= 0 else 'top',
                   fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'sharpe_ratio_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # 4. Comprehensive Summary Plot (4 panels)
    # ========================================================================
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Panel 1: Mean Return
    ax1 = fig.add_subplot(gs[0, 0])
    bars1 = ax1.bar(x - width/2, df_hmm_stats['mean'], width,
                    label='HMM', color='steelblue', alpha=0.8, edgecolor='black')
    bars2 = ax1.bar(x + width/2, df_cjm_stats['mean'], width,
                    label='CJM', color='coral', alpha=0.8, edgecolor='black')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_ylabel('Mean Daily Return (%)', fontsize=11)
    ax1.set_title('Mean Return', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(state_labels)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}%',
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=8)

    # Panel 2: Std Dev
    ax2 = fig.add_subplot(gs[0, 1])
    bars3 = ax2.bar(x - width/2, df_hmm_stats['std'], width,
                    label='HMM', color='steelblue', alpha=0.8, edgecolor='black')
    bars4 = ax2.bar(x + width/2, df_cjm_stats['std'], width,
                    label='CJM', color='coral', alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Std Dev (%)', fontsize=11)
    ax2.set_title('Return Volatility', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(state_labels)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}%',
                    ha='center', va='bottom', fontsize=8)

    # Panel 3: Sharpe Ratio
    ax3 = fig.add_subplot(gs[1, 0])
    bars5 = ax3.bar(x - width/2, df_hmm_stats['sharpe'], width,
                    label='HMM', color='steelblue', alpha=0.8, edgecolor='black')
    bars6 = ax3.bar(x + width/2, df_cjm_stats['sharpe'], width,
                    label='CJM', color='coral', alpha=0.8, edgecolor='black')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_ylabel('Sharpe Ratio', fontsize=11)
    ax3.set_title('Risk-Adjusted Return', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(state_labels)
    ax3.legend(fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    for bars in [bars5, bars6]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=8)

    # Panel 4: Sample Size
    ax4 = fig.add_subplot(gs[1, 1])
    bars7 = ax4.bar(x - width/2, df_hmm_stats['count'], width,
                    label='HMM', color='steelblue', alpha=0.8, edgecolor='black')
    bars8 = ax4.bar(x + width/2, df_cjm_stats['count'], width,
                    label='CJM', color='coral', alpha=0.8, edgecolor='black')
    ax4.set_ylabel('Number of Days', fontsize=11)
    ax4.set_title('Sample Size', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(state_labels)
    ax4.legend(fontsize=10)
    ax4.grid(axis='y', alpha=0.3)
    for bars in [bars7, bars8]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=8)

    plt.suptitle('Regime Statistics Comparison: HMM vs CJM', fontsize=14, fontweight='bold', y=0.995)
    plt.savefig(output_dir / 'comprehensive_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # 5. Summary Table
    # ========================================================================
    summary_data = []
    for i, state in enumerate(state_labels):
        summary_data.append({
            'Model': 'HMM',
            'Regime': state,
            'Mean (%)': f"{df_hmm_stats.loc[i, 'mean']:.4f}",
            'Std (%)': f"{df_hmm_stats.loc[i, 'std']:.4f}",
            'Sharpe': f"{df_hmm_stats.loc[i, 'sharpe']:.4f}",
            'Days': int(df_hmm_stats.loc[i, 'count'])
        })
        summary_data.append({
            'Model': 'CJM',
            'Regime': state,
            'Mean (%)': f"{df_cjm_stats.loc[i, 'mean']:.4f}",
            'Std (%)': f"{df_cjm_stats.loc[i, 'std']:.4f}",
            'Sharpe': f"{df_cjm_stats.loc[i, 'sharpe']:.4f}",
            'Days': int(df_cjm_stats.loc[i, 'count'])
        })

    df_summary = pd.DataFrame(summary_data)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=df_summary.values,
                    colLabels=df_summary.columns,
                    cellLoc='center',
                    loc='center',
                    colColours=['#f0f0f0']*len(df_summary.columns))

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)

    # Color code model rows
    for i in range(len(summary_data)):
        if i % 2 == 0:  # HMM rows
            table[(i+1, 0)].set_facecolor('#cfe2f3')
        else:  # CJM rows
            table[(i+1, 0)].set_facecolor('#fce5cd')

    plt.title('Regime Return Statistics: HMM vs CJM', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_dir / 'summary_table.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Save to CSV
    df_summary.to_csv(output_dir / 'regime_stats_comparison.csv', index=False)

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nAll visualizations saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  1. mean_std_comparison.png - Mean and Std side by side")
    print("  2. risk_return_scatter.png - Risk-return scatter plot")
    print("  3. sharpe_ratio_comparison.png - Sharpe ratio comparison")
    print("  4. comprehensive_comparison.png - 4-panel summary")
    print("  5. summary_table.png - Summary statistics table")
    print("  6. regime_stats_comparison.csv - Data export")
    print("\n" + "="*80)

if __name__ == "__main__":
    visualize_regime_return_stats()
