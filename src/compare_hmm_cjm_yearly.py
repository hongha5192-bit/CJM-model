#!/usr/bin/env python3
"""
Compare HMM vs CJM regime distributions by year
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def compare_yearly_distributions():
    """Compare HMM vs CJM regime distributions year by year"""

    print("="*80)
    print("YEARLY REGIME DISTRIBUTION: HMM vs CJM")
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

    # Add year
    df['year'] = df['date'].dt.year

    print(f"Data shape: {df.shape}")
    print(f"Years covered: {df['year'].min()} - {df['year'].max()}")

    # State labels and colors
    state_labels = {
        0: "Bullish",
        1: "Neutral",
        2: "Bearish"
    }

    state_colors = {
        0: '#2ecc71',
        1: '#f39c12',
        2: '#e74c3c'
    }

    # Create output directory
    output_dir = Path('outputs/hmm_cjm_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # 1. Calculate yearly distributions
    # ========================================================================
    print("\n" + "="*80)
    print("YEARLY DISTRIBUTIONS")
    print("="*80)

    years = sorted(df['year'].unique())

    # HMM distributions
    hmm_yearly = df.groupby(['year', 'hmm_state']).size().unstack(fill_value=0)
    hmm_yearly_pct = hmm_yearly.div(hmm_yearly.sum(axis=1), axis=0) * 100

    # CJM distributions
    cjm_yearly = df.groupby(['year', 'cjm_state']).size().unstack(fill_value=0)
    cjm_yearly_pct = cjm_yearly.div(cjm_yearly.sum(axis=1), axis=0) * 100

    # Print comparison table
    for year in years:
        print(f"\n--- {year} ---")
        print("HMM Distribution:")
        for state in range(3):
            count = hmm_yearly.loc[year, state] if state in hmm_yearly.columns else 0
            pct = hmm_yearly_pct.loc[year, state] if state in hmm_yearly_pct.columns else 0
            print(f"  {state_labels[state]:8s}: {count:3.0f} days ({pct:5.1f}%)")

        print("CJM Distribution:")
        for state in range(3):
            count = cjm_yearly.loc[year, state] if state in cjm_yearly.columns else 0
            pct = cjm_yearly_pct.loc[year, state] if state in cjm_yearly_pct.columns else 0
            print(f"  {state_labels[state]:8s}: {count:3.0f} days ({pct:5.1f}%)")

        # Calculate agreement
        year_data = df[df['year'] == year]
        agreement = (year_data['hmm_state'] == year_data['cjm_state']).sum()
        agreement_pct = agreement / len(year_data) * 100
        print(f"Agreement: {agreement}/{len(year_data)} ({agreement_pct:.1f}%)")

    # ========================================================================
    # 2. Side-by-side bar charts
    # ========================================================================
    print("\nCreating side-by-side comparison...")

    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    for idx, state in enumerate([0, 1, 2]):
        ax = axes[idx]

        x = np.arange(len(years))
        width = 0.35

        # Get data for this state
        hmm_data = [hmm_yearly_pct.loc[year, state] if state in hmm_yearly_pct.columns else 0
                    for year in years]
        cjm_data = [cjm_yearly_pct.loc[year, state] if state in cjm_yearly_pct.columns else 0
                    for year in years]

        bars1 = ax.bar(x - width/2, hmm_data, width, label='HMM',
                      color='steelblue', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x + width/2, cjm_data, width, label='CJM',
                      color='coral', alpha=0.8, edgecolor='black')

        ax.set_ylabel('Percentage (%)', fontsize=11)
        ax.set_title(f'{state_labels[state]} Regime Distribution by Year',
                    fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(years, rotation=45)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 100)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.0f}%',
                           ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'yearly_distribution_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # 3. Stacked bar charts
    # ========================================================================
    print("Creating stacked bar charts...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # HMM stacked
    bottom = np.zeros(len(years))
    for state in [0, 1, 2]:
        values = [hmm_yearly_pct.loc[year, state] if state in hmm_yearly_pct.columns else 0
                 for year in years]
        ax1.bar(years, values, bottom=bottom, label=state_labels[state],
               color=state_colors[state], alpha=0.8, edgecolor='black')
        bottom += values

    ax1.set_title('HMM Regime Distribution by Year', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Percentage (%)')
    ax1.legend(loc='upper left')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 100)

    # CJM stacked
    bottom = np.zeros(len(years))
    for state in [0, 1, 2]:
        values = [cjm_yearly_pct.loc[year, state] if state in cjm_yearly_pct.columns else 0
                 for year in years]
        ax2.bar(years, values, bottom=bottom, label=state_labels[state],
               color=state_colors[state], alpha=0.8, edgecolor='black')
        bottom += values

    ax2.set_title('CJM Regime Distribution by Year', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Percentage (%)')
    ax2.legend(loc='upper left')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_dir / 'yearly_distribution_stacked.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # 4. Difference heatmap
    # ========================================================================
    print("Creating difference heatmap...")

    # Calculate differences (CJM - HMM)
    diff_matrix = cjm_yearly_pct - hmm_yearly_pct
    diff_matrix = diff_matrix.fillna(0)

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.heatmap(diff_matrix.T, annot=True, fmt='.1f', cmap='RdBu_r', center=0,
                cbar_kws={'label': 'Difference (CJM - HMM) in %'},
                yticklabels=[state_labels[i] for i in range(3)],
                xticklabels=years,
                vmin=-30, vmax=30, ax=ax)

    ax.set_title('Regime Distribution Difference: CJM - HMM (%)\n(Positive = CJM assigns more to this regime)',
                fontsize=12, fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Regime')

    plt.tight_layout()
    plt.savefig(output_dir / 'yearly_distribution_difference.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # 5. Agreement rate by year
    # ========================================================================
    print("Creating agreement rate by year...")

    yearly_agreement = []
    for year in years:
        year_data = df[df['year'] == year]
        agreement = (year_data['hmm_state'] == year_data['cjm_state']).sum()
        agreement_pct = agreement / len(year_data) * 100
        yearly_agreement.append({
            'year': year,
            'agreement_pct': agreement_pct,
            'total_days': len(year_data)
        })

    df_agreement = pd.DataFrame(yearly_agreement)

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(df_agreement['year'], df_agreement['agreement_pct'],
                  color='steelblue', alpha=0.8, edgecolor='black')

    # Add overall average line
    overall_agreement = (df['hmm_state'] == df['cjm_state']).sum() / len(df) * 100
    ax.axhline(y=overall_agreement, color='red', linestyle='--', linewidth=2,
              label=f'Overall Average: {overall_agreement:.1f}%')

    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Agreement Rate (%)', fontsize=11)
    ax.set_title('HMM vs CJM Agreement Rate by Year', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%',
               ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'yearly_agreement_rate.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # 6. Line plot - regime percentages over time
    # ========================================================================
    print("Creating line plot...")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for idx, state in enumerate([0, 1, 2]):
        ax = axes[idx]

        hmm_data = [hmm_yearly_pct.loc[year, state] if state in hmm_yearly_pct.columns else 0
                    for year in years]
        cjm_data = [cjm_yearly_pct.loc[year, state] if state in cjm_yearly_pct.columns else 0
                    for year in years]

        ax.plot(years, hmm_data, marker='o', linewidth=2, markersize=8,
               label='HMM', color='steelblue')
        ax.plot(years, cjm_data, marker='s', linewidth=2, markersize=8,
               label='CJM', color='coral')

        ax.set_ylabel('Percentage (%)', fontsize=11)
        ax.set_title(f'{state_labels[state]} Regime', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(max(hmm_data), max(cjm_data)) * 1.1)

    axes[2].set_xlabel('Year', fontsize=11)

    plt.suptitle('Regime Distribution Trends: HMM vs CJM', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'yearly_distribution_trends.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # 7. Summary table
    # ========================================================================
    print("Creating summary table...")

    summary_data = []
    for year in years:
        year_data = df[df['year'] == year]
        agreement = (year_data['hmm_state'] == year_data['cjm_state']).sum()
        agreement_pct = agreement / len(year_data) * 100

        row = {'Year': year, 'Days': len(year_data), 'Agreement': f'{agreement_pct:.1f}%'}

        for state in range(3):
            hmm_count = hmm_yearly.loc[year, state] if state in hmm_yearly.columns else 0
            cjm_count = cjm_yearly.loc[year, state] if state in cjm_yearly.columns else 0
            row[f'HMM_{state_labels[state]}'] = f'{hmm_count:.0f}'
            row[f'CJM_{state_labels[state]}'] = f'{cjm_count:.0f}'

        summary_data.append(row)

    df_summary = pd.DataFrame(summary_data)

    # Save to CSV
    df_summary.to_csv(output_dir / 'yearly_comparison_summary.csv', index=False)

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=df_summary.values,
                    colLabels=df_summary.columns,
                    cellLoc='center',
                    loc='center',
                    colColours=['#f0f0f0']*len(df_summary.columns))

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)

    plt.title('Yearly Regime Distribution: HMM vs CJM (Day Counts)',
             fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_dir / 'yearly_comparison_table.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nAll visualizations saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  1. yearly_distribution_comparison.png - Side-by-side bar charts")
    print("  2. yearly_distribution_stacked.png - Stacked bar charts")
    print("  3. yearly_distribution_difference.png - Difference heatmap")
    print("  4. yearly_agreement_rate.png - Agreement rate by year")
    print("  5. yearly_distribution_trends.png - Line plot trends")
    print("  6. yearly_comparison_table.png - Summary table")
    print("  7. yearly_comparison_summary.csv - Data table")
    print("\n" + "="*80)

if __name__ == "__main__":
    compare_yearly_distributions()
