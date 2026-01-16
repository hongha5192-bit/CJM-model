#!/usr/bin/env python3
"""
Compare HMM vs CJM regime assignments
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, adjusted_rand_score, normalized_mutual_info_score

def compare_hmm_cjm():
    """Compare HMM and CJM regime assignments"""

    print("="*80)
    print("COMPARING HMM vs CJM REGIME ASSIGNMENTS")
    print("="*80)

    # Load HMM states
    print("\nLoading HMM state assignments...")
    df_hmm = pd.read_csv('outputs/step1_params/hmm_states_K3.csv')
    df_hmm['date'] = pd.to_datetime(df_hmm['date'])
    print(f"HMM data shape: {df_hmm.shape}")

    # Load CJM states (relabeled)
    print("Loading CJM regime assignments (relabeled)...")
    df_cjm = pd.read_csv('outputs/step4_apply/regimes_daily_K3_relabeled.csv')
    df_cjm['date'] = pd.to_datetime(df_cjm['date'])
    print(f"CJM data shape: {df_cjm.shape}")

    # Load OHLC data
    print("Loading OHLC data...")
    df_ohlc = pd.read_csv('data/vnindex_full.csv')
    df_ohlc['date'] = pd.to_datetime(df_ohlc['date'])
    df_ohlc = df_ohlc.rename(columns={'CLOSEINDEX': 'close'})

    # Merge all data
    print("Merging data...")
    df = df_hmm.merge(df_cjm[['date', 'regime_relabeled']],
                      on='date', how='inner')
    df = df.rename(columns={'regime_relabeled': 'cjm_state'})
    df = df.merge(df_ohlc[['date', 'close']], on='date', how='left')

    print(f"Merged data shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Create output directory
    output_dir = Path('outputs/hmm_cjm_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)

    # State labels
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

    # ========================================================================
    # 1. Agreement Analysis
    # ========================================================================
    print("\n" + "="*80)
    print("AGREEMENT ANALYSIS")
    print("="*80)

    # Calculate agreement
    agreement = (df['hmm_state'] == df['cjm_state']).sum()
    agreement_pct = agreement / len(df) * 100

    print(f"\nOverall Agreement: {agreement}/{len(df)} ({agreement_pct:.2f}%)")

    # Adjusted Rand Index (measure of clustering similarity)
    ari = adjusted_rand_score(df['hmm_state'], df['cjm_state'])
    print(f"Adjusted Rand Index: {ari:.4f} (1.0 = perfect agreement, 0 = random)")

    # Normalized Mutual Information
    nmi = normalized_mutual_info_score(df['hmm_state'], df['cjm_state'])
    print(f"Normalized Mutual Information: {nmi:.4f} (1.0 = perfect, 0 = independent)")

    # ========================================================================
    # 2. Confusion Matrix
    # ========================================================================
    print("\n" + "="*80)
    print("CONFUSION MATRIX")
    print("="*80)

    cm = confusion_matrix(df['hmm_state'], df['cjm_state'])
    print("\nRows = HMM States, Columns = CJM States")
    print(cm)

    # Normalize confusion matrix by HMM states (rows)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print("\nNormalized by HMM states (% of each HMM state assigned to each CJM state):")
    for i in range(3):
        print(f"\nHMM {state_labels[i]} (State {i}):")
        for j in range(3):
            print(f"  → CJM {state_labels[j]}: {cm_normalized[i,j]*100:.1f}% ({cm[i,j]} days)")

    # Plot confusion matrix
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Absolute counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=[state_labels[i] for i in range(3)],
                yticklabels=[state_labels[i] for i in range(3)],
                cbar_kws={'label': 'Count'})
    ax1.set_title('Confusion Matrix: HMM vs CJM\n(Absolute Counts)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('CJM Regime')
    ax1.set_ylabel('HMM Regime')

    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=ax2,
                xticklabels=[state_labels[i] for i in range(3)],
                yticklabels=[state_labels[i] for i in range(3)],
                cbar_kws={'label': 'Percentage'})
    ax2.set_title('Confusion Matrix: HMM vs CJM\n(Normalized by HMM)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('CJM Regime')
    ax2.set_ylabel('HMM Regime')

    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # 3. Distribution Comparison
    # ========================================================================
    print("\n" + "="*80)
    print("DISTRIBUTION COMPARISON")
    print("="*80)

    hmm_dist = df['hmm_state'].value_counts().sort_index()
    cjm_dist = df['cjm_state'].value_counts().sort_index()

    print("\nHMM Distribution:")
    for state in range(3):
        count = hmm_dist.get(state, 0)
        pct = count / len(df) * 100
        print(f"  {state_labels[state]} (State {state}): {count:4d} days ({pct:5.2f}%)")

    print("\nCJM Distribution:")
    for state in range(3):
        count = cjm_dist.get(state, 0)
        pct = count / len(df) * 100
        print(f"  {state_labels[state]} (State {state}): {count:4d} days ({pct:5.2f}%)")

    # Plot distribution comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(3)
    width = 0.35

    bars1 = ax.bar(x - width/2, [hmm_dist.get(i, 0) for i in range(3)], width,
                   label='HMM', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, [cjm_dist.get(i, 0) for i in range(3)], width,
                   label='CJM', color='coral', alpha=0.8)

    ax.set_xlabel('Regime')
    ax.set_ylabel('Number of Days')
    ax.set_title('Regime Distribution: HMM vs CJM', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([state_labels[i] for i in range(3)])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'distribution_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # 4. Timeline Comparison
    # ========================================================================
    print("\nCreating timeline comparison...")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    # Price
    ax1.plot(df['date'], df['close'], color='black', linewidth=1)
    ax1.set_ylabel('VNINDEX Close', fontsize=11)
    ax1.set_title('VNINDEX Price and Regime Comparison: HMM vs CJM', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # HMM regimes
    for state in range(3):
        state_data = df[df['hmm_state'] == state]
        ax2.scatter(state_data['date'], state_data['hmm_state'],
                   color=state_colors[state], s=8, alpha=0.6)
    ax2.set_ylabel('HMM Regime', fontsize=11)
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['Bullish', 'Neutral', 'Bearish'])
    ax2.grid(True, alpha=0.3)

    # CJM regimes
    for state in range(3):
        state_data = df[df['cjm_state'] == state]
        ax3.scatter(state_data['date'], state_data['cjm_state'],
                   color=state_colors[state], s=8, alpha=0.6)
    ax3.set_ylabel('CJM Regime', fontsize=11)
    ax3.set_xlabel('Date', fontsize=11)
    ax3.set_yticks([0, 1, 2])
    ax3.set_yticklabels(['Bullish', 'Neutral', 'Bearish'])
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'timeline_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # 5. Agreement over time
    # ========================================================================
    print("Creating agreement over time analysis...")

    df['agreement'] = (df['hmm_state'] == df['cjm_state']).astype(int)
    df['year_month'] = df['date'].dt.to_period('M')

    # Calculate monthly agreement rate
    monthly_agreement = df.groupby('year_month').agg({
        'agreement': 'mean',
        'date': 'count'
    }).rename(columns={'agreement': 'agreement_rate', 'date': 'count'})

    monthly_agreement.index = monthly_agreement.index.to_timestamp()

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(monthly_agreement.index, monthly_agreement['agreement_rate'] * 100,
            color='steelblue', linewidth=2, marker='o', markersize=4)
    ax.axhline(y=agreement_pct, color='red', linestyle='--', linewidth=2,
              label=f'Overall Agreement: {agreement_pct:.1f}%')

    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Agreement Rate (%)', fontsize=11)
    ax.set_title('HMM vs CJM Agreement Rate Over Time (Monthly)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_dir / 'agreement_over_time.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # 6. Return statistics comparison
    # ========================================================================
    print("\n" + "="*80)
    print("RETURN STATISTICS COMPARISON")
    print("="*80)

    print("\nHMM Regime Returns:")
    for state in range(3):
        state_data = df[df['hmm_state'] == state]
        print(f"  {state_labels[state]} (State {state}):")
        print(f"    Mean: {state_data['ret'].mean()*100:7.3f}%")
        print(f"    Std:  {state_data['ret'].std()*100:7.3f}%")
        print(f"    Count: {len(state_data):4d} days")

    print("\nCJM Regime Returns:")
    for state in range(3):
        state_data = df[df['cjm_state'] == state]
        if len(state_data) > 0:
            print(f"  {state_labels[state]} (State {state}):")
            print(f"    Mean: {state_data['ret'].mean()*100:7.3f}%")
            print(f"    Std:  {state_data['ret'].std()*100:7.3f}%")
            print(f"    Count: {len(state_data):4d} days")

    # ========================================================================
    # 7. Disagreement analysis
    # ========================================================================
    print("\n" + "="*80)
    print("DISAGREEMENT ANALYSIS")
    print("="*80)

    df['disagree'] = df['hmm_state'] != df['cjm_state']
    disagree_df = df[df['disagree']]

    print(f"\nTotal disagreements: {len(disagree_df)} days ({len(disagree_df)/len(df)*100:.2f}%)")

    print("\nDisagreement patterns:")
    disagree_patterns = disagree_df.groupby(['hmm_state', 'cjm_state']).size().reset_index(name='count')
    disagree_patterns = disagree_patterns.sort_values('count', ascending=False)

    for _, row in disagree_patterns.iterrows():
        hmm_s = int(row['hmm_state'])
        cjm_s = int(row['cjm_state'])
        count = int(row['count'])
        pct = count / len(disagree_df) * 100
        print(f"  HMM={state_labels[hmm_s]}, CJM={state_labels[cjm_s]}: {count:4d} ({pct:5.1f}%)")

    # ========================================================================
    # 8. Summary table
    # ========================================================================
    print("\nCreating summary comparison table...")

    summary_data = []
    for state in range(3):
        hmm_data = df[df['hmm_state'] == state]
        cjm_data = df[df['cjm_state'] == state]

        summary_data.append({
            'Model': 'HMM',
            'Regime': state_labels[state],
            'Days': len(hmm_data),
            'Percentage': f"{len(hmm_data)/len(df)*100:.1f}%",
            'Mean Return': f"{hmm_data['ret'].mean()*100:.3f}%",
            'Std Return': f"{hmm_data['ret'].std()*100:.3f}%"
        })

        summary_data.append({
            'Model': 'CJM',
            'Regime': state_labels[state],
            'Days': len(cjm_data),
            'Percentage': f"{len(cjm_data)/len(df)*100:.1f}%",
            'Mean Return': f"{cjm_data['ret'].mean()*100:.3f}%",
            'Std Return': f"{cjm_data['ret'].std()*100:.3f}%"
        })

    df_summary = pd.DataFrame(summary_data)

    fig, ax = plt.subplots(figsize=(12, 4))
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

    plt.title('HMM vs CJM Regime Comparison Summary', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_dir / 'comparison_summary_table.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Save detailed data
    df[['date', 'close', 'ret', 'hmm_state', 'cjm_state', 'agreement']].to_csv(
        output_dir / 'hmm_cjm_comparison_data.csv', index=False
    )

    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  1. confusion_matrix.png")
    print("  2. distribution_comparison.png")
    print("  3. timeline_comparison.png")
    print("  4. agreement_over_time.png")
    print("  5. comparison_summary_table.png")
    print("  6. hmm_cjm_comparison_data.csv")
    print("\n" + "="*80)

if __name__ == "__main__":
    compare_hmm_cjm()
