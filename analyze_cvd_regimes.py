"""
Analyze CVD behavior during different HMM regimes
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

def analyze_cvd_by_regime():
    """Analyze CVD characteristics in different market regimes"""

    print("="*80)
    print("CVD ANALYSIS BY MARKET REGIME")
    print("="*80)

    # Load CVD data
    print("\n1. Loading data...")
    cvd_df = pd.read_csv('outputs/vnindex_cvd_analysis.csv', parse_dates=['date'])
    print(f"   Loaded CVD data: {len(cvd_df)} rows")

    # Load HMM states
    hmm_path = Path('outputs/step1_params/hmm_states_K3.csv')
    if hmm_path.exists():
        hmm_df = pd.read_csv(hmm_path, parse_dates=['date'])
        print(f"   Loaded HMM states: {len(hmm_df)} rows")

        # Merge data
        df = pd.merge(cvd_df, hmm_df[['date', 'hmm_state']], on='date', how='left')

        # Fill missing HMM states (if any) with mode
        df['hmm_state'].fillna(df['hmm_state'].mode()[0], inplace=True)
    else:
        print("   HMM states not found. Using simple regime classification based on returns...")
        # Simple regime classification based on returns
        df = cvd_df.copy()
        df['returns'] = df['CLOSEINDEX'].pct_change()
        df['MA20'] = df['CLOSEINDEX'].rolling(20).mean()
        df['MA50'] = df['CLOSEINDEX'].rolling(50).mean()

        # Define regimes
        df['hmm_state'] = 1  # Default to neutral
        df.loc[(df['CLOSEINDEX'] > df['MA20']) & (df['MA20'] > df['MA50']), 'hmm_state'] = 0  # Bullish
        df.loc[(df['CLOSEINDEX'] < df['MA20']) & (df['MA20'] < df['MA50']), 'hmm_state'] = 2  # Bearish

    # Calculate CVD metrics
    df['CVD_CHANGE'] = df['CVD'].diff()
    df['CVD_PCT_CHANGE'] = df['CVD'].pct_change() * 100
    df['VOLUME_DELTA_RATIO'] = df['VOLUME_DELTA'] / df['TOTALMATCHVOLUME']

    # Regime labels
    regime_labels = {0: 'Bullish', 1: 'Neutral', 2: 'Bearish'}

    # Convert hmm_state to int
    df['hmm_state'] = df['hmm_state'].fillna(1).astype(int)

    print("\n2. CVD Statistics by Regime:")
    print("-" * 60)

    for regime in sorted(df['hmm_state'].unique()):
        regime_data = df[df['hmm_state'] == regime]
        print(f"\n{regime_labels[regime]} Regime (State {regime}):")
        print(f"   Days in regime: {len(regime_data)} ({len(regime_data)/len(df)*100:.1f}%)")
        print(f"   Avg Volume Delta: {regime_data['VOLUME_DELTA'].mean():,.0f}")
        print(f"   Avg CVD Change: {regime_data['CVD_CHANGE'].mean():,.0f}")
        print(f"   Volume Delta Ratio: {regime_data['VOLUME_DELTA_RATIO'].mean():.4f}")
        print(f"   Buy/Sell Ratio: {regime_data['BUY_VOLUME'].sum() / regime_data['SELL_VOLUME'].sum():.3f}")

        # Price performance
        price_return = (regime_data['CLOSEINDEX'].iloc[-1] / regime_data['CLOSEINDEX'].iloc[0] - 1) * 100 if len(regime_data) > 0 else 0
        print(f"   Avg Price Change: {regime_data['CLOSEINDEX'].pct_change().mean()*100:.3f}%")

    # Statistical tests
    print("\n3. Statistical Analysis:")
    print("-" * 60)

    # ANOVA test for Volume Delta across regimes
    regime_groups = [df[df['hmm_state'] == i]['VOLUME_DELTA'].dropna() for i in range(3)]
    f_stat, p_value = stats.f_oneway(*regime_groups)
    print(f"   ANOVA test for Volume Delta across regimes:")
    print(f"   F-statistic: {f_stat:.4f}")
    print(f"   P-value: {p_value:.6f}")

    if p_value < 0.05:
        print("   ✓ Significant difference in Volume Delta across regimes")
    else:
        print("   ✗ No significant difference in Volume Delta across regimes")

    # Correlation analysis
    print("\n4. Correlation Analysis:")
    print("-" * 60)

    for regime in sorted(df['hmm_state'].unique()):
        regime_data = df[df['hmm_state'] == regime]

        # Calculate returns
        regime_data['returns'] = regime_data['CLOSEINDEX'].pct_change()

        # Correlation between CVD change and returns
        corr = regime_data['CVD_CHANGE'].corr(regime_data['returns'])
        print(f"   {regime_labels[regime]}: CVD Change vs Returns correlation = {corr:.3f}")

    # Create visualization
    print("\n5. Creating visualizations...")

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # Plot 1: Volume Delta by Regime
    ax1 = axes[0, 0]
    regime_data_list = []
    for regime in sorted(df['hmm_state'].unique()):
        regime_data_list.append(df[df['hmm_state'] == regime]['VOLUME_DELTA'].values)

    bp = ax1.boxplot(regime_data_list, labels=[regime_labels[i] for i in range(3)], patch_artist=True)
    colors = ['green', 'yellow', 'red']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax1.set_ylabel('Volume Delta')
    ax1.set_title('Volume Delta Distribution by Regime')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)

    # Plot 2: CVD Change by Regime
    ax2 = axes[0, 1]
    for regime in sorted(df['hmm_state'].unique()):
        regime_data = df[df['hmm_state'] == regime]
        regime_idx = int(regime) if not pd.isna(regime) else 1  # Convert to int
        ax2.hist(regime_data['CVD_CHANGE'].dropna(), alpha=0.6,
                label=regime_labels[regime_idx], bins=50, color=colors[regime_idx])

    ax2.set_xlabel('CVD Change')
    ax2.set_ylabel('Frequency')
    ax2.set_title('CVD Change Distribution by Regime')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Buy/Sell Volume Ratio by Regime
    ax3 = axes[1, 0]
    buy_sell_ratios = []
    for regime in sorted(df['hmm_state'].unique()):
        regime_data = df[df['hmm_state'] == regime]
        ratio = regime_data['BUY_VOLUME'].sum() / regime_data['SELL_VOLUME'].sum()
        buy_sell_ratios.append(ratio)

    bars = ax3.bar([regime_labels[i] for i in range(3)], buy_sell_ratios, color=colors, alpha=0.7)
    ax3.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    ax3.set_ylabel('Buy/Sell Volume Ratio')
    ax3.set_title('Buy/Sell Volume Ratio by Regime')
    ax3.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, ratio in zip(bars, buy_sell_ratios):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{ratio:.3f}', ha='center', va='bottom')

    # Plot 4: Average Daily Volume by Regime
    ax4 = axes[1, 1]
    avg_volumes = []
    for regime in sorted(df['hmm_state'].unique()):
        regime_data = df[df['hmm_state'] == regime]
        avg_vol = regime_data['TOTALMATCHVOLUME'].mean()
        avg_volumes.append(avg_vol)

    bars = ax4.bar([regime_labels[i] for i in range(3)], avg_volumes, color=colors, alpha=0.7)
    ax4.set_ylabel('Average Daily Volume')
    ax4.set_title('Average Trading Volume by Regime')
    ax4.grid(True, alpha=0.3)

    # Plot 5: CVD Trend with Regime Colors
    ax5 = axes[2, 0]

    # Plot CVD colored by regime
    for regime in sorted(df['hmm_state'].unique()):
        regime_data = df[df['hmm_state'] == regime]
        regime_idx = int(regime) if not pd.isna(regime) else 1  # Convert to int
        ax5.scatter(regime_data['date'], regime_data['CVD'],
                   c=colors[regime_idx], alpha=0.6, s=1, label=regime_labels[regime_idx])

    ax5.set_xlabel('Date')
    ax5.set_ylabel('CVD')
    ax5.set_title('CVD Evolution Colored by Market Regime')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Volume Delta Efficiency (Delta/Volume ratio)
    ax6 = axes[2, 1]
    efficiency_data = []
    for regime in sorted(df['hmm_state'].unique()):
        regime_data = df[df['hmm_state'] == regime]
        efficiency_data.append(regime_data['VOLUME_DELTA_RATIO'].dropna().values)

    bp = ax6.boxplot(efficiency_data, labels=[regime_labels[i] for i in range(3)], patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax6.set_ylabel('Volume Delta / Total Volume')
    ax6.set_title('Volume Delta Efficiency by Regime')
    ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_path = 'outputs/cvd_regime_analysis.png'
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"   Visualization saved to: {output_path}")

    # Summary statistics table
    print("\n6. Summary Table:")
    print("-" * 80)

    summary_data = []
    for regime in sorted(df['hmm_state'].unique()):
        regime_data = df[df['hmm_state'] == regime]
        summary_data.append({
            'Regime': regime_labels[regime],
            'Days': len(regime_data),
            'Pct_Days': f"{len(regime_data)/len(df)*100:.1f}%",
            'Avg_Volume': f"{regime_data['TOTALMATCHVOLUME'].mean():,.0f}",
            'Avg_Delta': f"{regime_data['VOLUME_DELTA'].mean():,.0f}",
            'Buy_Sell_Ratio': f"{regime_data['BUY_VOLUME'].sum()/regime_data['SELL_VOLUME'].sum():.3f}",
            'Delta_Efficiency': f"{regime_data['VOLUME_DELTA_RATIO'].mean():.4f}"
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    # Save detailed results
    output_csv = 'outputs/cvd_regime_statistics.csv'
    df.to_csv(output_csv, index=False)
    print(f"\n   Detailed results saved to: {output_csv}")

    print("\n" + "="*80)
    print("CVD REGIME ANALYSIS COMPLETE")
    print("="*80)

    return df

if __name__ == "__main__":
    df_analysis = analyze_cvd_by_regime()