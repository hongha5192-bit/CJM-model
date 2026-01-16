#!/usr/bin/env python
"""
Visualize simulated data from one simulation
Shows synthetic close prices with regime states
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import json
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_synthetic_prices(states, features, T, initial_price=1000):
    """
    Generate synthetic close prices based on regime states and features
    """
    # Define return characteristics for each state
    # Based on HMM parameters from real data
    regime_returns = {
        0: {'mean': 0.002020, 'std': 0.009207},   # Bullish regime
        1: {'mean': 0.000309, 'std': 0.011246},   # Neutral regime
        2: {'mean': -0.001313, 'std': 0.015930}   # Bearish regime
    }

    prices = np.zeros(T)
    prices[0] = initial_price
    returns = np.zeros(T)

    for t in range(1, T):
        state = states[t]

        # Generate return based on state
        base_return = np.random.normal(
            regime_returns[state]['mean'],
            regime_returns[state]['std']
        )

        # Add some influence from features for more realistic behavior
        # Use URSI as momentum indicator (higher URSI = more bullish)
        ursi_effect = (features[t, 1] - 50) / 500  # URSI is column 1

        # Use ADX as volatility amplifier (higher ADX = stronger trend)
        adx_effect = features[t, 0] / 100  # ADX is column 0

        # Combine effects
        returns[t] = base_return + ursi_effect * 0.001 * (1 + adx_effect)

        # Apply return to get new price
        prices[t] = prices[t-1] * (1 + returns[t])

    return prices, returns


def plot_simulation_with_regimes(sim_path, output_dir):
    """Create visualization of simulated data with regime states"""

    # Load simulation
    df = pd.read_parquet(sim_path)

    # Extract features and states
    feature_cols = ['ADX', 'URSI', 'BBWP', 'BB_PCTB', 'URSI_0_50', 'URSI_0_20']
    features = df[feature_cols].values
    states = df['s_true'].values
    T = len(df)

    # Generate synthetic prices
    prices, returns = generate_synthetic_prices(states, features, T)
    df['close'] = prices
    df['returns'] = returns

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))

    # Define colors for regimes
    regime_colors = {
        0: '#2ECC71',  # Green for Bullish
        1: '#3498DB',  # Blue for Neutral
        2: '#E74C3C'   # Red for Bearish
    }

    regime_names = {
        0: 'Bullish (High URSI)',
        1: 'Neutral (Mid Range)',
        2: 'Bearish (Low URSI)'
    }

    # Plot 1: Price chart with regime backgrounds
    ax1 = plt.subplot(3, 1, 1)

    # Add regime backgrounds
    for i in range(T):
        if i == 0 or states[i] != states[i-1]:
            # Find the end of this regime
            j = i + 1
            while j < T and states[j] == states[i]:
                j += 1

            ax1.axvspan(i, j, alpha=0.3, color=regime_colors[states[i]])

    # Plot price line
    ax1.plot(df.index, prices, color='black', linewidth=1.5, label='Simulated Price')
    ax1.set_ylabel('Price', fontsize=12)
    ax1.set_title(f'Simulated Price Series with Regime States (T={T})', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')

    # Add regime legend
    patches = [mpatches.Patch(color=regime_colors[k], alpha=0.3, label=regime_names[k])
               for k in sorted(regime_colors.keys())]
    ax1.legend(handles=patches, loc='upper right')

    # Plot 2: Returns with regime colors
    ax2 = plt.subplot(3, 1, 2)

    # Color returns by regime
    for state in regime_colors.keys():
        mask = states == state
        ax2.scatter(np.where(mask)[0], returns[mask],
                   color=regime_colors[state], alpha=0.6, s=10,
                   label=regime_names[state])

    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax2.set_ylabel('Returns', fontsize=12)
    ax2.set_title('Daily Returns by Regime', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')

    # Plot 3: Feature evolution (ADX and URSI)
    ax3 = plt.subplot(3, 1, 3)

    # Plot ADX and URSI
    ax3_ursi = ax3.twinx()

    line1 = ax3.plot(df.index, df['ADX'], color='purple', linewidth=1.5, label='ADX')
    line2 = ax3_ursi.plot(df.index, df['URSI'], color='orange', linewidth=1.5, label='URSI')

    # Add regime backgrounds
    for i in range(T):
        if i == 0 or states[i] != states[i-1]:
            j = i + 1
            while j < T and states[j] == states[i]:
                j += 1
            ax3.axvspan(i, j, alpha=0.2, color=regime_colors[states[i]])

    ax3.set_xlabel('Time', fontsize=12)
    ax3.set_ylabel('ADX', fontsize=12, color='purple')
    ax3_ursi.set_ylabel('URSI', fontsize=12, color='orange')
    ax3.set_title('Technical Indicators Evolution', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper left')

    plt.tight_layout()

    # Save plot
    sim_name = Path(sim_path).stem
    plot_path = output_dir / f'simulation_visualization_{sim_name}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    return df, plot_path


def create_summary_statistics(df, states, prices, returns):
    """Generate summary statistics for the simulation"""

    stats = {}

    # Overall statistics
    stats['overall'] = {
        'total_return': (prices[-1] / prices[0] - 1) * 100,
        'mean_daily_return': returns[1:].mean() * 100,
        'std_daily_return': returns[1:].std() * 100,
        'sharpe_ratio': returns[1:].mean() / returns[1:].std() * np.sqrt(252) if returns[1:].std() > 0 else 0
    }

    # Per-regime statistics
    for state in [0, 1, 2]:
        mask = states == state
        state_returns = returns[mask]

        stats[f'regime_{state}'] = {
            'frequency': mask.sum() / len(states) * 100,
            'mean_return': state_returns.mean() * 100,
            'std_return': state_returns.std() * 100,
            'count': mask.sum()
        }

    # Feature statistics by regime
    feature_cols = ['ADX', 'URSI', 'BBWP', 'BB_PCTB', 'URSI_0_50', 'URSI_0_20']
    for state in [0, 1, 2]:
        mask = states == state
        stats[f'regime_{state}_features'] = {}
        for feat in feature_cols:
            stats[f'regime_{state}_features'][feat] = {
                'mean': df.loc[mask, feat].mean(),
                'std': df.loc[mask, feat].std()
            }

    return stats


def main():
    """Main visualization function"""

    logger.info("="*80)
    logger.info("VISUALIZING SIMULATED DATA")
    logger.info("="*80)

    # Load first simulation
    sim_dir = Path('outputs/step2_sim/K3_512_corrected')
    sim_files = sorted(sim_dir.glob('sim_*.parquet'))

    if not sim_files:
        logger.error(f"No simulation files found in {sim_dir}")
        return

    # Use first simulation
    sim_path = sim_files[0]
    logger.info(f"Loading simulation: {sim_path.name}")

    # Create output directory
    output_dir = Path('outputs/visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create visualization
    df, plot_path = plot_simulation_with_regimes(sim_path, output_dir)

    # Extract data
    states = df['s_true'].values
    prices = df['close'].values
    returns = df['returns'].values

    # Generate statistics
    stats = create_summary_statistics(df, states, prices, returns)

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("SIMULATION SUMMARY")
    logger.info("="*60)

    logger.info(f"\nOverall Performance:")
    logger.info(f"  Total Return: {stats['overall']['total_return']:.2f}%")
    logger.info(f"  Mean Daily Return: {stats['overall']['mean_daily_return']:.4f}%")
    logger.info(f"  Std Daily Return: {stats['overall']['std_daily_return']:.4f}%")
    logger.info(f"  Sharpe Ratio: {stats['overall']['sharpe_ratio']:.2f}")

    logger.info(f"\nRegime Distribution:")
    for state in [0, 1, 2]:
        regime_names = {0: 'Bullish', 1: 'Neutral', 2: 'Bearish'}
        logger.info(f"  {regime_names[state]:8s}: {stats[f'regime_{state}']['frequency']:5.1f}% "
                   f"({stats[f'regime_{state}']['count']:3d} days), "
                   f"Return: {stats[f'regime_{state}']['mean_return']:+.4f}% ± {stats[f'regime_{state}']['std_return']:.4f}%")

    logger.info(f"\nFeature Averages by Regime:")
    feature_cols = ['ADX', 'URSI', 'BBWP', 'BB_PCTB', 'URSI_0_50', 'URSI_0_20']

    logger.info(f"{'Feature':10s} | {'Bullish':>12s} | {'Neutral':>12s} | {'Bearish':>12s}")
    logger.info("-" * 55)
    for feat in feature_cols:
        bullish = stats['regime_0_features'][feat]['mean']
        neutral = stats['regime_1_features'][feat]['mean']
        bearish = stats['regime_2_features'][feat]['mean']
        logger.info(f"{feat:10s} | {bullish:12.2f} | {neutral:12.2f} | {bearish:12.2f}")

    # Save statistics to JSON
    stats_path = output_dir / f'simulation_statistics_{sim_path.stem}.json'
    with open(stats_path, 'w') as f:
        # Convert numpy values to Python types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        stats_json = json.loads(json.dumps(stats, default=convert))
        json.dump(stats_json, f, indent=2)

    logger.info(f"\n✅ Visualization saved to: {plot_path}")
    logger.info(f"📊 Statistics saved to: {stats_path}")

    return df, stats


if __name__ == "__main__":
    main()