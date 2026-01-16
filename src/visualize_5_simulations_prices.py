#!/usr/bin/env python
"""
Visualize daily close prices from 5 different simulations
Focused on price comparison across multiple simulations
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_synthetic_prices(states, features, T, initial_price=1000, seed=None):
    """
    Generate synthetic close prices based on regime states and features
    """
    if seed is not None:
        np.random.seed(seed)

    # Define return characteristics for each state
    regime_returns = {
        0: {'mean': 0.002020, 'std': 0.009207},   # Bullish regime
        1: {'mean': 0.000309, 'std': 0.011246},   # Neutral regime
        2: {'mean': -0.001313, 'std': 0.015930}   # Bearish regime
    }

    prices = np.zeros(T)
    prices[0] = initial_price

    for t in range(1, T):
        state = states[t]

        # Generate return based on state
        base_return = np.random.normal(
            regime_returns[state]['mean'],
            regime_returns[state]['std']
        )

        # Add some influence from features
        ursi_effect = (features[t, 1] - 50) / 500  # URSI is column 1
        adx_effect = features[t, 0] / 100  # ADX is column 0

        # Combine effects
        returns = base_return + ursi_effect * 0.001 * (1 + adx_effect)

        # Apply return to get new price
        prices[t] = prices[t-1] * (1 + returns)

    return prices


def plot_multiple_simulations(sim_files, num_sims=5):
    """Create visualization of close prices from multiple simulations"""

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    # Color palette for different simulations
    colors = ['#2E4057', '#048A81', '#54C6EB', '#8FE1A2', '#F18F01']

    # Store price data and statistics
    all_prices = []
    all_states = []
    price_stats = []

    logger.info(f"Loading {num_sims} simulations...")

    for i, sim_path in enumerate(sim_files[:num_sims]):
        # Load simulation
        df = pd.read_parquet(sim_path)

        # Extract features and states
        feature_cols = ['ADX', 'URSI', 'BBWP', 'BB_PCTB', 'URSI_0_50', 'URSI_0_20']
        features = df[feature_cols].values
        states = df['s_true'].values
        T = len(df)

        # Generate synthetic prices with unique seed for each simulation
        seed = 1000 + i  # Different seed for price generation
        prices = generate_synthetic_prices(states, features, T, initial_price=1000, seed=seed)

        all_prices.append(prices)
        all_states.append(states)

        # Calculate statistics
        total_return = (prices[-1] / prices[0] - 1) * 100
        daily_returns = np.diff(prices) / prices[:-1]
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0

        price_stats.append({
            'sim': i + 1,
            'final_price': prices[-1],
            'total_return': total_return,
            'sharpe': sharpe,
            'max_price': prices.max(),
            'min_price': prices.min()
        })

        # Plot on first axis - Individual price series
        axes[0].plot(range(T), prices, color=colors[i], alpha=0.7, linewidth=1.5,
                    label=f'Sim {i+1}: Return={total_return:.1f}%')

        logger.info(f"  Sim {i+1}: Final={prices[-1]:.0f}, Return={total_return:.1f}%, Sharpe={sharpe:.2f}")

    # First plot - Individual price series
    axes[0].set_title('Daily Close Prices - 5 Different Simulations (T=1000)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Time (Days)', fontsize=12)
    axes[0].set_ylabel('Price', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper left', fontsize=10)
    axes[0].axhline(y=1000, color='gray', linestyle='--', alpha=0.5, label='Initial Price')

    # Add statistics box
    stats_text = "Statistics:\n"
    stats_text += f"{'Sim':<4} {'Final':<7} {'Return':<8} {'Sharpe':<7}\n"
    stats_text += "-" * 35 + "\n"
    for stat in price_stats:
        stats_text += f"{stat['sim']:<4} {stat['final_price']:<7.0f} {stat['total_return']:<7.1f}% {stat['sharpe']:<7.2f}\n"

    axes[0].text(0.02, 0.98, stats_text, transform=axes[0].transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontfamily='monospace')

    # Second plot - Average price with confidence bands
    all_prices_array = np.array(all_prices)
    mean_price = np.mean(all_prices_array, axis=0)
    std_price = np.std(all_prices_array, axis=0)

    axes[1].plot(range(T), mean_price, color='black', linewidth=2, label='Average Price')
    axes[1].fill_between(range(T),
                         mean_price - std_price,
                         mean_price + std_price,
                         color='gray', alpha=0.3, label='±1 Std Dev')
    axes[1].fill_between(range(T),
                         mean_price - 2*std_price,
                         mean_price + 2*std_price,
                         color='gray', alpha=0.15, label='±2 Std Dev')

    # Plot individual simulations faintly
    for i, prices in enumerate(all_prices):
        axes[1].plot(range(T), prices, color=colors[i], alpha=0.2, linewidth=0.5)

    axes[1].set_title('Average Price Path with Confidence Bands', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Time (Days)', fontsize=12)
    axes[1].set_ylabel('Price', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper left', fontsize=10)
    axes[1].axhline(y=1000, color='gray', linestyle='--', alpha=0.5)

    # Add summary statistics for average path
    avg_final = mean_price[-1]
    avg_return = (avg_final / 1000 - 1) * 100

    summary_text = f"Average Final Price: {avg_final:.0f}\n"
    summary_text += f"Average Return: {avg_return:.1f}%\n"
    summary_text += f"Std Dev of Final Prices: {np.std([s['final_price'] for s in price_stats]):.0f}"

    axes[1].text(0.02, 0.98, summary_text, transform=axes[1].transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()

    return fig, all_prices, price_stats


def create_regime_comparison(sim_files, num_sims=5):
    """Create a comparison of regime distributions across simulations"""

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    regime_counts = []

    for i, sim_path in enumerate(sim_files[:num_sims]):
        df = pd.read_parquet(sim_path)
        states = df['s_true'].values

        # Count regime frequencies
        unique, counts = np.unique(states, return_counts=True)
        freq_dict = dict(zip(unique, counts / len(states) * 100))

        regime_counts.append({
            'sim': i + 1,
            'bullish': freq_dict.get(0, 0),
            'neutral': freq_dict.get(1, 0),
            'bearish': freq_dict.get(2, 0)
        })

    # Create bar chart
    x = np.arange(num_sims)
    width = 0.25

    bullish = [r['bullish'] for r in regime_counts]
    neutral = [r['neutral'] for r in regime_counts]
    bearish = [r['bearish'] for r in regime_counts]

    ax.bar(x - width, bullish, width, label='Bullish', color='#2ECC71')
    ax.bar(x, neutral, width, label='Neutral', color='#3498DB')
    ax.bar(x + width, bearish, width, label='Bearish', color='#E74C3C')

    ax.set_xlabel('Simulation', fontsize=12)
    ax.set_ylabel('Regime Frequency (%)', fontsize=12)
    ax.set_title('Regime Distribution Across 5 Simulations', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Sim {i+1}' for i in range(num_sims)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add expected values as horizontal lines
    ax.axhline(y=24.9, color='#2ECC71', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=43.2, color='#3498DB', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=31.8, color='#E74C3C', linestyle='--', alpha=0.5, linewidth=1)

    plt.tight_layout()

    return fig


def main():
    """Main function to create multi-simulation visualization"""

    logger.info("="*80)
    logger.info("VISUALIZING 5 SIMULATIONS - DAILY CLOSE PRICES")
    logger.info("="*80)

    # Load simulation files
    sim_dir = Path('outputs/step2_sim/K3_512_corrected')
    sim_files = sorted(sim_dir.glob('sim_*.parquet'))

    if len(sim_files) < 5:
        logger.error(f"Need at least 5 simulations, found {len(sim_files)}")
        return

    # Create output directory
    output_dir = Path('outputs/visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create price comparison visualization
    fig1, all_prices, price_stats = plot_multiple_simulations(sim_files, num_sims=5)

    # Save price comparison plot
    plot_path1 = output_dir / 'close_prices_5_simulations.png'
    fig1.savefig(plot_path1, dpi=150, bbox_inches='tight')
    plt.close(fig1)

    # Create regime comparison
    fig2 = create_regime_comparison(sim_files, num_sims=5)

    # Save regime comparison plot
    plot_path2 = output_dir / 'regime_distribution_5_simulations.png'
    fig2.savefig(plot_path2, dpi=150, bbox_inches='tight')
    plt.close(fig2)

    # Calculate cross-simulation statistics
    logger.info("\n" + "="*60)
    logger.info("CROSS-SIMULATION STATISTICS")
    logger.info("="*60)

    returns = [s['total_return'] for s in price_stats]
    sharpes = [s['sharpe'] for s in price_stats]
    final_prices = [s['final_price'] for s in price_stats]

    logger.info(f"\nTotal Returns:")
    logger.info(f"  Mean: {np.mean(returns):.1f}%")
    logger.info(f"  Std:  {np.std(returns):.1f}%")
    logger.info(f"  Min:  {np.min(returns):.1f}%")
    logger.info(f"  Max:  {np.max(returns):.1f}%")

    logger.info(f"\nSharpe Ratios:")
    logger.info(f"  Mean: {np.mean(sharpes):.2f}")
    logger.info(f"  Std:  {np.std(sharpes):.2f}")

    logger.info(f"\nFinal Prices:")
    logger.info(f"  Mean: {np.mean(final_prices):.0f}")
    logger.info(f"  Std:  {np.std(final_prices):.0f}")

    logger.info(f"\n✅ Price comparison saved to: {plot_path1}")
    logger.info(f"📊 Regime comparison saved to: {plot_path2}")

    return all_prices, price_stats


if __name__ == "__main__":
    main()