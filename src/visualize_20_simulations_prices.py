#!/usr/bin/env python
"""
Visualize daily close prices from 20 different simulations
Shows price diversity and patterns across multiple simulations
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import logging
import seaborn as sns

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


def plot_20_simulations(sim_files, num_sims=20):
    """Create comprehensive visualization of 20 simulations"""

    # Create figure with subplots
    fig = plt.figure(figsize=(18, 14))

    # Create grid layout
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1.5, 1], hspace=0.3, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, :])  # Main price chart (full width)
    ax2 = fig.add_subplot(gs[1, 0])  # Distribution of final prices
    ax3 = fig.add_subplot(gs[1, 1])  # Return distribution
    ax4 = fig.add_subplot(gs[2, :])  # Percentile bands

    # Color palette - use gradient for 20 simulations
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, num_sims))

    # Store price data and statistics
    all_prices = []
    all_returns = []
    final_prices = []
    total_returns = []

    logger.info(f"Loading {num_sims} simulations...")

    for i, sim_path in enumerate(sim_files[:num_sims]):
        # Load simulation
        df = pd.read_parquet(sim_path)

        # Extract features and states
        feature_cols = ['ADX', 'URSI', 'BBWP', 'BB_PCTB', 'URSI_0_50', 'URSI_0_20']
        features = df[feature_cols].values
        states = df['s_true'].values
        T = len(df)

        # Generate synthetic prices
        seed = 2000 + i  # Different seed for each simulation
        prices = generate_synthetic_prices(states, features, T, initial_price=1000, seed=seed)

        all_prices.append(prices)
        final_prices.append(prices[-1])
        total_return = (prices[-1] / prices[0] - 1) * 100
        total_returns.append(total_return)

        # Calculate daily returns
        daily_returns = np.diff(prices) / prices[:-1]
        all_returns.extend(daily_returns)

        # Plot on main axis with varying alpha based on performance
        alpha = 0.3 + 0.3 * (i / num_sims)  # Vary alpha for visual clarity
        ax1.plot(range(T), prices, color=colors[i], alpha=alpha, linewidth=0.8)

        if i % 5 == 0:  # Log every 5th simulation
            logger.info(f"  Sim {i+1:2d}: Final={prices[-1]:6.0f}, Return={total_return:6.1f}%")

    # Convert to arrays
    all_prices_array = np.array(all_prices)

    # Plot 1: Main price chart with statistics
    ax1.axhline(y=1000, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # Add mean path
    mean_price = np.mean(all_prices_array, axis=0)
    ax1.plot(range(T), mean_price, color='red', linewidth=2.5, label='Mean Path', zorder=100)

    # Add median path
    median_price = np.median(all_prices_array, axis=0)
    ax1.plot(range(T), median_price, color='black', linewidth=2, label='Median Path',
             linestyle='--', zorder=100)

    ax1.set_title(f'Daily Close Prices - {num_sims} Simulations (T=1000)',
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time (Days)', fontsize=12)
    ax1.set_ylabel('Price', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=10)

    # Add text box with summary statistics
    stats_text = f"Final Prices:\n"
    stats_text += f"Mean: {np.mean(final_prices):.0f}\n"
    stats_text += f"Median: {np.median(final_prices):.0f}\n"
    stats_text += f"Std: {np.std(final_prices):.0f}\n"
    stats_text += f"Min: {np.min(final_prices):.0f}\n"
    stats_text += f"Max: {np.max(final_prices):.0f}"

    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # Plot 2: Distribution of final prices
    ax2.hist(final_prices, bins=15, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(x=1000, color='red', linestyle='--', alpha=0.7, label='Initial Price')
    ax2.axvline(x=np.mean(final_prices), color='green', linestyle='-',
                linewidth=2, label=f'Mean: {np.mean(final_prices):.0f}')
    ax2.set_xlabel('Final Price', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Distribution of Final Prices', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Return distribution
    ax3.hist(total_returns, bins=15, color='coral', alpha=0.7, edgecolor='black')
    ax3.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    ax3.axvline(x=np.mean(total_returns), color='green', linestyle='-',
                linewidth=2, label=f'Mean: {np.mean(total_returns):.1f}%')

    # Add normal distribution overlay
    from scipy import stats
    x = np.linspace(min(total_returns), max(total_returns), 100)
    ax3_twin = ax3.twinx()
    ax3_twin.plot(x, stats.norm.pdf(x, np.mean(total_returns), np.std(total_returns)),
                  'r-', linewidth=2, alpha=0.6, label='Normal Fit')

    ax3.set_xlabel('Total Return (%)', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Distribution of Returns', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Percentile bands over time
    percentiles = [10, 25, 50, 75, 90]
    percentile_values = np.percentile(all_prices_array, percentiles, axis=0)

    ax4.fill_between(range(T), percentile_values[0], percentile_values[4],
                     color='blue', alpha=0.1, label='10-90th percentile')
    ax4.fill_between(range(T), percentile_values[1], percentile_values[3],
                     color='blue', alpha=0.2, label='25-75th percentile')
    ax4.plot(range(T), percentile_values[2], color='darkblue', linewidth=2,
             label='Median (50th)')

    ax4.axhline(y=1000, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Time (Days)', fontsize=12)
    ax4.set_ylabel('Price', fontsize=12)
    ax4.set_title('Price Percentile Bands', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper left', fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f'Comprehensive Analysis of {num_sims} Simulated Price Paths',
                 fontsize=16, fontweight='bold', y=1.02)

    return fig, all_prices, total_returns


def create_performance_heatmap(sim_files, num_sims=20):
    """Create a heatmap showing performance metrics across simulations"""

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # Prepare data for heatmap
    metrics_data = []

    for i, sim_path in enumerate(sim_files[:num_sims]):
        df = pd.read_parquet(sim_path)

        # Extract features and states
        feature_cols = ['ADX', 'URSI', 'BBWP', 'BB_PCTB', 'URSI_0_50', 'URSI_0_20']
        features = df[feature_cols].values
        states = df['s_true'].values
        T = len(df)

        # Generate prices
        seed = 2000 + i
        prices = generate_synthetic_prices(states, features, T, seed=seed)

        # Calculate metrics
        daily_returns = np.diff(prices) / prices[:-1]

        # Calculate regime frequencies
        bullish_freq = (states == 0).sum() / T * 100
        neutral_freq = (states == 1).sum() / T * 100
        bearish_freq = (states == 2).sum() / T * 100

        metrics_data.append([
            (prices[-1] / prices[0] - 1) * 100,  # Total return
            daily_returns.std() * np.sqrt(252) * 100,  # Annualized volatility
            daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0,  # Sharpe
            bullish_freq,
            neutral_freq,
            bearish_freq
        ])

    # Create DataFrame for heatmap
    metrics_df = pd.DataFrame(metrics_data,
                              columns=['Return %', 'Ann. Vol %', 'Sharpe',
                                      'Bullish %', 'Neutral %', 'Bearish %'],
                              index=[f'Sim {i+1}' for i in range(num_sims)])

    # Create heatmap
    sns.heatmap(metrics_df.T, annot=False, cmap='RdYlGn', center=0,
                cbar_kws={'label': 'Value'}, ax=ax)

    ax.set_title('Performance Metrics Heatmap - 20 Simulations', fontsize=14, fontweight='bold')
    ax.set_xlabel('Simulation', fontsize=12)
    ax.set_ylabel('Metric', fontsize=12)

    plt.tight_layout()

    return fig, metrics_df


def main():
    """Main function to create 20-simulation visualization"""

    logger.info("="*80)
    logger.info("VISUALIZING 20 SIMULATIONS - DAILY CLOSE PRICES")
    logger.info("="*80)

    # Load simulation files
    sim_dir = Path('outputs/step2_sim/K3_512_corrected')
    sim_files = sorted(sim_dir.glob('sim_*.parquet'))

    if len(sim_files) < 20:
        logger.error(f"Need at least 20 simulations, found {len(sim_files)}")
        return

    # Create output directory
    output_dir = Path('outputs/visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create main visualization
    fig1, all_prices, total_returns = plot_20_simulations(sim_files, num_sims=20)

    # Save main plot
    plot_path1 = output_dir / 'close_prices_20_simulations.png'
    fig1.savefig(plot_path1, dpi=150, bbox_inches='tight')
    plt.close(fig1)

    # Create performance heatmap
    fig2, metrics_df = create_performance_heatmap(sim_files, num_sims=20)

    # Save heatmap
    plot_path2 = output_dir / 'performance_heatmap_20_simulations.png'
    fig2.savefig(plot_path2, dpi=150, bbox_inches='tight')
    plt.close(fig2)

    # Calculate and display statistics
    logger.info("\n" + "="*60)
    logger.info("SUMMARY STATISTICS - 20 SIMULATIONS")
    logger.info("="*60)

    logger.info(f"\nReturns Distribution:")
    logger.info(f"  Mean:   {np.mean(total_returns):6.1f}%")
    logger.info(f"  Median: {np.median(total_returns):6.1f}%")
    logger.info(f"  Std:    {np.std(total_returns):6.1f}%")
    logger.info(f"  Min:    {np.min(total_returns):6.1f}%")
    logger.info(f"  Max:    {np.max(total_returns):6.1f}%")
    logger.info(f"  Positive: {sum(r > 0 for r in total_returns)}/20 ({sum(r > 0 for r in total_returns)/20*100:.0f}%)")

    # Calculate percentiles
    percentiles = np.percentile(total_returns, [10, 25, 50, 75, 90])
    logger.info(f"\nReturn Percentiles:")
    logger.info(f"  10th: {percentiles[0]:6.1f}%")
    logger.info(f"  25th: {percentiles[1]:6.1f}%")
    logger.info(f"  50th: {percentiles[2]:6.1f}%")
    logger.info(f"  75th: {percentiles[3]:6.1f}%")
    logger.info(f"  90th: {percentiles[4]:6.1f}%")

    logger.info(f"\n✅ Main visualization saved to: {plot_path1}")
    logger.info(f"📊 Performance heatmap saved to: {plot_path2}")

    # Save metrics to CSV
    metrics_path = output_dir / 'metrics_20_simulations.csv'
    metrics_df.to_csv(metrics_path)
    logger.info(f"📈 Metrics saved to: {metrics_path}")

    return all_prices, total_returns, metrics_df


if __name__ == "__main__":
    main()