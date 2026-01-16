"""
Generate price-based simulations with CVD_20 as additional feature
Including proper standardization for all features
"""
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_technical_indicators(prices, volumes):
    """Calculate technical indicators including 20-bar CVD"""

    df = pd.DataFrame({
        'Open': prices[:, 0],
        'High': prices[:, 1],
        'Low': prices[:, 2],
        'Close': prices[:, 3],
        'Volume': volumes
    })

    # Calculate returns
    df['returns'] = df['Close'].pct_change()

    # 1. ADX (Average Directional Index)
    high = df['High']
    low = df['Low']
    close = df['Close']

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)

    atr = tr.rolling(window=14).mean()
    plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    df['ADX'] = dx.rolling(window=14).mean()
    df['DMI_Plus'] = plus_di
    df['DMI_Minus'] = minus_di

    # 2. URSI (Ultimate RSI)
    window = 14
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['URSI'] = 100 - (100 / (1 + rs))

    # 3. Bollinger Bands
    sma20 = close.rolling(window=20).mean()
    std20 = close.rolling(window=20).std()
    upper_band = sma20 + (2 * std20)
    lower_band = sma20 - (2 * std20)

    df['BB_PCTB'] = (close - lower_band) / (upper_band - lower_band)

    # 4. BBWP (Bollinger Band Width Percentile)
    bb_width = upper_band - lower_band
    lookback = 252
    df['BBWP'] = bb_width.rolling(window=lookback).rank(pct=True) * 100

    # 5. Volume Delta using Intrabar Pressure Method
    df['Volume_Delta'] = 0.0
    for i in range(len(df)):
        if df['Volume'].iloc[i] > 0:
            high_val = df['High'].iloc[i]
            low_val = df['Low'].iloc[i]
            close_val = df['Close'].iloc[i]
            open_val = df['Open'].iloc[i]
            volume_val = df['Volume'].iloc[i]

            # Intrabar pressure calculation
            if high_val - low_val > 0:
                lower_range = close_val - low_val
                upper_range = high_val - close_val

                if lower_range > upper_range:
                    # Buying pressure
                    df.loc[i, 'Volume_Delta'] = volume_val
                elif lower_range < upper_range:
                    # Selling pressure
                    df.loc[i, 'Volume_Delta'] = -volume_val
                else:
                    # Use close vs open as tiebreaker
                    if close_val > open_val:
                        df.loc[i, 'Volume_Delta'] = volume_val
                    else:
                        df.loc[i, 'Volume_Delta'] = -volume_val

    # 6. CVD_20 (20-bar rolling sum of Volume Delta)
    df['CVD_20'] = df['Volume_Delta'].rolling(window=20, min_periods=1).sum()

    # Fill initial NaNs with reasonable values
    df['ADX'].fillna(20, inplace=True)
    df['DMI_Plus'].fillna(25, inplace=True)
    df['DMI_Minus'].fillna(25, inplace=True)
    df['URSI'].fillna(50, inplace=True)
    df['BB_PCTB'].fillna(0.5, inplace=True)
    df['BBWP'].fillna(50, inplace=True)
    df['CVD_20'].fillna(0, inplace=True)

    return df

def generate_ohlc_from_close(close_prices, daily_ranges, gap_stats):
    """Generate realistic OHLC from close prices"""

    T = len(close_prices)
    ohlc = np.zeros((T, 4))

    for t in range(T):
        close = close_prices[t]
        daily_range = daily_ranges[t]

        if t == 0:
            open_price = close * (1 + np.random.normal(0, 0.001))
        else:
            gap_mean, gap_std = gap_stats
            gap = np.random.normal(gap_mean, gap_std)
            open_price = close_prices[t-1] * (1 + gap)

        high = max(open_price, close) + daily_range * np.random.uniform(0.3, 0.7)
        low = min(open_price, close) - daily_range * np.random.uniform(0.3, 0.7)

        high = max(high, open_price, close)
        low = min(low, open_price, close)

        ohlc[t] = [open_price, high, low, close]

    return ohlc

def simulate_one_price_based(params, config, sim_id=0):
    """Simulate one path with technical indicators including CVD_20"""

    K = params['K']
    T = config['simulation']['T']

    # Initialize
    states = np.zeros(T, dtype=int)
    prices = np.zeros((T, 4))  # OHLC
    volumes = np.zeros(T)

    # Initial state from stationary distribution
    pi = np.array(params['pi'])
    states[0] = np.random.choice(K, p=pi)

    # Initial price
    initial_price = 1000.0
    close_prices = np.zeros(T)
    close_prices[0] = initial_price

    # Generate base volumes (simplified - could be more sophisticated)
    base_volume = 1e8  # 100 million base volume

    # State-dependent parameters
    state_params = {
        0: {'mu': 0.002, 'sigma': 0.015, 'gap_mean': 0.001, 'gap_std': 0.003, 'vol_mult': 1.2},  # Bullish
        1: {'mu': 0.0001, 'sigma': 0.010, 'gap_mean': 0.0, 'gap_std': 0.002, 'vol_mult': 1.0},    # Neutral
        2: {'mu': -0.002, 'sigma': 0.020, 'gap_mean': -0.001, 'gap_std': 0.004, 'vol_mult': 1.3}  # Bearish
    }

    # Generate state sequence using transition matrix
    P = np.array(params['P'])
    for t in range(1, T):
        states[t] = np.random.choice(K, p=P[states[t-1]])

    # Generate returns based on states
    returns = np.zeros(T)
    daily_ranges = np.zeros(T)

    for t in range(T):
        state = states[t]
        sp = state_params[state]

        # Generate return
        returns[t] = np.random.normal(sp['mu'], sp['sigma'])

        # Generate volume with state-dependent multiplier
        volumes[t] = base_volume * sp['vol_mult'] * np.random.lognormal(0, 0.3)

        # Calculate daily range
        daily_ranges[t] = abs(returns[t]) * initial_price * np.random.uniform(0.8, 1.5)

        # Update close price
        if t > 0:
            close_prices[t] = close_prices[t-1] * (1 + returns[t])

    # Generate OHLC for each state period
    for state in range(K):
        state_mask = states == state
        if np.sum(state_mask) > 0:
            sp = state_params[state]
            gap_stats = (sp['gap_mean'], sp['gap_std'])

            state_indices = np.where(state_mask)[0]
            for idx in state_indices:
                if idx == 0:
                    open_price = close_prices[0]
                else:
                    gap = np.random.normal(gap_stats[0], gap_stats[1])
                    open_price = close_prices[idx-1] * (1 + gap)

                close = close_prices[idx]
                daily_range = daily_ranges[idx]

                high = max(open_price, close) + daily_range * np.random.uniform(0.3, 0.7)
                low = min(open_price, close) - daily_range * np.random.uniform(0.3, 0.7)

                prices[idx] = [open_price, high, low, close]

    # Calculate technical indicators including CVD_20
    df = calculate_technical_indicators(prices, volumes)

    # Add true states and returns
    df['true_state'] = states
    df['sim_id'] = sim_id

    # Fill the first return value with 0 (no previous day)
    df['returns'].fillna(0, inplace=True)

    return df

def main():
    """Generate simulations with CVD_20 feature"""

    logger.info("="*80)
    logger.info("GENERATING SIMULATIONS WITH CVD_20 FEATURE")
    logger.info("="*80)

    # Load HMM parameters
    params_path = Path('outputs/step1_params/hmm_K3.json')
    with open(params_path, 'r') as f:
        params = json.load(f)

    # Load config
    config_path = Path('configs/daily_K3.yaml')
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Simulation parameters
    N_sim = 100  # Number of simulations
    T = config['simulation']['T']

    logger.info(f"\nGenerating {N_sim} simulations with T={T} days each...")
    logger.info("Features: ADX, DMI_Plus, DMI_Minus, URSI, BB_PCTB, BBWP, URSI_0_50, URSI_0_20, CVD_20")

    # Create output directory
    output_dir = Path('outputs/step2_sim/K3_with_cvd')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate URSI breadth statistics (from previous analysis)
    ursi_stats = {
        0: {'ursi_0_50_mean': 17.81, 'ursi_0_50_std': 8.81,
            'ursi_0_20_mean': 1.56, 'ursi_0_20_std': 1.21},  # Bullish
        1: {'ursi_0_50_mean': 43.41, 'ursi_0_50_std': 14.18,
            'ursi_0_20_mean': 6.93, 'ursi_0_20_std': 5.46},  # Neutral
        2: {'ursi_0_50_mean': 75.03, 'ursi_0_50_std': 11.74,
            'ursi_0_20_mean': 25.22, 'ursi_0_20_std': 17.78}  # Bearish
    }

    # Generate simulations
    all_stats = []

    for sim_id in tqdm(range(N_sim)):
        # Generate base simulation with CVD
        sim_df = simulate_one_price_based(params, config, sim_id)

        # Add URSI breadth features based on state
        sim_df['URSI_0_50'] = 0.0
        sim_df['URSI_0_20'] = 0.0

        for state in range(params['K']):
            state_mask = sim_df['true_state'] == state
            n_state = state_mask.sum()

            if n_state > 0:
                # Generate URSI breadth values
                stats = ursi_stats[state]
                ursi_0_50 = np.random.normal(stats['ursi_0_50_mean'],
                                           stats['ursi_0_50_std'], n_state)
                ursi_0_20 = np.random.normal(stats['ursi_0_20_mean'],
                                           stats['ursi_0_20_std'], n_state)

                # Clip to valid range [0, 100]
                ursi_0_50 = np.clip(ursi_0_50, 0, 100)
                ursi_0_20 = np.clip(ursi_0_20, 0, 100)

                sim_df.loc[state_mask, 'URSI_0_50'] = ursi_0_50
                sim_df.loc[state_mask, 'URSI_0_20'] = ursi_0_20

        # Smooth transitions between states
        sim_df['URSI_0_50'] = sim_df['URSI_0_50'].rolling(5, min_periods=1).mean()
        sim_df['URSI_0_20'] = sim_df['URSI_0_20'].rolling(5, min_periods=1).mean()

        # IMPORTANT: Standardize CVD_20 before saving
        # We'll standardize all features during fitting, but let's also save the raw CVD_20 stats
        cvd_stats = {
            'mean': sim_df['CVD_20'].mean(),
            'std': sim_df['CVD_20'].std()
        }

        # Save simulation
        output_file = output_dir / f'sim_{sim_id:04d}.parquet'
        sim_df.to_parquet(output_file)

        # Collect statistics
        all_stats.append({
            'sim_id': sim_id,
            'cvd_20_mean': cvd_stats['mean'],
            'cvd_20_std': cvd_stats['std'],
            'cvd_20_min': sim_df['CVD_20'].min(),
            'cvd_20_max': sim_df['CVD_20'].max(),
            'state_0_pct': (sim_df['true_state'] == 0).mean(),
            'state_1_pct': (sim_df['true_state'] == 1).mean(),
            'state_2_pct': (sim_df['true_state'] == 2).mean()
        })

    # Save simulation statistics
    stats_df = pd.DataFrame(all_stats)
    stats_df.to_csv(output_dir / 'simulation_stats.csv', index=False)

    logger.info(f"\n✓ Generated {N_sim} simulations")
    logger.info(f"✓ Saved to: {output_dir}")

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("SIMULATION STATISTICS")
    logger.info("="*80)

    logger.info("\nCVD_20 Statistics across simulations:")
    logger.info(f"  Mean CVD_20 mean: {stats_df['cvd_20_mean'].mean():.2f}")
    logger.info(f"  Mean CVD_20 std: {stats_df['cvd_20_std'].mean():.2f}")
    logger.info(f"  Overall CVD_20 range: [{stats_df['cvd_20_min'].min():.2f}, {stats_df['cvd_20_max'].max():.2f}]")

    logger.info("\nState distribution:")
    logger.info(f"  Bullish (State 0): {stats_df['state_0_pct'].mean():.1%}")
    logger.info(f"  Neutral (State 1): {stats_df['state_1_pct'].mean():.1%}")
    logger.info(f"  Bearish (State 2): {stats_df['state_2_pct'].mean():.1%}")

    logger.info("\n" + "="*80)

    return stats_df

if __name__ == "__main__":
    stats = main()