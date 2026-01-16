#!/usr/bin/env python
"""
Generate price-based simulations with OHLC data and derived technical indicators
This is a more realistic approach where:
1. States determine price returns
2. OHLC prices are generated from returns
3. Technical indicators are calculated from price data
"""
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
import yaml
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_state_sequence(pi, P, T, seed):
    """Generate Markov chain state sequence"""
    np.random.seed(seed)
    K = len(pi)

    states = np.zeros(T, dtype=int)

    # Initial state from stationary distribution
    states[0] = np.random.choice(K, p=pi)

    # Generate sequence
    for t in range(1, T):
        states[t] = np.random.choice(K, p=P[states[t-1]])

    return states


def generate_ohlc_prices(states, T, initial_price=1000, seed=None):
    """
    Generate OHLC prices based on regime states using measured gap statistics

    Algorithm for each day t:
    1. Generate state s_t from HMM transition matrix
    2. Generate return from N(μ_{s_t}, σ_{s_t})
    3. Calculate close_t = close_{t-1} * (1 + return_t)
    4. Set open_t = close_{t-1}
    5. Sample gap_hc ~ |N(μ_hc[s_t], σ_hc[s_t])| (high-close gap in %)
    6. Sample gap_cl ~ |N(μ_cl[s_t], σ_cl[s_t])| (close-low gap in %)
    7. Calculate high_t = close_t * (1 + gap_hc/100)
    8. Calculate low_t = close_t * (1 - gap_cl/100)
    9. Enforce constraints: high ≥ max(open, close), low ≤ min(open, close)

    Returns:
        DataFrame with columns: open, high, low, close, returns, state
    """
    if seed is not None:
        np.random.seed(seed)

    # Regime-specific parameters measured from real HMM-labeled VNINDEX data
    regime_params = {
        0: {  # Bullish
            'ret_mean': 0.002020,      # Daily return mean
            'ret_std': 0.009207,       # Daily return std
            'gap_hc_mean': 0.5320,     # High-Close gap mean (%)
            'gap_hc_std': 0.6825,      # High-Close gap std (%)
            'gap_cl_mean': 0.7147,     # Close-Low gap mean (%)
            'gap_cl_std': 0.5947       # Close-Low gap std (%)
        },
        1: {  # Neutral
            'ret_mean': 0.000309,
            'ret_std': 0.011246,
            'gap_hc_mean': 0.6356,
            'gap_hc_std': 0.7930,
            'gap_cl_mean': 0.6544,
            'gap_cl_std': 0.6886
        },
        2: {  # Bearish
            'ret_mean': -0.001313,
            'ret_std': 0.015930,
            'gap_hc_mean': 0.8062,
            'gap_hc_std': 0.9574,
            'gap_cl_mean': 0.9250,
            'gap_cl_std': 0.9979
        }
    }

    # Initialize arrays
    opens = np.zeros(T)
    highs = np.zeros(T)
    lows = np.zeros(T)
    closes = np.zeros(T)
    returns = np.zeros(T)

    # Set initial price
    prev_close = initial_price

    for t in range(T):
        state = states[t]
        params = regime_params[state]

        # 1. Generate daily return
        daily_return = np.random.normal(params['ret_mean'], params['ret_std'])
        returns[t] = daily_return

        # 2. Calculate close price
        closes[t] = prev_close * (1 + daily_return)

        # 3. Set open = close(t-1)
        opens[t] = prev_close

        # 4. Sample high-close gap (ensure positive by rejection sampling)
        gap_hc = -1
        while gap_hc < 0:
            gap_hc = np.random.normal(params['gap_hc_mean'], params['gap_hc_std'])

        # 5. Sample close-low gap (ensure positive by rejection sampling)
        gap_cl = -1
        while gap_cl < 0:
            gap_cl = np.random.normal(params['gap_cl_mean'], params['gap_cl_std'])

        # 6. Calculate high and low relative to close
        # Note: measured gaps are (high-close)/close and (close-low)/close from real data
        # where high is always >= close and low is always <= close
        highs[t] = closes[t] * (1 + gap_hc / 100)
        lows[t] = closes[t] * (1 - gap_cl / 100)

        # 7. Enforce OHLC constraints to ensure valid candlestick
        # When open != close, we need: high >= max(open, close) and low <= min(open, close)
        highs[t] = max(highs[t], opens[t], closes[t])
        lows[t] = min(lows[t], opens[t], closes[t])

        # Update for next iteration
        prev_close = closes[t]

    # Create DataFrame
    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'returns': returns,
        's_true': states
    })

    return df


def calculate_technical_indicators(df):
    """
    Calculate all technical indicators from OHLC data

    Features to calculate:
    1. ADX - Average Directional Index
    2. URSI - Ultimate RSI
    3. BBWP - Bollinger Band Width Percentile
    4. BB_PCTB - Bollinger Band Percent B
    5. URSI_0_50 - URSI smoothed with 0.50 factor
    6. URSI_0_20 - URSI smoothed with 0.20 factor
    """

    # Helper functions
    def rma(series, length):
        """Rolling Moving Average (Wilder's smoothing)"""
        alpha = 1 / length
        return series.ewm(alpha=alpha, adjust=False).mean()

    def sma(series, length):
        """Simple Moving Average"""
        return series.rolling(window=length).mean()

    def ema(series, length):
        """Exponential Moving Average"""
        return series.ewm(span=length, adjust=False).mean()

    # 1. Calculate ADX and DMI
    def calculate_adx_dmi(df, period=14):
        """Calculate ADX (Average Directional Index) and DMI Plus/Minus"""
        high = df['high']
        low = df['low']
        close = df['close']

        # Calculate directional movement
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        # When both are positive, keep only the larger
        mask = (plus_dm > 0) & (minus_dm > 0)
        plus_dm[mask & (plus_dm < minus_dm)] = 0
        minus_dm[mask & (minus_dm < plus_dm)] = 0

        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Smooth the values
        atr = rma(tr, period)
        plus_di = 100 * rma(plus_dm, period) / atr
        minus_di = 100 * rma(minus_dm, period) / atr

        # Calculate ADX
        sum_di = plus_di + minus_di
        # Handle division by zero as per formula: use 1 when sum is 0
        sum_di = sum_di.replace(0, 1)
        dx = 100 * abs(plus_di - minus_di) / sum_di
        dx = dx.fillna(0)
        adx = rma(dx, period)

        return adx, plus_di, minus_di

    # 2. Calculate URSI (Ultimate RSI)
    def calculate_ursi(df, period=14):
        """Calculate Ultimate RSI"""
        close = df['close']

        # Calculate components
        highest = close.rolling(window=period).max()
        lowest = close.rolling(window=period).min()
        r = highest - lowest

        # Calculate diff
        d = close.diff()

        # Determine diff based on conditions
        diff = pd.Series(index=close.index, dtype=float)
        for i in range(1, len(close)):
            if pd.isna(highest.iloc[i]) or pd.isna(lowest.iloc[i]):
                diff.iloc[i] = np.nan
            elif highest.iloc[i] > highest.iloc[i-1]:
                diff.iloc[i] = r.iloc[i]
            elif lowest.iloc[i] < lowest.iloc[i-1]:
                diff.iloc[i] = -r.iloc[i]
            else:
                diff.iloc[i] = d.iloc[i]

        # Calculate URSI
        num = rma(diff, period)
        den = rma(abs(diff), period)
        ursi = 50 + 50 * (num / den)
        ursi = ursi.replace([np.inf, -np.inf], 50).fillna(50)

        return ursi

    # 3. Calculate Bollinger Bands components
    def calculate_bollinger_bands(df, period=20, std_dev=2):
        """Calculate Bollinger Bands components"""
        close = df['close']

        # Calculate bands
        basis = sma(close, period)
        dev = close.rolling(window=period).std() * std_dev
        upper = basis + dev
        lower = basis - dev

        # BB %B
        bb_pctb = (close - lower) / (upper - lower)
        bb_pctb = bb_pctb.replace([np.inf, -np.inf], 0.5).fillna(0.5)

        # BB Width
        bb_width = (upper - lower) / basis
        bb_width = bb_width.replace([np.inf, -np.inf], 0).fillna(0)

        return bb_pctb, bb_width

    # 4. Calculate BBWP (Bollinger Band Width Percentile)
    def calculate_bbwp(bb_width, lookback=100):
        """Calculate BB Width Percentile"""
        bbwp = bb_width.rolling(window=lookback).apply(
            lambda x: (x.iloc[-1] >= x[:-1]).mean() * 100 if len(x) == lookback else np.nan
        )
        return bbwp.fillna(50)

    # Calculate all indicators
    adx, plus_di, minus_di = calculate_adx_dmi(df)
    df['ADX'] = adx
    df['DMI_Plus'] = plus_di
    df['DMI_Minus'] = minus_di
    df['URSI'] = calculate_ursi(df)

    bb_pctb, bb_width = calculate_bollinger_bands(df)
    df['BB_PCTB'] = bb_pctb
    df['BBWP'] = calculate_bbwp(bb_width)

    # Note: URSI_0_50 and URSI_0_20 are market breadth indicators
    # They cannot be reconstructed from single stock price data
    # We'll skip these features in price-based simulation

    # Fill initial NaN values with reasonable defaults
    df['ADX'] = df['ADX'].fillna(25)
    df['DMI_Plus'] = df['DMI_Plus'].fillna(14)
    df['DMI_Minus'] = df['DMI_Minus'].fillna(14)
    df['URSI'] = df['URSI'].fillna(50)
    df['BBWP'] = df['BBWP'].fillna(50)
    df['BB_PCTB'] = df['BB_PCTB'].fillna(0.5)

    return df


def simulate_price_based_hmm(pi, P, T, initial_price=1000, seed=None):
    """
    Complete simulation: states -> prices -> indicators
    """
    # Generate state sequence
    states = generate_state_sequence(pi, P, T, seed)

    # Generate OHLC prices
    df = generate_ohlc_prices(states, T, initial_price, seed)

    # Calculate technical indicators
    df = calculate_technical_indicators(df)

    return df


def main():
    """Generate price-based simulations with technical indicators"""

    logger.info("="*80)
    logger.info("GENERATING PRICE-BASED SIMULATIONS")
    logger.info("="*80)

    # Load configuration
    config_path = Path('configs/daily_K3.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load HMM parameters
    params_path = Path('outputs/step1_params/hmm_K3_features.json')
    if not params_path.exists():
        logger.error(f"HMM parameters not found at {params_path}")
        return

    with open(params_path, 'r') as f:
        params = json.load(f)

    K = params['K']
    P = np.array(params['P'])
    pi = np.array(params['pi'])
    feature_names = ['ADX', 'DMI_Plus', 'DMI_Minus', 'URSI', 'BBWP', 'BB_PCTB']  # Excluding URSI_0_50, URSI_0_20 (market breadth indicators)

    # Simulation parameters
    N_sim = 100  # Start with 100 for testing
    T = 1000
    seed0 = 6000  # New seed base
    initial_price = 1000

    logger.info(f"Configuration:")
    logger.info(f"  K = {K}")
    logger.info(f"  N_sim = {N_sim}")
    logger.info(f"  T = {T}")
    logger.info(f"  Initial price = {initial_price}")
    logger.info(f"  Features = {feature_names}")

    # Create output directory
    output_dir = Path(f'outputs/step2_sim/K{K}_price_based')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear existing simulations
    existing_files = list(output_dir.glob('sim_*.parquet'))
    if existing_files:
        logger.info(f"Clearing {len(existing_files)} existing files...")
        for f in existing_files:
            f.unlink()

    # Track statistics
    feature_stats = {feat: {'min': float('inf'), 'max': float('-inf'),
                           'sum': 0, 'sum_sq': 0, 'count': 0}
                    for feat in feature_names}
    state_frequencies = [[] for _ in range(K)]
    price_stats = {'min': float('inf'), 'max': float('-inf'),
                  'returns_sum': 0, 'returns_sum_sq': 0}

    # Generate simulations
    logger.info("\nGenerating simulations...")
    successful = 0
    failed = 0

    start_time = time.time()

    with tqdm(total=N_sim, desc="Simulating") as pbar:
        for sim_idx in range(N_sim):
            try:
                # Generate simulation
                seed = seed0 + sim_idx
                df = simulate_price_based_hmm(pi, P, T, initial_price, seed)

                # Update statistics
                for feat in feature_names:
                    values = df[feat].values
                    feature_stats[feat]['min'] = min(feature_stats[feat]['min'], values.min())
                    feature_stats[feat]['max'] = max(feature_stats[feat]['max'], values.max())
                    feature_stats[feat]['sum'] += values.sum()
                    feature_stats[feat]['sum_sq'] += (values ** 2).sum()
                    feature_stats[feat]['count'] += len(values)

                # Price statistics
                price_stats['min'] = min(price_stats['min'], df['close'].min())
                price_stats['max'] = max(price_stats['max'], df['close'].max())
                price_stats['returns_sum'] += df['returns'].sum()
                price_stats['returns_sum_sq'] += (df['returns'] ** 2).sum()

                # State frequencies
                for k in range(K):
                    state_frequencies[k].append((df['s_true'] == k).mean())

                # Save simulation
                filepath = output_dir / f'sim_{sim_idx:04d}.parquet'
                df.to_parquet(filepath, index=False)

                successful += 1

            except Exception as e:
                logger.debug(f"Failed simulation {sim_idx}: {e}")
                failed += 1

            pbar.update(1)

    elapsed = time.time() - start_time

    # Calculate final statistics
    for feat in feature_stats:
        n = feature_stats[feat]['count']
        if n > 0:
            feature_stats[feat]['mean'] = feature_stats[feat]['sum'] / n
            feature_stats[feat]['std'] = np.sqrt(
                feature_stats[feat]['sum_sq'] / n -
                (feature_stats[feat]['sum'] / n) ** 2
            )

    # Summary
    logger.info("\n" + "="*80)
    logger.info("SIMULATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Successfully generated: {successful}/{N_sim} simulations")
    if failed > 0:
        logger.info(f"Failed: {failed} simulations")
    logger.info(f"Time elapsed: {elapsed:.1f} seconds")
    logger.info(f"Average time per simulation: {elapsed/N_sim:.3f} seconds")

    # Feature ranges
    logger.info("\nFeature Ranges Across All Simulations:")
    logger.info(f"{'Feature':10s} | {'Min':>8s} | {'Mean':>8s} | {'Std':>8s} | {'Max':>8s}")
    logger.info("-" * 50)
    for feat in feature_names:
        logger.info(f"{feat:10s} | {feature_stats[feat]['min']:8.2f} | "
                   f"{feature_stats[feat]['mean']:8.2f} | {feature_stats[feat]['std']:8.2f} | "
                   f"{feature_stats[feat]['max']:8.2f}")

    # Price statistics
    logger.info(f"\nPrice Statistics:")
    logger.info(f"  Min price: {price_stats['min']:.2f}")
    logger.info(f"  Max price: {price_stats['max']:.2f}")
    logger.info(f"  Mean return: {price_stats['returns_sum']/(successful*T):.6f}")
    logger.info(f"  Std return: {np.sqrt(price_stats['returns_sum_sq']/(successful*T) - (price_stats['returns_sum']/(successful*T))**2):.6f}")

    # State distribution
    logger.info("\nState Distribution (mean ± std):")
    for k in range(K):
        mean_freq = np.mean(state_frequencies[k])
        std_freq = np.std(state_frequencies[k])
        logger.info(f"  State {k}: {mean_freq:.3f} ± {std_freq:.3f} (expected: {pi[k]:.3f})")

    # Verify first simulation
    logger.info("\nFirst Simulation Sample (rows 100-110):")
    df_sample = pd.read_parquet(list(output_dir.glob('sim_*.parquet'))[0])
    logger.info(df_sample.iloc[100:110].to_string())

    # Save metadata
    metadata = {
        'N_sim': N_sim,
        'T': T,
        'K': K,
        'initial_price': initial_price,
        'features': feature_names,
        'successful': successful,
        'failed': failed,
        'elapsed_seconds': elapsed,
        'feature_stats': feature_stats,
        'price_stats': {
            'min': float(price_stats['min']),
            'max': float(price_stats['max']),
            'mean_return': float(price_stats['returns_sum']/(successful*T)),
            'std_return': float(np.sqrt(price_stats['returns_sum_sq']/(successful*T) -
                                       (price_stats['returns_sum']/(successful*T))**2))
        },
        'state_distributions': {
            f'state_{k}': {
                'mean': float(np.mean(state_frequencies[k])),
                'std': float(np.std(state_frequencies[k])),
                'expected': float(pi[k])
            } for k in range(K)
        },
        'timestamp': pd.Timestamp.now().isoformat()
    }

    metadata_path = output_dir / 'simulation_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\n✅ Simulations saved to: {output_dir}")
    logger.info(f"📊 Metadata saved to: {metadata_path}")

    return successful


if __name__ == "__main__":
    main()