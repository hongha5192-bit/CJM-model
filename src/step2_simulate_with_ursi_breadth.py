"""
Generate price-based simulations with URSI market breadth indicators
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from tqdm import tqdm
import warnings
import time
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_states(pi, P, T, seed=None):
    """Generate state sequence from HMM"""
    if seed is not None:
        np.random.seed(seed)

    K = len(pi)
    states = np.zeros(T, dtype=int)

    # Initial state from pi
    states[0] = np.random.choice(K, p=pi)

    # Generate subsequent states
    for t in range(1, T):
        states[t] = np.random.choice(K, p=P[states[t-1]])

    return states

def generate_ohlc_prices(states, T, initial_price=1000, seed=None):
    """Generate OHLC prices based on regime states using measured gap statistics"""
    if seed is not None:
        np.random.seed(seed + 1000)

    # Regime-specific parameters measured from real HMM-labeled VNINDEX data
    regime_params = {
        0: {  # Bullish
            'ret_mean': 0.002020, 'ret_std': 0.009207,
            'gap_hc_mean': 0.5320, 'gap_hc_std': 0.6825,  # High-Close gap (%)
            'gap_cl_mean': 0.7147, 'gap_cl_std': 0.5947   # Close-Low gap (%)
        },
        1: {  # Neutral
            'ret_mean': 0.000309, 'ret_std': 0.011246,
            'gap_hc_mean': 0.6356, 'gap_hc_std': 0.7930,
            'gap_cl_mean': 0.6544, 'gap_cl_std': 0.6886
        },
        2: {  # Bearish
            'ret_mean': -0.001313, 'ret_std': 0.015930,
            'gap_hc_mean': 0.8062, 'gap_hc_std': 0.9574,
            'gap_cl_mean': 0.9250, 'gap_cl_std': 0.9979
        }
    }

    opens = np.zeros(T)
    highs = np.zeros(T)
    lows = np.zeros(T)
    closes = np.zeros(T)
    returns = np.zeros(T)

    # Initialize
    opens[0] = initial_price
    closes[0] = initial_price * (1 + np.random.normal(regime_params[states[0]]['ret_mean'],
                                                      regime_params[states[0]]['ret_std']))
    highs[0] = max(opens[0], closes[0]) * (1 + abs(np.random.normal(0, 0.005)))
    lows[0] = min(opens[0], closes[0]) * (1 - abs(np.random.normal(0, 0.005)))
    returns[0] = (closes[0] - initial_price) / initial_price

    prev_close = closes[0]

    for t in range(1, T):
        state = states[t]
        params = regime_params[state]

        # 1. Generate daily return
        daily_return = np.random.normal(params['ret_mean'], params['ret_std'])

        # 2. Calculate close price
        closes[t] = prev_close * (1 + daily_return)

        # 3. Set open = close(t-1) (assumption from user)
        opens[t] = prev_close

        # 4. Sample high-close gap (reject negative values)
        gap_hc = -1
        while gap_hc < 0:
            gap_hc = np.random.normal(params['gap_hc_mean'], params['gap_hc_std'])

        # 5. Sample close-low gap (reject negative values)
        gap_cl = -1
        while gap_cl < 0:
            gap_cl = np.random.normal(params['gap_cl_mean'], params['gap_cl_std'])

        # 6. Calculate high and low
        highs[t] = closes[t] * (1 + gap_hc / 100)
        lows[t] = closes[t] * (1 - gap_cl / 100)

        # 7. Enforce OHLC constraints
        highs[t] = max(highs[t], opens[t], closes[t])
        lows[t] = min(lows[t], opens[t], closes[t])

        # 8. Calculate return
        returns[t] = (closes[t] - prev_close) / prev_close

        prev_close = closes[t]

    return opens, highs, lows, closes, returns

def generate_ursi_breadth_features(states, T, seed=None):
    """
    Generate URSI_0_50 and URSI_0_20 based on regime statistics
    Using mean and std from HMM analysis
    """
    if seed is not None:
        np.random.seed(seed + 2000)

    # Statistics from HMM analysis
    ursi_params = {
        0: {  # Bullish
            'ursi_0_50_mean': 17.65, 'ursi_0_50_std': 8.81,
            'ursi_0_20_mean': 1.55, 'ursi_0_20_std': 1.21
        },
        1: {  # Neutral
            'ursi_0_50_mean': 43.34, 'ursi_0_50_std': 14.18,
            'ursi_0_20_mean': 6.90, 'ursi_0_20_std': 5.46
        },
        2: {  # Bearish
            'ursi_0_50_mean': 74.94, 'ursi_0_50_std': 11.74,
            'ursi_0_20_mean': 25.13, 'ursi_0_20_std': 17.78
        }
    }

    ursi_0_50 = np.zeros(T)
    ursi_0_20 = np.zeros(T)

    for t in range(T):
        state = states[t]
        params = ursi_params[state]

        # Generate URSI_0_50 with constraints [0, 100]
        ursi_0_50[t] = np.random.normal(params['ursi_0_50_mean'], params['ursi_0_50_std'])
        ursi_0_50[t] = np.clip(ursi_0_50[t], 0, 100)

        # Generate URSI_0_20 with constraints [0, 100]
        ursi_0_20[t] = np.random.normal(params['ursi_0_20_mean'], params['ursi_0_20_std'])
        ursi_0_20[t] = np.clip(ursi_0_20[t], 0, 100)

        # Add some temporal smoothing to make it more realistic
        if t > 0:
            # 70% current value, 30% previous value for smoothing
            ursi_0_50[t] = 0.7 * ursi_0_50[t] + 0.3 * ursi_0_50[t-1]
            ursi_0_20[t] = 0.7 * ursi_0_20[t] + 0.3 * ursi_0_20[t-1]

            # Ensure constraints after smoothing
            ursi_0_50[t] = np.clip(ursi_0_50[t], 0, 100)
            ursi_0_20[t] = np.clip(ursi_0_20[t], 0, 100)

    return ursi_0_50, ursi_0_20

def calculate_technical_indicators(df):
    """Calculate technical indicators from OHLC data"""

    def rma(series, period):
        """Relative Moving Average (RMA) - Wilder's smoothing"""
        alpha = 1 / period
        return series.ewm(alpha=alpha, adjust=False).mean()

    # 1. Calculate ADX and DMI
    def calculate_adx_dmi(df, period=14):
        """Calculate ADX, DMI_Plus, DMI_Minus"""
        high = df['high']
        low = df['low']
        close = df['close']

        # Calculate directional movement
        plus_dm = high.diff()
        minus_dm = -low.diff()

        # Apply conditions
        plus_dm[(plus_dm < 0) | (plus_dm < minus_dm)] = 0
        minus_dm[(minus_dm < 0) | (minus_dm < plus_dm)] = 0

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

        # Handle division by zero
        ursi = pd.Series(index=close.index, dtype=float)
        for i in range(len(close)):
            if pd.isna(den.iloc[i]) or den.iloc[i] == 0:
                ursi.iloc[i] = 50  # Default to neutral
            else:
                ursi.iloc[i] = 50 + 50 * num.iloc[i] / den.iloc[i]

        return ursi

    # 3. Calculate Bollinger Band indicators
    def calculate_bb_indicators(df, period=20, mult=2, lookback=252):
        """Calculate BB_PCTB and BBWP"""
        close = df['close']

        # Calculate Bollinger Bands
        basis = close.rolling(window=period).mean()
        dev = mult * close.rolling(window=period).std()
        upper = basis + dev
        lower = basis - dev
        bbw = upper - lower

        # BB_PCTB (Percent B)
        bb_pctb = (close - lower) / (upper - lower) * 100
        bb_pctb = bb_pctb.fillna(50)

        # BBWP (Bandwidth Percentile)
        bbwp = pd.Series(index=close.index, dtype=float)
        for i in range(lookback, len(bbw)):
            window = bbw.iloc[i-lookback:i+1]
            valid_window = window.dropna()
            if len(valid_window) > 0:
                bbwp.iloc[i] = (valid_window < bbw.iloc[i]).sum() / len(valid_window) * 100
            else:
                bbwp.iloc[i] = 50

        # Fill initial values
        bbwp.fillna(50, inplace=True)

        return bb_pctb, bbwp

    # Calculate all indicators
    adx, dmi_plus, dmi_minus = calculate_adx_dmi(df)
    ursi = calculate_ursi(df)
    bb_pctb, bbwp = calculate_bb_indicators(df)

    # Add to dataframe
    df['ADX'] = adx.fillna(0)
    df['DMI_Plus'] = dmi_plus.fillna(0)
    df['DMI_Minus'] = dmi_minus.fillna(0)
    df['URSI'] = ursi.fillna(50)
    df['BB_PCTB'] = bb_pctb.fillna(50)
    df['BBWP'] = bbwp.fillna(50)

    return df

def simulate_price_based_hmm_with_ursi(pi, P, T, initial_price=1000, seed=None):
    """Generate a single price-based HMM simulation with all features including URSI breadth"""

    # 1. Generate state sequence
    states = generate_states(pi, P, T, seed)

    # 2. Generate OHLC prices
    opens, highs, lows, closes, returns = generate_ohlc_prices(states, T, initial_price, seed)

    # 3. Generate URSI market breadth features
    ursi_0_50, ursi_0_20 = generate_ursi_breadth_features(states, T, seed)

    # 4. Create DataFrame
    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'returns': returns,
        's_true': states,
        'URSI_0_50': ursi_0_50,
        'URSI_0_20': ursi_0_20
    })

    # 5. Calculate technical indicators from prices
    df = calculate_technical_indicators(df)

    return df

def main():
    """Main function to generate price-based simulations with URSI breadth indicators"""

    logger.info("="*80)
    logger.info("GENERATING PRICE-BASED SIMULATIONS WITH URSI BREADTH INDICATORS")
    logger.info("="*80)

    # Load HMM parameters
    params_path = Path('outputs/step1_params/hmm_K3.json')
    if not params_path.exists():
        logger.error(f"HMM parameters not found at {params_path}")
        return

    with open(params_path, 'r') as f:
        params = json.load(f)

    K = params['K']
    P = np.array(params['P'])
    pi = np.array(params['pi'])

    # Updated feature list - now includes URSI_0_50 and URSI_0_20
    feature_names = ['ADX', 'DMI_Plus', 'DMI_Minus', 'URSI', 'BB_PCTB', 'BBWP', 'URSI_0_50', 'URSI_0_20']

    # Simulation parameters
    N_sim = 100
    T = 1000
    seed0 = 7000  # New seed base for new simulations
    initial_price = 1000

    logger.info(f"Configuration:")
    logger.info(f"  K = {K}")
    logger.info(f"  N_sim = {N_sim}")
    logger.info(f"  T = {T}")
    logger.info(f"  Initial price = {initial_price}")
    logger.info(f"  Features = {feature_names}")
    logger.info(f"  NEW: Including URSI_0_50 and URSI_0_20 market breadth indicators")

    # Create output directory
    output_dir = Path(f'outputs/step2_sim/K{K}_with_ursi_breadth')
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
                df = simulate_price_based_hmm_with_ursi(pi, P, T, initial_price, seed)

                # Update statistics
                for feat in feature_names:
                    if feat in df.columns:
                        vals = df[feat].values
                        valid_vals = vals[~np.isnan(vals)]
                        if len(valid_vals) > 0:
                            feature_stats[feat]['min'] = min(feature_stats[feat]['min'], valid_vals.min())
                            feature_stats[feat]['max'] = max(feature_stats[feat]['max'], valid_vals.max())
                            feature_stats[feat]['sum'] += valid_vals.sum()
                            feature_stats[feat]['sum_sq'] += (valid_vals ** 2).sum()
                            feature_stats[feat]['count'] += len(valid_vals)

                # Track state frequencies
                state_counts = np.bincount(df['s_true'].values, minlength=K)
                state_freq = state_counts / T
                for k in range(K):
                    state_frequencies[k].append(state_freq[k])

                # Save simulation
                output_path = output_dir / f'sim_{sim_idx:04d}.parquet'
                df.to_parquet(output_path, index=False)
                successful += 1

            except Exception as e:
                logger.warning(f"Failed to generate simulation {sim_idx}: {e}")
                failed += 1

            pbar.update(1)

    elapsed_time = time.time() - start_time

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("GENERATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Successful: {successful}/{N_sim}")
    logger.info(f"Failed: {failed}/{N_sim}")
    logger.info(f"Time: {elapsed_time:.1f} seconds ({elapsed_time/N_sim:.2f} sec/sim)")

    # Print feature statistics
    logger.info("\nFeature Statistics:")
    for feat in feature_names:
        if feature_stats[feat]['count'] > 0:
            mean = feature_stats[feat]['sum'] / feature_stats[feat]['count']
            var = (feature_stats[feat]['sum_sq'] / feature_stats[feat]['count']) - mean**2
            std = np.sqrt(max(0, var))
            logger.info(f"  {feat:12s}: min={feature_stats[feat]['min']:7.2f}, "
                       f"max={feature_stats[feat]['max']:7.2f}, "
                       f"mean={mean:7.2f}, std={std:7.2f}")

    # Print state frequencies
    logger.info("\nState Frequency Statistics:")
    for k in range(K):
        if state_frequencies[k]:
            mean_freq = np.mean(state_frequencies[k])
            std_freq = np.std(state_frequencies[k])
            logger.info(f"  State {k}: {mean_freq:.3f} ± {std_freq:.3f}")

    logger.info(f"\nSimulations saved to: {output_dir}")
    logger.info("="*80)

if __name__ == "__main__":
    main()