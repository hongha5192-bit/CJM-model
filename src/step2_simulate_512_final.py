#!/usr/bin/env python
"""
Generate 512 feature-based simulations - FINAL CORRECT VERSION
Using the actual HMM parameters with proper scaling
"""
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
import yaml
from tqdm import tqdm
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def simulate_hmm_sequence(pi, P, means_by_state, covs_by_state, T, seed):
    """Generate HMM sequence with features"""
    np.random.seed(seed)
    K = len(pi)
    n_features = len(means_by_state[0])

    states = np.zeros(T, dtype=int)
    features = np.zeros((T, n_features))

    # Initial state from stationary distribution
    states[0] = np.random.choice(K, p=pi)
    features[0] = np.random.multivariate_normal(
        means_by_state[states[0]],
        covs_by_state[states[0]]
    )

    # Generate sequence
    for t in range(1, T):
        # State transition
        states[t] = np.random.choice(K, p=P[states[t-1]])
        # Generate features
        features[t] = np.random.multivariate_normal(
            means_by_state[states[t]],
            covs_by_state[states[t]]
        )

    return features, states


def main():
    """Generate 512 simulations with correct parameters"""

    logger.info("="*80)
    logger.info("GENERATING 512 SIMULATIONS - FINAL VERSION")
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
    feature_names = params['features']

    # Get means in original scale (these are already correct!)
    means_by_state = []
    for k in range(K):
        state_means = [
            params['means'][f'state_{k}']['ADX'],
            params['means'][f'state_{k}']['URSI'],
            params['means'][f'state_{k}']['BBWP'],
            params['means'][f'state_{k}']['BB_PCTB'],
            params['means'][f'state_{k}']['URSI_0_50'],
            params['means'][f'state_{k}']['URSI_0_20']
        ]
        means_by_state.append(state_means)

    # Get covariances (need to construct from scaled stds)
    stds_scaled = np.array(params['stds_scaled'])  # K x n_features
    scaler_std = np.array(params['scaler_params']['scale'])

    # Transform scaled stds to original scale
    covs_by_state = []
    for k in range(K):
        # Convert scaled std to original scale
        stds_original = stds_scaled[k] * scaler_std
        # Create diagonal covariance matrix
        cov = np.diag(stds_original ** 2)
        covs_by_state.append(cov)

    # Transition matrix and stationary distribution
    P = np.array(params['P'])
    pi = np.array(params['pi'])

    # Simulation parameters
    N_sim = 512
    T = 1000
    seed0 = 4000  # New seed base

    logger.info(f"Configuration:")
    logger.info(f"  K = {K}")
    logger.info(f"  N_sim = {N_sim}")
    logger.info(f"  T = {T}")
    logger.info(f"  Features = {feature_names}")

    # Display the correct parameter ranges
    logger.info("\nHMM Parameters (Original Scale):")
    for k in range(K):
        logger.info(f"State {k}:")
        for j, feat in enumerate(feature_names):
            mean_val = means_by_state[k][j]
            std_val = np.sqrt(covs_by_state[k][j, j])
            logger.info(f"  {feat:10s}: μ={mean_val:6.2f}, σ={std_val:6.2f}")

    # Create output directory
    output_dir = Path(f'outputs/step2_sim/K{K}_512_final')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear existing simulations
    existing_files = list(output_dir.glob('sim_*.parquet'))
    if existing_files:
        logger.info(f"Clearing {len(existing_files)} existing files...")
        for f in existing_files:
            f.unlink()

    logger.info(f"\nGenerating {N_sim} simulations...")

    successful = 0
    failed = 0
    start_time = time.time()

    # Track statistics
    feature_ranges = {feat: {'min': float('inf'), 'max': float('-inf'), 'mean': 0}
                     for feat in feature_names}
    state_frequencies = {k: [] for k in range(K)}

    with tqdm(total=N_sim, desc="Simulating", unit="sim") as pbar:
        for sim_idx in range(N_sim):
            try:
                seed = seed0 + sim_idx

                # Generate sequence
                features, states = simulate_hmm_sequence(
                    pi, P, means_by_state, covs_by_state, T, seed
                )

                # Create DataFrame
                df = pd.DataFrame({
                    't': range(T),
                    's_true': states
                })

                # Add features with reasonable bounds
                for i, feat_name in enumerate(feature_names):
                    values = features[:, i]

                    # Apply reasonable bounds based on feature type
                    if feat_name == 'BB_PCTB':
                        # BB_PCTB can be outside [0,1] but typically in [-0.5, 1.5]
                        values = np.clip(values, -0.5, 1.5)
                    elif feat_name in ['ADX']:
                        # ADX typically 0-60
                        values = np.clip(values, 0, 60)
                    elif feat_name in ['URSI', 'BBWP']:
                        # Percentile-based, 0-100
                        values = np.clip(values, 0, 100)
                    elif feat_name == 'URSI_0_50':
                        # Market breadth 0-100
                        values = np.clip(values, 0, 100)
                    elif feat_name == 'URSI_0_20':
                        # Extreme breadth 0-50
                        values = np.clip(values, 0, 50)

                    df[feat_name] = values

                    # Track statistics
                    feature_ranges[feat_name]['min'] = min(feature_ranges[feat_name]['min'], values.min())
                    feature_ranges[feat_name]['max'] = max(feature_ranges[feat_name]['max'], values.max())
                    feature_ranges[feat_name]['mean'] += values.mean() / N_sim

                # Track state frequencies
                state_counts = df['s_true'].value_counts(normalize=True).to_dict()
                for k in range(K):
                    state_frequencies[k].append(state_counts.get(k, 0))

                # Save to parquet
                filename = f'sim_T{T:04d}_seed{seed:05d}.parquet'
                filepath = output_dir / filename
                df.to_parquet(filepath, index=False)

                successful += 1

            except Exception as e:
                logger.debug(f"Failed simulation {sim_idx}: {e}")
                failed += 1

            pbar.update(1)

    elapsed = time.time() - start_time

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
    logger.info(f"{'Feature':10s} | {'Min':>8s} | {'Mean':>8s} | {'Max':>8s}")
    logger.info("-" * 45)
    for feat in feature_names:
        logger.info(f"{feat:10s} | {feature_ranges[feat]['min']:8.2f} | "
                   f"{feature_ranges[feat]['mean']:8.2f} | {feature_ranges[feat]['max']:8.2f}")

    # State distribution
    logger.info("\nState Distribution (mean ± std):")
    for k in range(K):
        mean_freq = np.mean(state_frequencies[k])
        std_freq = np.std(state_frequencies[k])
        logger.info(f"  State {k}: {mean_freq:.3f} ± {std_freq:.3f} (expected: {pi[k]:.3f})")

    # Verify first simulation
    logger.info("\nFirst Simulation Sample:")
    df_sample = pd.read_parquet(list(output_dir.glob('sim_*.parquet'))[0])
    logger.info(df_sample.head(10).to_string())

    # Save metadata
    metadata = {
        'N_sim': N_sim,
        'T': T,
        'K': K,
        'features': feature_names,
        'successful': successful,
        'failed': failed,
        'elapsed_seconds': elapsed,
        'feature_ranges': feature_ranges,
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