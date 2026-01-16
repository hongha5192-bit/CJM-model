#!/usr/bin/env python
"""
Step 2: Generate 256 feature-based simulations for MD file compliance
Following MD file specifications with N_sim=256, T=1000
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


def simulate_hmm_sequence(pi, P, means_scaled, covs_scaled, scaler_mean, scaler_std, T, seed):
    """Generate a single HMM sequence with features"""
    np.random.seed(seed)
    K = len(pi)
    n_features = len(means_scaled[0])

    states = np.zeros(T, dtype=int)
    features_scaled = np.zeros((T, n_features))

    # Initial state from stationary distribution
    states[0] = np.random.choice(K, p=pi)
    features_scaled[0] = np.random.multivariate_normal(
        means_scaled[states[0]],
        covs_scaled[states[0]]
    )

    # Generate sequence
    for t in range(1, T):
        # State transition
        states[t] = np.random.choice(K, p=P[states[t-1]])
        # Observation from state
        features_scaled[t] = np.random.multivariate_normal(
            means_scaled[states[t]],
            covs_scaled[states[t]]
        )

    # Transform back to original scale
    features = features_scaled * scaler_std + scaler_mean

    return features, states


def main():
    """Generate 256 simulations following MD file specs"""

    logger.info("="*80)
    logger.info("GENERATING 256 FEATURE-BASED SIMULATIONS")
    logger.info("="*80)

    # Load configuration
    config_path = Path('configs/daily_K3.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load HMM parameters
    params_path = Path('outputs/step1_params/hmm_K3_features.json')
    if not params_path.exists():
        logger.error(f"HMM parameters not found at {params_path}")
        logger.info("Please run step1_fit_hmm_K3_features.py first")
        return

    with open(params_path, 'r') as f:
        params = json.load(f)

    K = params['K']
    feature_names = params['features']
    n_features = params['n_features']

    # Get scaled parameters (what HMM uses internally)
    means_scaled = np.array(params['means_matrix'])  # K x n_features
    stds_scaled = np.array(params['stds_scaled'])    # K x n_features

    # Convert stds to covariance matrices (diagonal)
    covs_scaled = []
    for k in range(K):
        cov = np.diag(stds_scaled[k] ** 2)
        covs_scaled.append(cov)

    # Transition matrix and stationary distribution
    P = np.array(params['P'])
    pi = np.array(params['pi'])

    # Scaler parameters for inverse transform
    scaler_mean = np.array(params['scaler_params']['mean'])
    scaler_std = np.array(params['scaler_params']['scale'])

    # Simulation parameters (MD file specs)
    N_sim = 256  # MD file: 256-1024, we use 256
    T = 1000     # Daily data, 1000 days (~4 years)
    seed0 = 1000

    logger.info(f"Configuration:")
    logger.info(f"  K = {K}")
    logger.info(f"  N_sim = {N_sim}")
    logger.info(f"  T = {T}")
    logger.info(f"  Features = {feature_names}")
    logger.info(f"  Output format = parquet")

    # Create output directory
    output_dir = Path(f'outputs/step2_sim/K{K}_256')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear existing simulations
    existing_files = list(output_dir.glob('sim_*.parquet'))
    if existing_files:
        logger.info(f"Clearing {len(existing_files)} existing simulation files...")
        for f in existing_files:
            f.unlink()

    logger.info(f"\nGenerating {N_sim} simulations with T={T}...")

    # Generate simulations with progress bar
    successful = 0
    failed = 0

    start_time = time.time()

    with tqdm(total=N_sim, desc="Simulating", unit="sim") as pbar:
        for sim_idx in range(N_sim):
            try:
                seed = seed0 + sim_idx

                # Generate sequence
                features, states = simulate_hmm_sequence(
                    pi, P, means_scaled, covs_scaled,
                    scaler_mean, scaler_std, T, seed
                )

                # Create DataFrame
                df = pd.DataFrame({
                    't': range(T),
                    's_true': states
                })

                # Add features
                for i, feat_name in enumerate(feature_names):
                    df[feat_name] = features[:, i]

                # Save to parquet
                filename = f'sim_T{T:04d}_seed{seed:04d}.parquet'
                filepath = output_dir / filename
                df.to_parquet(filepath, index=False)

                successful += 1

            except Exception as e:
                logger.debug(f"Failed simulation {sim_idx}: {e}")
                failed += 1

            pbar.update(1)

    elapsed = time.time() - start_time

    # Summary statistics
    logger.info("\n" + "="*80)
    logger.info("SIMULATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Successfully generated: {successful}/{N_sim} simulations")
    if failed > 0:
        logger.info(f"Failed: {failed} simulations")
    logger.info(f"Time elapsed: {elapsed:.1f} seconds")
    logger.info(f"Average time per simulation: {elapsed/N_sim:.3f} seconds")

    # Analyze state distributions
    logger.info("\nState Distribution Analysis:")
    all_state_counts = {k: [] for k in range(K)}

    for filepath in output_dir.glob('sim_*.parquet'):
        df = pd.read_parquet(filepath)
        counts = df['s_true'].value_counts(normalize=True).to_dict()
        for k in range(K):
            all_state_counts[k].append(counts.get(k, 0))

    for k in range(K):
        mean_freq = np.mean(all_state_counts[k])
        std_freq = np.std(all_state_counts[k])
        logger.info(f"  State {k}: {mean_freq:.3f} ± {std_freq:.3f} (expected: {pi[k]:.3f})")

    # Save metadata
    metadata = {
        'N_sim': N_sim,
        'T': T,
        'K': K,
        'features': feature_names,
        'successful': successful,
        'failed': failed,
        'elapsed_seconds': elapsed,
        'timestamp': pd.Timestamp.now().isoformat()
    }

    metadata_path = output_dir / 'simulation_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\nSimulations saved to: {output_dir}")
    logger.info(f"Metadata saved to: {metadata_path}")

    return successful


if __name__ == "__main__":
    main()