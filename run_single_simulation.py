#!/usr/bin/env python
"""
Run a single HMM simulation
"""
import numpy as np
import pandas as pd
import sys
import os
sys.path.append('.')
from src.utils_io import load_config, load_json, save_parquet, setup_logging

logger = setup_logging('single_simulation')


def simulate_hmm_sequence(pi, P, mu, sigma, T, seed):
    """
    Simulate a single HMM sequence

    Parameters:
    -----------
    pi : np.ndarray
        Initial state distribution (K,)
    P : np.ndarray
        Transition matrix (K, K)
    mu : np.ndarray
        State means (K,)
    sigma : np.ndarray
        State standard deviations (K,)
    T : int
        Sequence length
    seed : int
        Random seed

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: t, y, s_true
    """
    np.random.seed(seed)

    K = len(pi)
    states = np.zeros(T, dtype=int)
    observations = np.zeros(T)

    # Draw initial state s_1 ~ Categorical(pi)
    states[0] = np.random.choice(K, p=pi)

    # For t=2..T, draw s_t ~ Categorical(P[s_{t-1}, :])
    for t in range(1, T):
        states[t] = np.random.choice(K, p=P[states[t-1], :])

    # Draw observations y_t ~ Normal(mu[s_t], sigma[s_t]^2)
    for t in range(T):
        observations[t] = np.random.normal(mu[states[t]], sigma[states[t]])

    # Create DataFrame
    df = pd.DataFrame({
        't': np.arange(T),
        'y': observations,
        's_true': states
    })

    return df


def main():
    """Main execution function"""
    # Load configuration
    config = load_config('configs/daily.yaml')

    # Load HMM parameters
    hmm_params = load_json('outputs/step1_params/hmm_K2.json')

    logger.info("=" * 80)
    logger.info("RUNNING SINGLE SIMULATION")
    logger.info("=" * 80)

    logger.info("\nLoaded HMM parameters:")
    logger.info(f"  K = {hmm_params['K']}")
    logger.info(f"  mu = {hmm_params['mu']}")
    logger.info(f"  sigma = {hmm_params['sigma']}")
    logger.info(f"  pi = {hmm_params['pi']}")

    # Extract parameters
    pi = np.array(hmm_params['pi'])
    P = np.array(hmm_params['P'])
    mu = np.array(hmm_params['mu'])
    sigma = np.array(hmm_params['sigma'])

    T = config['simulation']['T']
    seed0 = config['simulation']['seed0']
    K = hmm_params['K']

    logger.info(f"\nSimulation parameters:")
    logger.info(f"  Sequence length (T) = {T}")
    logger.info(f"  Random seed = {seed0}")

    # Simulate single sequence
    logger.info("\nSimulating sequence...")
    df_sim = simulate_hmm_sequence(pi, P, mu, sigma, T, seed0)

    # Add metadata
    df_sim['sim_id'] = 0
    df_sim['seed'] = seed0
    df_sim['T'] = T
    df_sim['K'] = K

    # Create output directory if needed
    output_dir = f"outputs/step2_sim/K{K}"
    os.makedirs(output_dir, exist_ok=True)

    # Save to parquet
    output_path = f"{output_dir}/sim_T{T:04d}_seed{0:03d}.parquet"
    save_parquet(df_sim, output_path)

    logger.info(f"\n✓ Simulation complete!")
    logger.info(f"  Output saved to: {output_path}")

    # Display summary statistics
    logger.info("\nSimulation summary:")
    logger.info(f"  Total timesteps: {len(df_sim)}")
    logger.info(f"  Observation mean: {df_sim['y'].mean():.6f}")
    logger.info(f"  Observation std: {df_sim['y'].std():.6f}")
    logger.info(f"  State 0 count: {(df_sim['s_true'] == 0).sum()} ({(df_sim['s_true'] == 0).sum()/len(df_sim)*100:.1f}%)")
    logger.info(f"  State 1 count: {(df_sim['s_true'] == 1).sum()} ({(df_sim['s_true'] == 1).sum()/len(df_sim)*100:.1f}%)")

    # Show first few rows
    logger.info("\nFirst 10 rows of simulation:")
    print(df_sim.head(10).to_string(index=False))

    logger.info("\n" + "=" * 80)
    logger.info("DONE!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
