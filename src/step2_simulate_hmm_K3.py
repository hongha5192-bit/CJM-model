#!/usr/bin/env python
"""
Step 2: Simulate from fitted HMM K=3 (Fast Mode)
Generates N_sim=256 sequences for lambda selection
"""
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
import yaml
from tqdm import tqdm
from joblib import Parallel, delayed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def simulate_one_sequence(sim_id, T, K, mu, sigma, P, pi, seed):
    """Simulate a single HMM sequence"""
    np.random.seed(seed)

    # Initialize arrays
    states = np.zeros(T, dtype=int)
    observations = np.zeros(T)

    # Sample initial state from stationary distribution
    states[0] = np.random.choice(K, p=pi)
    observations[0] = np.random.normal(mu[states[0]], sigma[states[0]])

    # Generate sequence
    for t in range(1, T):
        # Sample next state from transition matrix
        states[t] = np.random.choice(K, p=P[states[t-1]])
        # Sample observation from state distribution
        observations[t] = np.random.normal(mu[states[t]], sigma[states[t]])

    # Create DataFrame
    df = pd.DataFrame({
        't': np.arange(T),
        'y': observations,
        's_true': states,
        'sim_id': sim_id,
        'seed': seed
    })

    return df


def simulate_hmm_K3(config_path='configs/daily_K3.yaml'):
    """Generate simulations from fitted HMM K=3 parameters"""

    # Load config
    logger.info("Loading configuration...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load HMM parameters
    params_path = Path('outputs/step1_params/hmm_K3.json')
    if not params_path.exists():
        raise FileNotFoundError(f"HMM parameters not found at {params_path}")

    with open(params_path, 'r') as f:
        hmm_params = json.load(f)

    K = hmm_params['K']
    assert K == 3, f"Expected K=3, got K={K}"

    mu = hmm_params['mu']
    sigma = hmm_params['sigma']
    P = np.array(hmm_params['P'])
    pi = np.array(hmm_params['pi'])

    # Simulation parameters (Fast Mode)
    N_sim = config['simulation']['N_sim']
    T = config['simulation']['T']
    seed0 = config['simulation']['seed0']
    file_format = config['simulation'].get('file_format', 'parquet')

    logger.info(f"Simulation parameters (Fast Mode):")
    logger.info(f"  N_sim = {N_sim} sequences")
    logger.info(f"  T = {T} time points each")
    logger.info(f"  Base seed = {seed0}")
    logger.info(f"  File format = {file_format}")

    # Create output directory
    output_dir = Path(f'outputs/step2_sim/K{K}')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear existing simulations
    existing_files = list(output_dir.glob(f'sim_T{T}_*.{file_format}'))
    if existing_files:
        logger.info(f"Removing {len(existing_files)} existing simulation files...")
        for f in existing_files:
            f.unlink()

    # Simulate sequences
    logger.info(f"Generating {N_sim} simulated sequences...")

    # Use parallel processing with conservative settings for Mac Air M4
    n_jobs = min(2, config.get('n_jobs', 2))  # Cap at 2 workers for stability
    logger.info(f"Using {n_jobs} parallel workers")

    # Define simulation function for parallel execution
    def simulate_and_save(i):
        seed = seed0 + i
        sim_id = f"{i:04d}"

        # Simulate
        df = simulate_one_sequence(
            sim_id=sim_id,
            T=T,
            K=K,
            mu=mu,
            sigma=sigma,
            P=P,
            pi=pi,
            seed=seed
        )

        # Save
        if file_format == 'parquet':
            filepath = output_dir / f'sim_T{T}_seed{i:04d}_id{sim_id}.parquet'
            df.to_parquet(filepath, index=False)
        else:
            filepath = output_dir / f'sim_T{T}_seed{i:04d}_id{sim_id}.csv'
            df.to_csv(filepath, index=False)

        return {
            'sim_id': sim_id,
            'seed': seed,
            'mean_y': df['y'].mean(),
            'std_y': df['y'].std(),
            'n_switches': (df['s_true'].diff() != 0).sum() - 1,  # Subtract 1 to exclude first row NaN
            'state_counts': df['s_true'].value_counts().to_dict()
        }

    # Run simulations in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(simulate_and_save)(i)
        for i in tqdm(range(N_sim), desc="Simulating")
    )

    # Analyze simulation statistics
    logger.info("Analyzing simulation statistics...")

    stats_df = pd.DataFrame(results)

    # Calculate state frequencies
    state_freqs = np.zeros((N_sim, K))
    for i, counts in enumerate(stats_df['state_counts']):
        for state, count in counts.items():
            state_freqs[i, state] = count / T

    # Summary statistics
    print("\n" + "="*60)
    print("SIMULATION SUMMARY (K=3, Fast Mode)")
    print("="*60)
    print(f"Generated {N_sim} sequences of length {T}")
    print(f"\nHMM Parameters:")
    print(f"  States: {K}")
    for i in range(K):
        print(f"  State {i}: μ={mu[i]:7.4f}, σ={sigma[i]:7.4f}")

    print(f"\nStationary distribution:")
    for i in range(K):
        print(f"  π[{i}] = {pi[i]:.4f}")

    print(f"\nSimulation Statistics:")
    print(f"  Mean return: {stats_df['mean_y'].mean():.4f} ± {stats_df['mean_y'].std():.4f}")
    print(f"  Mean volatility: {stats_df['std_y'].mean():.4f} ± {stats_df['std_y'].std():.4f}")
    print(f"  Mean switches: {stats_df['n_switches'].mean():.1f} ± {stats_df['n_switches'].std():.1f}")

    print(f"\nState Frequencies (mean ± std):")
    for i in range(K):
        freq_mean = state_freqs[:, i].mean()
        freq_std = state_freqs[:, i].std()
        print(f"  State {i}: {freq_mean:.3f} ± {freq_std:.3f} (expected: {pi[i]:.3f})")

    # Save summary
    summary = {
        'K': K,
        'N_sim': N_sim,
        'T': T,
        'seed0': seed0,
        'hmm_params': hmm_params,
        'statistics': {
            'mean_return': {
                'mean': float(stats_df['mean_y'].mean()),
                'std': float(stats_df['mean_y'].std())
            },
            'volatility': {
                'mean': float(stats_df['std_y'].mean()),
                'std': float(stats_df['std_y'].std())
            },
            'switches': {
                'mean': float(stats_df['n_switches'].mean()),
                'std': float(stats_df['n_switches'].std())
            },
            'state_frequencies': {
                f'state_{i}': {
                    'mean': float(state_freqs[:, i].mean()),
                    'std': float(state_freqs[:, i].std()),
                    'expected': float(pi[i])
                }
                for i in range(K)
            }
        }
    }

    summary_path = output_dir / 'simulation_summary_K3.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved simulation summary to {summary_path}")
    logger.info(f"All {N_sim} simulations saved to {output_dir}")

    print("="*60)

    return summary


if __name__ == "__main__":
    try:
        simulate_hmm_K3()
    except Exception as e:
        logger.error(f"Error in simulation: {e}", exc_info=True)
        raise