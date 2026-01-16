"""
Step 2: Simulate sequences from fitted HMM
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append('.')
from src.utils_io import load_config, load_json, save_parquet, setup_logging

logger = setup_logging('step2_simulate')


def simulate_hmm_sequence(pi, P, mu, sigma, T, seed):
    """
    Simulate a single HMM sequence following the paper methodology

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


def simulate_all_sequences(hmm_params, config):
    """
    Simulate all N_sim sequences

    Parameters:
    -----------
    hmm_params : dict
        HMM parameters from step1
    config : dict
        Configuration dictionary
    """
    # Extract parameters
    pi = np.array(hmm_params['pi'])
    P = np.array(hmm_params['P'])
    mu = np.array(hmm_params['mu'])
    sigma = np.array(hmm_params['sigma'])

    N_sim = config['simulation']['N_sim']
    T = config['simulation']['T']
    seed0 = config['simulation']['seed0']
    K = hmm_params['K']

    logger.info(f"Simulating {N_sim} sequences of length {T}")

    # Simulate each sequence
    for sim_id in tqdm(range(N_sim), desc="Simulating sequences"):
        seed = seed0 + sim_id

        # Simulate sequence
        df_sim = simulate_hmm_sequence(pi, P, mu, sigma, T, seed)

        # Add metadata
        df_sim['sim_id'] = sim_id
        df_sim['seed'] = seed
        df_sim['T'] = T
        df_sim['K'] = K

        # Save to parquet
        output_path = f"outputs/step2_sim/K{K}/sim_T{T:04d}_seed{sim_id:03d}.parquet"
        save_parquet(df_sim, output_path)

    logger.info(f"Completed simulating {N_sim} sequences")


def main():
    """Main execution function"""
    # Load configuration
    config = load_config('configs/daily.yaml')

    # Load HMM parameters
    hmm_params = load_json('outputs/step1_params/hmm_K2.json')

    logger.info("Loaded HMM parameters:")
    logger.info(f"  K = {hmm_params['K']}")
    logger.info(f"  mu = {hmm_params['mu']}")
    logger.info(f"  sigma = {hmm_params['sigma']}")
    logger.info(f"  pi = {hmm_params['pi']}")

    # Simulate sequences
    simulate_all_sequences(hmm_params, config)


if __name__ == "__main__":
    main()