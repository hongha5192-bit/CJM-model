#!/usr/bin/env python
"""
Step 1: Fit Gaussian HMM with K=3 for simulation parameters
"""
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
import yaml
from hmmlearn.hmm import GaussianHMM

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def fit_hmm_K3(config_path='configs/daily_K3.yaml'):
    """Fit HMM with K=3 states"""

    # Load config
    logger.info("Loading configuration...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    K = config['hmm']['K']
    assert K == 3, f"Expected K=3, got K={K}"

    # Load prepared data
    data_path = Path('outputs/prepared_data.csv')
    if not data_path.exists():
        raise FileNotFoundError(f"Prepared data not found at {data_path}")

    df = pd.read_csv(data_path, parse_dates=['date'])
    logger.info(f"Loaded {len(df)} rows of data")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Get returns
    ret = df['ret'].values
    ret_clean = ret[~np.isnan(ret)]
    logger.info(f"Fitting HMM on {len(ret_clean)} returns")

    # Fit HMM with K=3
    logger.info("Fitting HMM with K=3...")
    model = GaussianHMM(
        n_components=K,
        covariance_type=config['hmm']['covariance_type'],
        n_iter=config['hmm']['n_iter'],
        tol=1e-8,
        random_state=config['hmm']['seed'],
        init_params='stmc',  # Valid init_params for hmmlearn: 's' startprob, 't' transmat, 'm' means, 'c' covars
        verbose=True
    )

    # Set transition matrix prior for stability
    transmat_prior = config['hmm']['transmat_prior']
    model.transmat_prior = np.ones((K, K)) * transmat_prior

    # Fit model
    X = ret_clean.reshape(-1, 1)
    model.fit(X)

    # Extract parameters
    logger.info("Extracting HMM parameters...")

    # Means and standard deviations
    means = model.means_.flatten().tolist()

    # For diagonal covariance, extract standard deviations
    if config['hmm']['covariance_type'] == 'diag':
        stds = np.sqrt(model.covars_.flatten()).tolist()
    else:
        # For full covariance, extract diagonal
        stds = np.sqrt(np.diag(model.covars_).flatten()).tolist()

    # Transition matrix
    P = model.transmat_.tolist()

    # Compute stationary distribution (left eigenvector of P with eigenvalue 1)
    # Solve pi * P = pi, or equivalently (P^T - I) * pi^T = 0
    P_array = np.array(P)
    eigenvalues, eigenvectors = np.linalg.eig(P_array.T)

    # Find the eigenvector corresponding to eigenvalue 1
    stationary_idx = np.argmin(np.abs(eigenvalues - 1.0))
    pi_unnormalized = np.real(eigenvectors[:, stationary_idx])

    # Normalize to sum to 1
    pi = (pi_unnormalized / pi_unnormalized.sum()).tolist()

    # Initial distribution (for reference)
    initial_pi = model.startprob_.tolist()

    # Create output dictionary
    output = {
        'K': K,
        'mu': means,
        'sigma': stds,
        'P': P,
        'pi': pi,
        'train_start': str(df['date'].min()),
        'train_end': str(df['date'].max()),
        'n_obs': len(ret_clean),
        'seed': config['hmm']['seed'],
        'log_likelihood': float(model.score(X)),  # score() already returns total log-likelihood
        'n_iter': int(getattr(model, 'n_iter_', -1))
    }

    # Validation checks
    logger.info("Running validation checks...")

    # Check transition matrix rows sum to 1
    P_array = np.array(P)
    row_sums = P_array.sum(axis=1)
    assert np.allclose(row_sums, 1.0), f"Transition matrix rows don't sum to 1: {row_sums}"

    # Check all standard deviations are positive
    assert all(s > 0 for s in stds), f"Non-positive standard deviations: {stds}"

    # Check pi sums to 1
    assert np.allclose(sum(pi), 1.0), f"Initial distribution doesn't sum to 1: {sum(pi)}"

    # Check all probabilities are non-negative
    assert all(p >= 0 for p in pi), f"Negative initial probabilities: {pi}"
    assert all(p >= 0 for row in P for p in row), "Negative transition probabilities"

    logger.info("All validation checks passed!")

    # Sort states by mean for consistency (low/medium/high volatility)
    sorted_idx = np.argsort([abs(m) for m in means])  # Sort by absolute mean

    output['mu'] = [means[i] for i in sorted_idx]
    output['sigma'] = [stds[i] for i in sorted_idx]
    output['pi'] = [pi[i] for i in sorted_idx]

    # Reorder transition matrix
    P_sorted = [[P[i][j] for j in sorted_idx] for i in sorted_idx]
    output['P'] = P_sorted

    # Save to JSON
    output_path = Path('outputs/step1_params/hmm_K3.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"Saved HMM parameters to {output_path}")

    # Print summary
    print("\n" + "="*60)
    print("HMM K=3 FIT SUMMARY")
    print("="*60)
    print(f"Number of states: {K}")
    print(f"Training period: {output['train_start']} to {output['train_end']}")
    print(f"Number of observations: {output['n_obs']}")
    print(f"Log-likelihood: {output['log_likelihood']:.2f}")
    print(f"Iterations: {output['n_iter']}")

    print("\nState parameters (sorted by volatility):")
    for i in range(K):
        print(f"  State {i}: μ={output['mu'][i]:7.4f}, σ={output['sigma'][i]:7.4f}")

    print("\nStationary distribution:")
    for i in range(K):
        print(f"  π[{i}] = {output['pi'][i]:.4f}")

    print("\nTransition matrix:")
    for i in range(K):
        row = "  " + " ".join([f"{p:6.4f}" for p in output['P'][i]])
        print(f"  P[{i}→*] = [{row}]")

    print("="*60)

    return output


if __name__ == "__main__":
    try:
        fit_hmm_K3()
    except Exception as e:
        logger.error(f"Error fitting HMM: {e}", exc_info=True)
        raise