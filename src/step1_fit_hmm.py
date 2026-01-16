"""
Step 1: Fit Gaussian HMM on Vietnam returns
"""
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import sys
sys.path.append('.')
from src.utils_io import load_config, save_json, load_parquet, save_csv, setup_logging, numpy_to_python

logger = setup_logging('step1_fit_hmm')


def compute_stationary_distribution(P):
    """
    Compute stationary distribution of transition matrix P
    Solves pi = pi * P with constraint sum(pi) = 1

    Parameters:
    -----------
    P : np.ndarray
        Transition matrix (K x K)

    Returns:
    --------
    np.ndarray
        Stationary distribution vector
    """
    K = P.shape[0]

    # Method: solve (I - P^T)pi = 0 with normalization constraint
    # We solve the system with the constraint that pi sums to 1
    A = np.eye(K) - P.T
    A = np.vstack([A[:-1], np.ones(K)])  # Replace last row with sum constraint
    b = np.zeros(K)
    b[-1] = 1  # Sum constraint = 1

    pi = np.linalg.solve(A, b)

    # Ensure non-negative and normalized
    pi = np.maximum(pi, 0)
    pi = pi / pi.sum()

    return pi


def fit_hmm(df, config):
    """
    Fit Gaussian HMM on returns

    Parameters:
    -----------
    df : pd.DataFrame
        Prepared data with 'ret' column
    config : dict
        Configuration dictionary

    Returns:
    --------
    dict
        HMM parameters including mu, sigma, P, pi
    """
    # Extract returns
    X = df['ret'].values.reshape(-1, 1)

    logger.info(f"Fitting HMM on {len(X)} returns")

    # Initialize HMM
    K = config['hmm']['K']
    model = GaussianHMM(
        n_components=K,
        covariance_type=config['hmm']['covariance_type'],
        n_iter=config['hmm']['n_iter'],
        random_state=config['hmm']['seed'],
        verbose=True,
        init_params='stmc'  # Initialize all parameters
    )

    # Set transition matrix prior (smoothing)
    model.transmat_prior = config['hmm']['transmat_prior']

    # Fit the model
    logger.info("Fitting HMM...")
    model.fit(X)

    # Extract parameters
    mu = model.means_.flatten()
    sigma = np.sqrt(model.covars_.flatten())
    P = model.transmat_

    # Compute stationary distribution
    pi = compute_stationary_distribution(P)

    # Validation checks
    logger.info("Running validation checks...")

    # Check 1: Rows of P sum to 1
    row_sums = P.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), f"P rows don't sum to 1: {row_sums}"

    # Check 2: All sigma > 0
    assert np.all(sigma > 0), f"Some sigma values are non-positive: {sigma}"

    # Check 3: pi sums to 1 and non-negative
    assert np.allclose(pi.sum(), 1.0, atol=1e-6), f"pi doesn't sum to 1: {pi.sum()}"
    assert np.all(pi >= 0), f"Some pi values are negative: {pi}"

    logger.info("All validation checks passed!")

    # Create results dictionary
    results = {
        'K': K,
        'mu': mu.tolist(),
        'sigma': sigma.tolist(),
        'P': P.tolist(),
        'pi': pi.tolist(),
        'convergence': {
            'n_iter': getattr(model, 'n_iter_', model.monitor_.iter if hasattr(model.monitor_, 'iter') else -1),
            'converged': model.monitor_.converged if hasattr(model.monitor_, 'converged') else True,
            'log_likelihood': float(model.score(X))
        }
    }

    # Log parameters
    logger.info(f"HMM Parameters (K={K}):")
    logger.info(f"  mu: {mu}")
    logger.info(f"  sigma: {sigma}")
    logger.info(f"  P:\n{P}")
    logger.info(f"  pi: {pi}")

    return results


def main():
    """Main execution function"""
    # Load configuration
    config = load_config('configs/daily.yaml')

    # Load prepared data
    df = pd.read_csv('outputs/prepared_data.csv')
    df['date'] = pd.to_datetime(df['date'])

    logger.info(f"Loaded {len(df)} rows of data")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Fit HMM
    hmm_params = fit_hmm(df, config)

    # Save parameters
    output_path = 'outputs/step1_params/hmm_K2.json'
    save_json(numpy_to_python(hmm_params), output_path)

    logger.info(f"HMM parameters saved to {output_path}")


if __name__ == "__main__":
    main()