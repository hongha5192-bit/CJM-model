#!/usr/bin/env python
"""
Step 1: Fit Gaussian HMM with K=3 using ALL FEATURES
Modified to use the same 6 features as CJM, not just returns
"""
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
import yaml
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def fit_hmm_K3_features(config_path='configs/daily_K3.yaml'):
    """Fit HMM with K=3 states using all features"""

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

    # Get features (same as CJM)
    feature_cols = config['features']['cols']
    logger.info(f"Using features: {feature_cols}")

    # Extract feature matrix
    X_raw = df[feature_cols].values
    ret = df['ret'].values  # Keep returns for reference

    # Remove NaN rows
    valid_idx = ~np.any(np.isnan(X_raw), axis=1) & ~np.isnan(ret)
    X_clean = X_raw[valid_idx]
    ret_clean = ret[valid_idx]

    logger.info(f"Valid observations: {len(X_clean)}")

    # Standardize features for HMM
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    logger.info(f"Feature matrix shape: {X_scaled.shape}")
    logger.info(f"Features: {feature_cols}")

    # Fit HMM with K=3 on standardized features
    logger.info("Fitting HMM with K=3 on feature matrix...")
    model = GaussianHMM(
        n_components=K,
        covariance_type='full',  # Use full covariance for multivariate
        n_iter=config['hmm']['n_iter'],
        tol=1e-8,
        random_state=config['hmm']['seed'],
        init_params='stmc',
        verbose=True
    )

    # Set transition matrix prior for stability
    transmat_prior = config['hmm']['transmat_prior']
    model.transmat_prior = np.ones((K, K)) * transmat_prior

    # Fit model on feature matrix
    model.fit(X_scaled)

    # Extract parameters
    logger.info("Extracting HMM parameters...")

    # Means for each feature and state
    means_scaled = model.means_  # Shape: (K, n_features)

    # Transform means back to original scale
    means_original = scaler.inverse_transform(means_scaled)

    # Covariances
    covars = model.covars_  # Shape: (K, n_features, n_features)

    # For summary, extract standard deviations (diagonal of covariance)
    stds_scaled = np.sqrt(np.diagonal(covars, axis1=1, axis2=2))

    # Transition matrix
    P = model.transmat_.tolist()

    # Compute stationary distribution
    P_array = np.array(P)
    eigenvalues, eigenvectors = np.linalg.eig(P_array.T)
    stationary_idx = np.argmin(np.abs(eigenvalues - 1.0))
    pi_unnormalized = np.real(eigenvectors[:, stationary_idx])
    pi = (pi_unnormalized / pi_unnormalized.sum()).tolist()

    # Get state assignments for return statistics
    states = model.predict(X_scaled)

    # Calculate return statistics per state
    return_stats = {}
    for state in range(K):
        state_mask = states == state
        state_returns = ret_clean[state_mask]
        return_stats[state] = {
            'mean_return': float(np.mean(state_returns)),
            'std_return': float(np.std(state_returns)),
            'count': int(np.sum(state_mask))
        }

    # Create output dictionary
    output = {
        'K': K,
        'features': feature_cols,
        'n_features': len(feature_cols),
        'means': {
            f'state_{k}': {
                feat: float(means_original[k, i])
                for i, feat in enumerate(feature_cols)
            }
            for k in range(K)
        },
        'means_matrix': means_original.tolist(),
        'stds_scaled': stds_scaled.tolist(),
        'covars_shape': list(covars.shape),
        'P': P,
        'pi': pi,
        'return_stats': return_stats,
        'scaler_params': {
            'mean': scaler.mean_.tolist(),
            'scale': scaler.scale_.tolist()
        },
        'train_start': str(df['date'].min()),
        'train_end': str(df['date'].max()),
        'n_obs': len(X_clean),
        'seed': config['hmm']['seed'],
        'log_likelihood': float(model.score(X_scaled)),
        'n_iter': int(getattr(model, 'n_iter_', -1))
    }

    # Sort states by return volatility for consistency
    sorted_idx = np.argsort([return_stats[i]['std_return'] for i in range(K)])

    # Reorder all state-dependent parameters
    output['means'] = {
        f'state_{i}': output['means'][f'state_{sorted_idx[i]}']
        for i in range(K)
    }
    output['means_matrix'] = [output['means_matrix'][i] for i in sorted_idx]
    output['stds_scaled'] = [output['stds_scaled'][i] for i in sorted_idx]
    output['return_stats'] = {
        i: return_stats[sorted_idx[i]]
        for i in range(K)
    }
    output['pi'] = [pi[i] for i in sorted_idx]

    # Reorder transition matrix
    P_sorted = [[P[i][j] for j in sorted_idx] for i in sorted_idx]
    output['P'] = P_sorted

    # Validation checks
    logger.info("Running validation checks...")

    # Check transition matrix rows sum to 1
    P_array = np.array(output['P'])
    row_sums = P_array.sum(axis=1)
    assert np.allclose(row_sums, 1.0), f"Transition matrix rows don't sum to 1: {row_sums}"

    # Check pi sums to 1
    assert np.allclose(sum(output['pi']), 1.0), f"Stationary distribution doesn't sum to 1"

    logger.info("All validation checks passed!")

    # Save to JSON
    output_path = Path('outputs/step1_params/hmm_K3_features.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save full output
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"Saved HMM parameters to {output_path}")

    # Save state assignments with reordered states
    state_mapping = {sorted_idx[i]: i for i in range(K)}
    states_reordered = np.array([state_mapping[s] for s in states])

    states_df = pd.DataFrame({
        'date': df[valid_idx]['date'].values,
        'hmm_state': states_reordered,
        'ret': ret_clean
    })

    states_path = output_path.parent / 'hmm_states_K3.csv'
    states_df.to_csv(states_path, index=False)
    logger.info(f"Saved HMM state assignments to {states_path}")

    # Also save simplified version for simulation
    simple_output = {
        'K': K,
        'n_features': len(feature_cols),
        'features': feature_cols,
        'means_matrix': output['means_matrix'],
        'covars': covars.tolist(),  # Full covariance matrices
        'P': output['P'],
        'pi': output['pi'],
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist()
    }

    simple_path = Path('outputs/step1_params/hmm_K3_features_sim.json')
    with open(simple_path, 'w') as f:
        json.dump(simple_output, f, indent=2)

    # Print summary
    print("\n" + "="*70)
    print("HMM K=3 FIT SUMMARY (WITH ALL FEATURES)")
    print("="*70)
    print(f"Number of states: {K}")
    print(f"Number of features: {len(feature_cols)}")
    print(f"Features: {', '.join(feature_cols)}")
    print(f"Training period: {output['train_start']} to {output['train_end']}")
    print(f"Number of observations: {output['n_obs']}")
    print(f"Log-likelihood: {output['log_likelihood']:.2f}")

    print("\nState Return Statistics (sorted by volatility):")
    for i in range(K):
        stats = output['return_stats'][i]
        print(f"  State {i}: μ_ret={stats['mean_return']:7.4f}, σ_ret={stats['std_return']:7.4f}, n={stats['count']}")

    print("\nStationary distribution:")
    for i in range(K):
        print(f"  π[{i}] = {output['pi'][i]:.4f}")

    print("\nTransition matrix:")
    for i in range(K):
        row = " ".join([f"{p:6.4f}" for p in output['P'][i]])
        print(f"  P[{i}→*] = [{row}]")

    print("\nFeature Means by State:")
    for state in range(K):
        print(f"\n  State {state} (σ_ret={output['return_stats'][state]['std_return']:.4f}):")
        for feat in feature_cols[:3]:  # Show first 3 features
            val = output['means'][f'state_{state}'][feat]
            print(f"    {feat:10s}: {val:8.3f}")

    print("="*70)

    return output


if __name__ == "__main__":
    try:
        fit_hmm_K3_features()
    except Exception as e:
        logger.error(f"Error fitting HMM: {e}", exc_info=True)
        raise