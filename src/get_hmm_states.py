#!/usr/bin/env python3
"""
Get HMM state assignments for real data
"""
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
import yaml
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_hmm_states():
    """Get HMM state assignments from fitted parameters"""

    logger.info("="*80)
    logger.info("GETTING HMM STATE ASSIGNMENTS")
    logger.info("="*80)

    # Load HMM parameters
    logger.info("\nLoading HMM parameters...")
    params_path = Path('outputs/step1_params/hmm_K3_features.json')
    with open(params_path, 'r') as f:
        params = json.load(f)

    K = params['K']
    logger.info(f"K = {K}")

    # Load prepared data
    logger.info("Loading data...")
    data_path = Path('outputs/prepared_data.csv')
    df = pd.read_csv(data_path, parse_dates=['date'])
    logger.info(f"Loaded {len(df)} rows")

    # Get features
    feature_cols = params['features']
    logger.info(f"Features: {feature_cols}")

    # Extract feature matrix
    X_raw = df[feature_cols].values
    ret = df['ret'].values
    dates = df['date'].values

    # Remove NaN rows
    valid_idx = ~np.any(np.isnan(X_raw), axis=1) & ~np.isnan(ret)
    X_clean = X_raw[valid_idx]
    ret_clean = ret[valid_idx]
    dates_clean = dates[valid_idx]

    logger.info(f"Valid observations: {len(X_clean)}")

    # Standardize features using saved scaler params
    scaler = StandardScaler()
    scaler.mean_ = np.array(params['scaler_params']['mean'])
    scaler.scale_ = np.array(params['scaler_params']['scale'])
    X_scaled = scaler.transform(X_clean)

    # Reconstruct HMM model
    logger.info("Reconstructing HMM model...")
    model = GaussianHMM(
        n_components=K,
        covariance_type='full',
        random_state=123
    )

    # Set parameters
    model.startprob_ = np.array(params['pi'])
    model.transmat_ = np.array(params['P'])
    model.means_ = np.array(params['means_matrix'])

    # Reconstruct covariances from stds_scaled
    # Note: The saved params have scaled stds, we need to reconstruct the full covariance
    stds_scaled = np.array(params['stds_scaled'])
    n_features = len(feature_cols)

    # Create diagonal covariance matrices from scaled stds
    covars = np.zeros((K, n_features, n_features))
    for k in range(K):
        covars[k] = np.diag(stds_scaled[k] ** 2)

    model.covars_ = covars

    logger.info("Model parameters set")

    # Predict states using Viterbi algorithm
    logger.info("Predicting states...")
    states = model.predict(X_scaled)
    probas = model.predict_proba(X_scaled)

    logger.info("States predicted")

    # Create output dataframe
    output_df = pd.DataFrame({
        'date': dates_clean,
        'ret': ret_clean,
        'hmm_state': states,
        'proba_0': probas[:, 0],
        'proba_1': probas[:, 1],
        'proba_2': probas[:, 2]
    })

    # Add full data columns
    df_clean = df[valid_idx].reset_index(drop=True)
    for col in df.columns:
        if col not in output_df.columns and col != 'date':
            output_df[col] = df_clean[col].values

    # Save
    output_path = Path('outputs/step1_params/hmm_states_K3.csv')
    output_df.to_csv(output_path, index=False)
    logger.info(f"\n✓ Saved HMM state assignments to: {output_path}")

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("HMM STATE DISTRIBUTION")
    logger.info("="*80)

    state_labels = {
        0: "State 0",
        1: "State 1",
        2: "State 2"
    }

    for state in range(K):
        count = (states == state).sum()
        pct = count / len(states) * 100
        logger.info(f"{state_labels[state]}: {count:4d} days ({pct:5.2f}%)")

    # Compare with return statistics
    logger.info("\n" + "="*80)
    logger.info("RETURN STATISTICS BY HMM STATE")
    logger.info("="*80)

    for state in range(K):
        state_returns = ret_clean[states == state]
        logger.info(f"\n{state_labels[state]}:")
        logger.info(f"  Mean return: {state_returns.mean():.6f}")
        logger.info(f"  Std return:  {state_returns.std():.6f}")
        logger.info(f"  Count: {len(state_returns)}")

    logger.info("\n" + "="*80)

    return output_df

if __name__ == "__main__":
    get_hmm_states()
