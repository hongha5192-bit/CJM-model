#!/usr/bin/env python
"""
Test lambda=1 on 20 feature-based simulations (K=3)
Based on feature-based lambda scan methodology
"""
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
import yaml
from tqdm import tqdm
from jumpmodels.jump import JumpModel
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from itertools import permutations
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def best_balanced_accuracy_K3(y_true, y_pred):
    """
    Compute best balanced accuracy for K=3 with permutation fix
    Tries all 6 permutations of {0, 1, 2}
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    best_bac = 0.0
    best_perm = None

    # Try all 6 permutations of {0, 1, 2}
    for perm in permutations([0, 1, 2]):
        mapping = {0: perm[0], 1: perm[1], 2: perm[2]}
        y_pred_mapped = np.array([mapping[label] for label in y_pred])
        bac = balanced_accuracy_score(y_true, y_pred_mapped)
        if bac > best_bac:
            best_bac = bac
            best_perm = perm

    return best_bac, best_perm


def fit_cjm_to_simulation(sim_file, lam, feature_names):
    """Fit CJM to a single simulation"""
    try:
        # Load simulation
        sim_data = pd.read_parquet(sim_file)

        # Extract features and true labels
        X = sim_data[feature_names].values
        s_true = sim_data['s_true'].values

        # Check for NaNs
        if np.any(np.isnan(X)) or np.any(np.isnan(s_true)):
            logger.warning(f"NaNs found in {sim_file.name}")
            return np.nan, None, None

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit CJM with mode loss
        cjm = JumpModel(
            n_components=3,
            jump_penalty=lam,
            cont=True,
            mode_loss=True,
            grid_size=0.05,  # K=3 optimized
            n_init=3,  # Balance speed/accuracy
            max_iter=200,
            tol=1e-4,
            random_state=42,
            verbose=0
        )

        cjm.fit(X_scaled)
        s_pred = cjm.predict(X_scaled)

        # Compute permutation-invariant BAC
        bac, best_perm = best_balanced_accuracy_K3(s_true, s_pred)

        return bac, best_perm, s_pred

    except Exception as e:
        logger.error(f"Failed to fit CJM on {sim_file.name}: {e}")
        return np.nan, None, None


def main():
    """Test lambda=1 on 20 feature-based simulations"""

    logger.info("="*80)
    logger.info("Testing lambda=1 on 20 K=3 feature-based simulations")
    logger.info("="*80)

    # Load configuration
    config_path = Path('configs/daily_K3.yaml')
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    feature_names = config['features']['cols']
    logger.info(f"Features: {', '.join(feature_names)}")
    logger.info(f"K = {config['hmm']['K']}")

    # Lambda to test
    lam = 1.0
    logger.info(f"Testing lambda = {lam}")

    # CJM settings
    grid_size = config['lambda_scan']['cjm']['grid_size']
    n_init = config['lambda_scan']['jumpmodels']['n_init']
    max_iter = config['lambda_scan']['jumpmodels']['max_iter']
    tol = config['lambda_scan']['jumpmodels']['tol']

    logger.info(f"CJM settings:")
    logger.info(f"  grid_size = {grid_size}")
    logger.info(f"  n_init = {n_init}")
    logger.info(f"  max_iter = {max_iter}")
    logger.info(f"  tol = {tol}")

    # Get simulation files from K3_features directory
    sim_dir = Path('outputs/step2_sim/K3_features')
    if not sim_dir.exists():
        raise FileNotFoundError(f"Simulation directory not found: {sim_dir}")

    sim_files = sorted(sim_dir.glob('sim_T1000_*.parquet'))[:20]

    if len(sim_files) < 20:
        logger.warning(f"Only found {len(sim_files)} simulation files")

    logger.info(f"Processing {len(sim_files)} simulations...")
    logger.info("")

    # Process each simulation
    results = []

    for i, sim_file in enumerate(tqdm(sim_files, desc="Fitting CJM")):
        bac, best_perm, s_pred = fit_cjm_to_simulation(sim_file, lam, feature_names)

        results.append({
            'sim_id': i,
            'sim_file': sim_file.name,
            'lambda': lam,
            'bac': bac,
            'best_permutation': str(best_perm) if best_perm else None
        })

        logger.info(f"Sim {i:2d}: BAC = {bac:.4f}, Best perm = {best_perm}")

    # Create results dataframe
    df_results = pd.DataFrame(results)

    # Compute statistics
    valid_bacs = df_results['bac'].dropna()

    logger.info("")
    logger.info("="*80)
    logger.info("RESULTS SUMMARY")
    logger.info("="*80)
    logger.info(f"Lambda tested: {lam}")
    logger.info(f"Number of simulations: {len(sim_files)}")
    logger.info(f"Successful fits: {len(valid_bacs)}")
    logger.info(f"Failed fits: {len(df_results) - len(valid_bacs)}")
    logger.info("")
    logger.info(f"BAC Statistics:")
    logger.info(f"  Mean:   {valid_bacs.mean():.4f}")
    logger.info(f"  Median: {valid_bacs.median():.4f}")
    logger.info(f"  Std:    {valid_bacs.std():.4f}")
    logger.info(f"  Min:    {valid_bacs.min():.4f}")
    logger.info(f"  Max:    {valid_bacs.max():.4f}")
    logger.info("")
    logger.info("Individual Results:")
    logger.info("-" * 80)
    logger.info(f"{'Sim':>5} {'BAC':>8} {'Best Permutation'}")
    logger.info("-" * 80)
    for _, row in df_results.iterrows():
        bac_str = f"{row['bac']:.4f}" if not np.isnan(row['bac']) else "FAILED"
        perm_str = row['best_permutation'] if row['best_permutation'] else "N/A"
        logger.info(f"{row['sim_id']:>5} {bac_str:>8} {perm_str}")
    logger.info("="*80)

    # Save results
    output_dir = Path('outputs/test_lambda1')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / 'results_lambda1_20sims.csv'
    df_results.to_csv(output_file, index=False)
    logger.info(f"\nResults saved to: {output_file}")

    # Save summary JSON
    summary = {
        'lambda': lam,
        'n_simulations': len(sim_files),
        'n_successful': len(valid_bacs),
        'n_failed': len(df_results) - len(valid_bacs),
        'bac_mean': float(valid_bacs.mean()),
        'bac_median': float(valid_bacs.median()),
        'bac_std': float(valid_bacs.std()),
        'bac_min': float(valid_bacs.min()),
        'bac_max': float(valid_bacs.max()),
        'features': feature_names,
        'grid_size': grid_size,
        'n_init': n_init,
        'max_iter': max_iter
    }

    summary_file = output_dir / 'summary_lambda1_20sims.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary saved to: {summary_file}")

    # Comparison with lambda=5 if available
    lambda5_summary = Path('outputs/test_lambda5/summary_lambda5_10sims.json')
    if lambda5_summary.exists():
        with open(lambda5_summary, 'r') as f:
            lambda5_data = json.load(f)

        logger.info("")
        logger.info("="*80)
        logger.info("COMPARISON: Lambda=1 vs Lambda=5")
        logger.info("="*80)
        logger.info(f"Lambda=1 (20 sims): Mean BAC = {valid_bacs.mean():.4f} ± {valid_bacs.std():.4f}")
        logger.info(f"Lambda=5 (10 sims): Mean BAC = {lambda5_data['bac_mean']:.4f} ± {lambda5_data['bac_std']:.4f}")
        logger.info("="*80)

    return df_results


if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise
