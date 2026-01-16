#!/usr/bin/env python
"""
Step 3: Lambda scan with 1 simulation, lambda=1, max_iter=500
Based on step3_lambda_scan_7points.py but modified for single sim testing
"""
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
import yaml
from tqdm import tqdm
from joblib import Parallel, delayed
from jumpmodels.jump import JumpModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from itertools import permutations
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def best_balanced_accuracy_K3(y_true, y_pred):
    """Compute best BAC for K=3 with permutation fix"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    best_bac = 0.0
    best_perm = None

    for perm in permutations([0, 1, 2]):
        mapping = {0: perm[0], 1: perm[1], 2: perm[2]}
        y_pred_mapped = np.array([mapping[label] for label in y_pred])
        bac = balanced_accuracy_score(y_true, y_pred_mapped)
        if bac > best_bac:
            best_bac = bac
            best_perm = perm

    return best_bac, best_perm


def fit_cjm_single(sim_path, lam, K, feature_names, grid_size=0.05, max_iter=500):
    """Fit CJM to a single simulation and compute BAC"""
    try:
        # Load simulation
        df = pd.read_parquet(sim_path)
        X = df[feature_names].values
        s_true = df['s_true'].values

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit CJM with mode loss - HIGH ITERATIONS VERSION
        logger.info(f"Fitting CJM with λ={lam}, max_iter={max_iter}...")

        cjm = JumpModel(
            n_components=K,
            jump_penalty=lam,
            cont=True,              # Continuous model
            mode_loss=True,         # Mode loss version
            grid_size=grid_size,    # 0.05 for K=3
            n_init=2,               # As per original script
            max_iter=max_iter,      # MODIFIED: 500 (was 300)
            tol=1e-5,              # As per original
            random_state=42,
            verbose=1               # MODIFIED: Show convergence info
        )

        cjm.fit(X_scaled)
        s_pred = cjm.predict(X_scaled)

        # Compute BAC with permutation fix
        bac, best_perm = best_balanced_accuracy_K3(s_true, s_pred)

        return bac, best_perm

    except Exception as e:
        logger.error(f"Failed to fit CJM for {sim_path}: {e}")
        return np.nan, None


def process_lambda_batch(lambda_batch, sim_files, K, feature_names, grid_size, max_iter, n_jobs=1):
    """Process a batch of lambdas"""
    results = []

    for lam in lambda_batch:
        # Process simulations for this lambda
        logger.info(f"\nProcessing λ={lam:.2e} with {len(sim_files)} simulation(s)...")

        # Use sequential processing for single simulation
        if len(sim_files) == 1:
            bac, best_perm = fit_cjm_single(sim_files[0], lam, K, feature_names, grid_size, max_iter)
            bac_scores = [bac] if not np.isnan(bac) else []
            best_perms = [best_perm] if not np.isnan(bac) else []
        else:
            # Parallel processing for multiple simulations
            results_list = Parallel(n_jobs=n_jobs)(
                delayed(fit_cjm_single)(sim_path, lam, K, feature_names, grid_size, max_iter)
                for sim_path in sim_files
            )
            bac_scores = [r[0] for r in results_list if not np.isnan(r[0])]
            best_perms = [r[1] for r in results_list if not np.isnan(r[0])]

        # Filter out NaN values
        valid_scores = [s for s in bac_scores if not np.isnan(s)]

        if len(valid_scores) > 0:
            mean_bac = np.mean(valid_scores)
            std_bac = np.std(valid_scores)
            n_valid = len(valid_scores)
        else:
            mean_bac = np.nan
            std_bac = np.nan
            n_valid = 0

        result = {
            'lambda': lam,
            'mean_bac': mean_bac,
            'std_bac': std_bac,
            'n_valid': n_valid,
            'n_total': len(sim_files),
            'best_permutations': [str(p) for p in best_perms] if best_perms else []
        }

        results.append(result)

        # Log progress
        if not np.isnan(mean_bac):
            logger.info(f"✓ λ={lam:10.2e}: BAC={mean_bac:.4f} ± {std_bac:.4f} ({n_valid}/{len(sim_files)} valid)")
            if best_perms:
                logger.info(f"  Best permutation(s): {best_perms}")
        else:
            logger.info(f"✗ λ={lam:10.2e}: BAC=NaN (all failed)")

    return results


def main():
    """Run lambda scan with 1 simulation, lambda=1, max_iter=500"""

    logger.info("="*80)
    logger.info("LAMBDA TEST: 1 SIMULATION, λ=1, MAX_ITER=500")
    logger.info("="*80)

    # Load configuration
    config_path = Path('configs/daily_K3.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    K = 3
    feature_names = config['features']['cols']
    grid_size = 0.05  # For K=3 as per paper

    # MODIFIED: Single lambda value = 1
    lambda_grid = np.array([1.0])

    # MODIFIED: max_iter = 500
    max_iter = 500

    logger.info(f"Configuration:")
    logger.info(f"  K = {K}")
    logger.info(f"  Features = {feature_names}")
    logger.info(f"  Lambda values: {lambda_grid}")
    logger.info(f"  Grid size = {grid_size}")
    logger.info(f"  CJM settings: n_init=2, max_iter={max_iter}, tol=1e-5")

    # Load corrected simulations - MODIFIED: use only first 1
    sim_dir = Path('outputs/step2_sim/K3_features')
    all_sim_files = sorted(sim_dir.glob('sim_T1000_*.parquet'))

    # MODIFIED: Use only 1 simulation
    n_sims_to_use = 1
    sim_files = all_sim_files[:n_sims_to_use]

    if not sim_files:
        logger.error(f"No simulation files found in {sim_dir}")
        return

    logger.info(f"Using {len(sim_files)} simulation (out of {len(all_sim_files)} available)")
    logger.info(f"Selected file: {sim_files[0].name}")

    # Verify simulation data
    df_test = pd.read_parquet(sim_files[0])
    logger.info("\nSimulation data info:")
    logger.info(f"  Shape: {df_test.shape}")
    logger.info(f"  Columns: {list(df_test.columns)}")
    logger.info(f"\nFeature statistics:")
    for feat in feature_names:
        logger.info(f"  {feat:10s}: mean={df_test[feat].mean():6.2f}, std={df_test[feat].std():6.2f}, "
                   f"min={df_test[feat].min():6.2f}, max={df_test[feat].max():6.2f}")

    # Check true states distribution
    state_counts = df_test['s_true'].value_counts().sort_index()
    logger.info(f"\nTrue state distribution:")
    for state, count in state_counts.items():
        logger.info(f"  State {state}: {count:4d} samples ({count/len(df_test)*100:.1f}%)")

    # Number of parallel jobs (not needed for 1 simulation)
    n_jobs = 1
    logger.info(f"\nUsing sequential processing (n_jobs={n_jobs})")

    # Calculate total fits
    total_fits = len(lambda_grid) * len(sim_files)
    logger.info(f"Total CJM fits to perform: {total_fits}")

    # Process lambdas
    results = []
    start_time = time.time()

    logger.info("\nStarting lambda test...")
    logger.info("-" * 80)

    batch_results = process_lambda_batch(
        lambda_grid, sim_files, K, feature_names, grid_size, max_iter, n_jobs
    )
    results.extend(batch_results)

    elapsed = time.time() - start_time

    # Create results DataFrame
    df_results = pd.DataFrame(results)

    # Get results for lambda=1
    lambda1_result = df_results[df_results['lambda'] == 1.0].iloc[0]
    best_lambda = lambda1_result['lambda']
    best_bac = lambda1_result['mean_bac']
    best_std = lambda1_result['std_bac']

    logger.info("\n" + "="*80)
    logger.info("RESULTS SUMMARY")
    logger.info("="*80)
    logger.info(f"Time elapsed: {elapsed:.2f} seconds")
    logger.info(f"Lambda tested: {best_lambda}")

    if not np.isnan(best_bac):
        logger.info(f"\n🎯 Lambda = {best_lambda}")
        logger.info(f"   BAC: {best_bac:.6f} ± {best_std:.6f}")
        logger.info(f"   Result: {'PERFECT' if best_bac == 1.0 else 'EXCELLENT' if best_bac > 0.99 else 'GOOD' if best_bac > 0.90 else 'MODERATE'}")
    else:
        logger.error("   BAC: NaN (fit failed)")

    # Display permutation used
    if lambda1_result['best_permutations']:
        logger.info(f"   Best permutation: {lambda1_result['best_permutations'][0]}")

    # Save results
    output_dir = Path('outputs/test_lambda1_highiter')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results JSON
    result_dict = {
        "lambda": float(best_lambda),
        "bac": float(best_bac) if not np.isnan(best_bac) else None,
        "std_bac": float(best_std) if not np.isnan(best_std) else None,
        "n_simulations": len(sim_files),
        "simulation_file": sim_files[0].name,
        "max_iter": max_iter,
        "grid_size": grid_size,
        "n_init": 2,
        "tol": 1e-5,
        "features": feature_names,
        "K": K,
        "elapsed_time_seconds": elapsed,
        "best_permutation": lambda1_result['best_permutations'][0] if lambda1_result['best_permutations'] else None
    }

    json_path = output_dir / 'result_lambda1_1sim_highiter.json'
    with open(json_path, 'w') as f:
        json.dump(result_dict, f, indent=2)
    logger.info(f"\n📁 Results saved to: {json_path}")

    # Save CSV for consistency
    csv_path = output_dir / 'result_lambda1_1sim_highiter.csv'
    df_results['K'] = K
    df_results['model_type'] = 'CJM_mode'
    df_results['T'] = 1000
    df_results['max_iter'] = max_iter
    df_results = df_results[['K', 'model_type', 'lambda', 'T', 'max_iter', 'mean_bac', 'std_bac', 'n_valid', 'n_total']]
    df_results.to_csv(csv_path, index=False)
    logger.info(f"📊 CSV saved to: {csv_path}")

    # Display final comparison
    logger.info("\n" + "="*80)
    logger.info("COMPARISON WITH PREVIOUS TESTS")
    logger.info("="*80)

    # Try to load previous results for comparison
    try:
        lambda1_20sims = Path('outputs/test_lambda1/summary_lambda1_20sims.json')
        lambda5_10sims = Path('outputs/test_lambda5/summary_lambda5_10sims.json')

        if lambda1_20sims.exists():
            with open(lambda1_20sims, 'r') as f:
                data_lambda1 = json.load(f)
            logger.info(f"λ=1 (20 sims, max_iter=200): BAC = {data_lambda1['bac_mean']:.4f} ± {data_lambda1['bac_std']:.4f}")

        if lambda5_10sims.exists():
            with open(lambda5_10sims, 'r') as f:
                data_lambda5 = json.load(f)
            logger.info(f"λ=5 (10 sims, max_iter=200): BAC = {data_lambda5['bac_mean']:.4f} ± {data_lambda5['bac_std']:.4f}")

        if not np.isnan(best_bac):
            logger.info(f"λ=1 ( 1 sim,  max_iter=500): BAC = {best_bac:.4f} ± {best_std:.4f} ← CURRENT")

    except Exception as e:
        logger.debug(f"Could not load previous results: {e}")

    logger.info("="*80)
    logger.info("TEST COMPLETE")
    logger.info("="*80)

    return best_lambda, best_bac


if __name__ == "__main__":
    main()
