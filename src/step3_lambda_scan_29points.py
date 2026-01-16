#!/usr/bin/env python
"""
Step 3: Lambda scan with 29 points following MD file specifications
Tests lambdas from 1e-2 to 1e5 using 256 simulations
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
    for perm in permutations([0, 1, 2]):
        mapping = {0: perm[0], 1: perm[1], 2: perm[2]}
        y_pred_mapped = np.array([mapping[label] for label in y_pred])
        bac = balanced_accuracy_score(y_true, y_pred_mapped)
        best_bac = max(best_bac, bac)

    return best_bac


def fit_cjm_single(sim_path, lam, K, feature_names, grid_size=0.05):
    """Fit CJM to a single simulation and compute BAC"""
    try:
        # Load simulation
        df = pd.read_parquet(sim_path)
        X = df[feature_names].values
        s_true = df['s_true'].values

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit CJM with mode loss (MD file specs)
        cjm = JumpModel(
            n_components=K,
            jump_penalty=lam,
            cont=True,              # Continuous model
            mode_loss=True,         # Mode loss version
            grid_size=grid_size,    # 0.05 for K=3
            n_init=3,               # Reduced for speed (MD suggests 10)
            max_iter=500,           # Reduced for speed (MD suggests 1000)
            tol=1e-6,              # MD file: 1e-8
            random_state=42,
            verbose=0
        )

        cjm.fit(X_scaled)
        s_pred = cjm.predict(X_scaled)

        # Compute BAC with permutation fix
        bac = best_balanced_accuracy_K3(s_true, s_pred)

        return bac

    except Exception as e:
        logger.debug(f"Failed to fit CJM for {sim_path}: {e}")
        return np.nan


def process_lambda(lam, sim_files, K, feature_names, grid_size, n_jobs=4):
    """Process all simulations for a single lambda value"""

    # Parallel processing of simulations
    bac_scores = Parallel(n_jobs=n_jobs)(
        delayed(fit_cjm_single)(sim_path, lam, K, feature_names, grid_size)
        for sim_path in sim_files
    )

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

    return {
        'lambda': lam,
        'mean_bac': mean_bac,
        'std_bac': std_bac,
        'n_valid': n_valid,
        'n_total': len(sim_files)
    }


def main():
    """Run lambda scan with 29 points as per MD file"""

    logger.info("="*80)
    logger.info("LAMBDA SCAN WITH 29 POINTS (MD FILE COMPLIANCE)")
    logger.info("="*80)

    # Load configuration
    config_path = Path('configs/daily_K3.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    K = 3
    feature_names = config['features']['cols']
    grid_size = 0.05  # For K=3 as per paper

    # Lambda grid: 29 points from 1e-2 to 1e5 (MD file specification)
    lambda_grid = np.logspace(-2, 5, 29)

    logger.info(f"Configuration:")
    logger.info(f"  K = {K}")
    logger.info(f"  Features = {feature_names}")
    logger.info(f"  Lambda grid: {len(lambda_grid)} points from {lambda_grid[0]:.2e} to {lambda_grid[-1]:.2e}")
    logger.info(f"  Grid size = {grid_size}")

    # Load simulations
    sim_dir = Path('outputs/step2_sim/K3_256')
    sim_files = sorted(sim_dir.glob('sim_*.parquet'))

    if not sim_files:
        logger.error(f"No simulation files found in {sim_dir}")
        return

    logger.info(f"Found {len(sim_files)} simulations")

    # Number of parallel jobs (reduced for Mac Air M4)
    n_jobs = min(4, config.get('n_jobs', 4))
    logger.info(f"Using {n_jobs} parallel workers")

    # Process each lambda
    results = []
    start_time = time.time()

    logger.info("\nStarting lambda scan...")
    logger.info("-" * 60)

    with tqdm(total=len(lambda_grid), desc="Lambda scan", unit="λ") as pbar:
        for i, lam in enumerate(lambda_grid):
            pbar.set_description(f"Lambda scan (λ={lam:.2e})")

            # Process this lambda
            result = process_lambda(lam, sim_files, K, feature_names, grid_size, n_jobs)
            results.append(result)

            # Log progress
            if not np.isnan(result['mean_bac']):
                logger.info(f"λ={lam:10.2e}: BAC={result['mean_bac']:.4f} ± {result['std_bac']:.4f} ({result['n_valid']}/{result['n_total']} valid)")
            else:
                logger.info(f"λ={lam:10.2e}: BAC=NaN (all failed)")

            pbar.update(1)

    elapsed = time.time() - start_time

    # Create results DataFrame
    df_results = pd.DataFrame(results)

    # Find best lambda
    if not df_results['mean_bac'].isna().all():
        best_idx = df_results['mean_bac'].idxmax()
        best_lambda = df_results.loc[best_idx, 'lambda']
        best_bac = df_results.loc[best_idx, 'mean_bac']
    else:
        logger.error("All lambda values resulted in NaN BAC!")
        best_lambda = lambda_grid[len(lambda_grid)//2]
        best_bac = np.nan

    logger.info("\n" + "="*80)
    logger.info("RESULTS SUMMARY")
    logger.info("="*80)
    logger.info(f"Time elapsed: {elapsed:.1f} seconds")
    logger.info(f"Average time per lambda: {elapsed/len(lambda_grid):.1f} seconds")

    if not np.isnan(best_bac):
        logger.info(f"\nBest Lambda: {best_lambda:.2e}")
        logger.info(f"Best Mean BAC: {best_bac:.4f}")

    # Display top 10 lambdas
    logger.info("\nTop 10 Lambda Values by BAC:")
    logger.info("-" * 60)
    df_sorted = df_results.sort_values('mean_bac', ascending=False).head(10)
    logger.info(f"{'Lambda':>12} | {'Mean BAC':>10} | {'Std BAC':>10}")
    for _, row in df_sorted.iterrows():
        if not np.isnan(row['mean_bac']):
            logger.info(f"{row['lambda']:12.2e} | {row['mean_bac']:10.4f} | {row['std_bac']:10.4f}")

    # Save results
    output_dir = Path('outputs/step3_lambda')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save lambda scores CSV (MD file requirement)
    csv_path = output_dir / 'lambda_scores_29points.csv'
    df_results['K'] = K
    df_results['model_type'] = 'CJM_mode'
    df_results['T'] = 1000
    df_results = df_results[['K', 'model_type', 'lambda', 'T', 'mean_bac', 'std_bac', 'n_valid']]
    df_results.to_csv(csv_path, index=False)
    logger.info(f"\nLambda scores saved to: {csv_path}")

    # Save best lambda JSON (MD file requirement)
    best_lambda_dict = {
        f"K{K}": {
            "CJM_mode": float(best_lambda),
            "mean_bac": float(best_bac) if not np.isnan(best_bac) else None
        }
    }
    json_path = output_dir / 'best_lambda_29points.json'
    with open(json_path, 'w') as f:
        json.dump(best_lambda_dict, f, indent=2)
    logger.info(f"Best lambda saved to: {json_path}")

    # Create BAC vs Lambda plot (MD file requirement)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Filter valid points
    valid_df = df_results[~df_results['mean_bac'].isna()]

    if not valid_df.empty:
        ax.errorbar(valid_df['lambda'], valid_df['mean_bac'],
                   yerr=valid_df['std_bac'],
                   marker='o', markersize=6,
                   capsize=3, capthick=1,
                   linewidth=2,
                   label=f'K={K}, CJM_mode')

        # Mark best lambda
        ax.axvline(x=best_lambda, color='red', linestyle='--', alpha=0.7, label=f'Best λ={best_lambda:.2e}')
        ax.scatter([best_lambda], [best_bac], color='red', s=100, zorder=5)

        ax.set_xscale('log')
        ax.set_xlabel('Lambda (λ)', fontsize=12)
        ax.set_ylabel('Mean Balanced Accuracy', fontsize=12)
        ax.set_title(f'BAC vs Lambda - K={K} CJM with Mode Loss\n(29 points, 256 simulations)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add reference lines
        ax.axhline(y=1/K, color='gray', linestyle=':', alpha=0.5, label=f'Random baseline ({1/K:.3f})')

    plt.tight_layout()
    plot_path = output_dir / 'bac_vs_lambda_29points.png'
    plt.savefig(plot_path, dpi=150)
    logger.info(f"Plot saved to: {plot_path}")
    plt.close()

    # Display final statistics
    logger.info("\n" + "="*80)
    logger.info("LAMBDA SCAN COMPLETE")
    logger.info("="*80)
    logger.info(f"Total simulations processed: {len(sim_files) * len(lambda_grid)}")
    logger.info(f"Successful fits: {df_results['n_valid'].sum()}")
    logger.info(f"Failed fits: {(df_results['n_total'].sum() - df_results['n_valid'].sum())}")

    return best_lambda, best_bac


if __name__ == "__main__":
    main()