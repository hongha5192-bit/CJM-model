#!/usr/bin/env python
"""
Step 3: Lambda scan for CJM K=3 - Fast version with 50 simulations
Finds optimal lambda by maximizing mean BAC across reduced simulation set
"""
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
import yaml
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from jumpmodels.jump import JumpModel
from sklearn.metrics import balanced_accuracy_score
from itertools import permutations
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def best_balanced_accuracy_K3(y_true, y_pred):
    """
    Compute best balanced accuracy for K=3 with permutation fix.
    Tries all 6 permutations of {0, 1, 2}, returns the max BAC.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    best_bac = 0.0

    # Try all 6 permutations of {0, 1, 2}
    for perm in permutations([0, 1, 2]):
        # Create mapping
        mapping = {0: perm[0], 1: perm[1], 2: perm[2]}

        # Apply mapping to predictions
        y_pred_mapped = np.array([mapping[label] for label in y_pred])

        # Calculate BAC for this permutation
        bac = balanced_accuracy_score(y_true, y_pred_mapped)
        best_bac = max(best_bac, bac)

    return best_bac


def fit_cjm_single_sim(sim_path, lam, K, grid_size, n_init, max_iter, tol):
    """Fit CJM to a single simulation and return BAC"""
    try:
        # Load simulation
        if sim_path.suffix == '.parquet':
            df_sim = pd.read_parquet(sim_path)
        else:
            df_sim = pd.read_csv(sim_path)

        y = df_sim['y'].values
        s_true = df_sim['s_true'].values

        # Prepare data for JumpModel (must be 2D)
        X = y.reshape(-1, 1)

        # Fit CJM with mode loss
        cjm = JumpModel(
            n_components=K,
            jump_penalty=lam,
            cont=True,
            mode_loss=True,
            grid_size=grid_size,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            random_state=123,
            verbose=0
        )

        cjm.fit(X)

        # Get predictions
        s_pred = cjm.predict(X)

        # Compute permutation-invariant BAC
        bac = best_balanced_accuracy_K3(s_true, s_pred)

        return bac

    except Exception as e:
        logger.warning(f"Failed to fit {sim_path.name}: {e}")
        return np.nan


def lambda_scan_K3_fast(config_path='configs/daily_K3.yaml', n_sim_subset=50):
    """Perform lambda scan for K=3 with reduced simulations for speed"""

    # Load config
    logger.info("Loading configuration...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    K = config['hmm']['K']
    assert K == 3, f"Expected K=3, got K={K}"

    # Lambda grid (Fast Mode: 21 points)
    log10_min = config['lambda_scan']['lambda_grid']['log10_min']
    log10_max = config['lambda_scan']['lambda_grid']['log10_max']
    n_grid = config['lambda_scan']['lambda_grid']['n_grid']

    lambda_grid = np.logspace(log10_min, log10_max, n_grid)

    logger.info(f"Lambda grid: {n_grid} points from {lambda_grid[0]:.2e} to {lambda_grid[-1]:.2e}")

    # CJM settings
    grid_size = config['lambda_scan']['cjm']['grid_size']
    n_init = config['lambda_scan']['jumpmodels']['n_init']
    max_iter = config['lambda_scan']['jumpmodels']['max_iter']
    tol = config['lambda_scan']['jumpmodels']['tol']
    n_jobs = min(2, config['lambda_scan']['jumpmodels'].get('n_jobs', 2))  # Limit parallelism

    logger.info(f"CJM settings:")
    logger.info(f"  grid_size = {grid_size} (critical for K=3)")
    logger.info(f"  n_init = {n_init}")
    logger.info(f"  max_iter = {max_iter}")
    logger.info(f"  tol = {tol}")
    logger.info(f"  n_jobs = {n_jobs}")

    # Get simulation files - USE ONLY SUBSET
    sim_dir = Path(f'outputs/step2_sim/K{K}')
    sim_files = sorted(sim_dir.glob('sim_T*.parquet'))
    if not sim_files:
        sim_files = sorted(sim_dir.glob('sim_T*.csv'))

    # Use only first n_sim_subset simulations for speed
    sim_files = sim_files[:n_sim_subset]
    N_sim = len(sim_files)

    logger.info(f"Using {N_sim} simulations (subset of {n_sim_subset})")

    if N_sim == 0:
        raise ValueError(f"No simulation files found in {sim_dir}")

    # Create output directory
    output_path = Path('outputs/step3_lambda')
    output_path.mkdir(parents=True, exist_ok=True)

    # Output file for fast version
    scores_path = output_path / f'lambda_scores_K3_fast_{n_sim_subset}.csv'

    results = []

    # Scan over lambda values
    logger.info(f"Starting lambda scan with {N_sim} simulations...")

    for lam_idx, lam in enumerate(lambda_grid):
        logger.info(f"Processing λ={lam:.2e} ({lam_idx+1}/{n_grid})...")

        # Fit CJM to subset of simulations in parallel
        bac_scores = Parallel(n_jobs=n_jobs)(
            delayed(fit_cjm_single_sim)(
                sim_path, lam, K, grid_size, n_init, max_iter, tol
            )
            for sim_path in tqdm(sim_files, desc=f"λ={lam:.2e}", leave=False)
        )

        # Filter out failed fits
        bac_scores = [s for s in bac_scores if not np.isnan(s)]

        if len(bac_scores) == 0:
            logger.warning(f"All fits failed for λ={lam:.2e}")
            continue

        # Calculate statistics
        mean_bac = np.mean(bac_scores)
        std_bac = np.std(bac_scores)

        logger.info(f"  λ={lam:.2e}: BAC = {mean_bac:.4f} ± {std_bac:.4f} (n={len(bac_scores)})")

        # Store result
        result = {
            'K': K,
            'model_type': 'CJM_mode',
            'lambda': lam,
            'T': 1000,
            'mean_BAC': mean_bac,
            'std_BAC': std_bac,
            'n_sim': len(bac_scores),
            'n_sim_subset': n_sim_subset
        }
        results.append(result)

        # Save incrementally
        df_results = pd.DataFrame(results)
        df_results.to_csv(scores_path, index=False)

    # Final results
    df_final = pd.DataFrame(results)

    # Find best lambda
    best_idx = df_final['mean_BAC'].idxmax()
    best_row = df_final.loc[best_idx]
    best_lambda = best_row['lambda']
    best_bac = best_row['mean_BAC']

    logger.info(f"\nBest lambda: {best_lambda:.2e} with mean BAC = {best_bac:.4f}")

    # Save best lambda
    best_lambda_dict = {
        f'K{K}': {
            'CJM_mode': float(best_lambda),
            'n_simulations_used': n_sim_subset,
            'mean_BAC': float(best_bac),
            'std_BAC': float(best_row['std_BAC'])
        }
    }

    best_lambda_path = output_path / f'best_lambda_K3_fast_{n_sim_subset}.json'
    with open(best_lambda_path, 'w') as f:
        json.dump(best_lambda_dict, f, indent=2)

    logger.info(f"Saved best lambda to {best_lambda_path}")

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot BAC vs lambda
    ax.errorbar(
        df_final['lambda'],
        df_final['mean_BAC'],
        yerr=df_final['std_BAC'],
        marker='o',
        markersize=8,
        linewidth=2,
        capsize=5,
        capthick=2,
        label=f'CJM K=3 ({n_sim_subset} simulations)'
    )

    # Mark best lambda
    ax.axvline(x=best_lambda, color='red', linestyle='--', alpha=0.7,
               label=f'Best λ={best_lambda:.2e}')

    # Add horizontal line at random performance
    ax.axhline(y=1/K, color='gray', linestyle=':', alpha=0.5,
               label=f'Random (BAC={1/K:.3f})')

    ax.set_xscale('log')
    ax.set_xlabel('Lambda (λ)', fontsize=12)
    ax.set_ylabel('Mean Balanced Accuracy', fontsize=12)
    ax.set_title(f'Lambda Selection for K={K} (Fast: {n_sim_subset} simulations)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    # Add text box with best result
    textstr = f'Best λ = {best_lambda:.2e}\nMean BAC = {best_bac:.4f} ± {best_row["std_BAC"]:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plot_path = output_path / f'bac_vs_lambda_K3_fast_{n_sim_subset}.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()

    logger.info(f"Saved plot to {plot_path}")

    # Print summary
    print("\n" + "="*60)
    print(f"LAMBDA SCAN SUMMARY (K=3, Fast: {n_sim_subset} simulations)")
    print("="*60)
    print(f"Number of lambda values: {len(df_final)}")
    print(f"Lambda range: [{df_final['lambda'].min():.2e}, {df_final['lambda'].max():.2e}]")
    print(f"Number of simulations used: {n_sim_subset} (out of 256 available)")
    print(f"\nTop 5 lambda values by mean BAC:")
    top_5 = df_final.nlargest(5, 'mean_BAC')[['lambda', 'mean_BAC', 'std_BAC']]
    for _, row in top_5.iterrows():
        print(f"  λ={row['lambda']:.2e}: BAC = {row['mean_BAC']:.4f} ± {row['std_BAC']:.4f}")
    print(f"\nSelected lambda: {best_lambda:.2e}")
    print(f"Mean BAC at best lambda: {best_bac:.4f} ± {best_row['std_BAC']:.4f}")

    # Compare to random baseline
    improvement = (best_bac - 1/K) / (1/K) * 100
    print(f"Improvement over random: {improvement:.1f}%")
    print("="*60)

    return best_lambda


if __name__ == "__main__":
    try:
        lambda_scan_K3_fast(n_sim_subset=50)
    except Exception as e:
        logger.error(f"Error in lambda scan: {e}", exc_info=True)
        raise